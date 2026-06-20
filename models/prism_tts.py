from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig

from models.llama_backbone import LlamaBackbone
from utils import model_utils as MU

from models.generation import (
    causal as causal_generation,
    parallel as parallel_generation,
    parallel_stable as parallel_stable_generation,
    e2e as e2e_generation,
)


class PrismTTS(nn.Module):
    """
    Prism-TTS masked reconstruction model.

    Training sequence layout (per sample):
    text_target -> EOT -> speech_target -> EOS

    Speech is flattened block-wise. Each speech block has (N + 1) streams:
    N discrete streams + 1 continuous stream.
    """

    def __init__(
        self,
        llama_config: LlamaConfig,
        num_discrete_tokens: int,
        discrete_vocab_size: int,
        continuous_latent_size: int,
        continuous_loss_weight: float = 1.0,
        discrete_regular_token_loss_weight: float = 1.0,
        discrete_special_token_loss_weight: float = 1.0,
        parallel_sample_steps: int = 64,
    ):
        """Initialize model modules, embeddings, loss weights, and special-token ids."""
        super().__init__()
        if num_discrete_tokens < 1:
            raise ValueError("num_discrete_tokens must be at least 1.")
        if discrete_vocab_size < 1:
            raise ValueError("discrete_vocab_size must be at least 1.")
        if continuous_latent_size < 1:
            raise ValueError("continuous_latent_size must be at least 1.")
        if continuous_loss_weight < 0.0:
            raise ValueError("continuous_loss_weight must be >= 0.")
        if discrete_regular_token_loss_weight < 0.0:
            raise ValueError("discrete_regular_token_loss_weight must be >= 0.")
        if discrete_special_token_loss_weight < 0.0:
            raise ValueError("discrete_special_token_loss_weight must be >= 0.")
        if (
            discrete_regular_token_loss_weight == 0.0
            and discrete_special_token_loss_weight == 0.0
        ):
            raise ValueError(
                "At least one of discrete_regular_token_loss_weight or "
                "discrete_special_token_loss_weight must be > 0."
            )
        if parallel_sample_steps < 1:
            raise ValueError("parallel_sample_steps must be at least 1.")

        self.hidden_size = int(llama_config.hidden_size)
        self.num_discrete_tokens = int(num_discrete_tokens)
        self.discrete_vocab_size = int(discrete_vocab_size)
        self.continuous_latent_size = int(continuous_latent_size)
        self.speech_block_size = self.num_discrete_tokens + 1

        self.continuous_loss_weight = float(continuous_loss_weight)
        self.discrete_regular_token_loss_weight = float(discrete_regular_token_loss_weight)
        self.discrete_special_token_loss_weight = float(discrete_special_token_loss_weight)
        self.parallel_sample_steps = int(parallel_sample_steps)

        self.backbone = LlamaBackbone(llama_config)
        self.discrete_embeddings = nn.ModuleList(
            nn.Embedding(self.discrete_vocab_size, self.hidden_size)
            for _ in range(self.num_discrete_tokens)
        )
        self.discrete_lm_heads = nn.ModuleList(
            nn.Linear(self.hidden_size, self.discrete_vocab_size, bias=False)
            for _ in range(self.num_discrete_tokens)
        )
        self.continuous_proj = nn.Linear(self.continuous_latent_size, self.hidden_size)
        self.continuous_prior_head = nn.Linear(self.hidden_size, self.continuous_latent_size)

        self.token_type_embeddings = nn.Parameter(torch.empty(3, self.hidden_size))
        self.speech_stream_embeddings = nn.Parameter(
            torch.empty(self.speech_block_size, self.hidden_size)
        )
        self.active_stream_count_embedding = nn.Embedding(
            self.num_discrete_tokens + 1,
            self.hidden_size,
        )
        self.masked_discrete_embeddings = nn.Parameter(
            torch.empty(self.num_discrete_tokens, self.hidden_size)
        )
        self.masked_continuous_embedding = nn.Parameter(torch.empty(self.hidden_size))

        vocab_size = int(self.backbone.config.vocab_size)
        pad_candidate = self.backbone.config.pad_token_id
        self.pad_token_id = 0
        if pad_candidate is not None and 0 <= int(pad_candidate) < vocab_size:
            self.pad_token_id = int(pad_candidate)

        eos_candidate = self.backbone.config.eos_token_id
        self.eos_token_id = 0
        if eos_candidate is not None and 0 <= int(eos_candidate) < vocab_size:
            self.eos_token_id = int(eos_candidate)
        # Shared layout: EOT is typically EOS - 1.
        self.eot_token_id = max(0, self.eos_token_id - 1)

        self.training_special_discrete_token_ids = MU.infer_special_discrete_token_ids(
            MU.resolve_generation_discrete_eos_token_id(
                None,
                backbone_eos_token_id=self.backbone.config.eos_token_id,
                discrete_vocab_size=self.discrete_vocab_size,
            ),
            backbone_eos_token_id=self.backbone.config.eos_token_id,
            backbone_pad_token_id=self.backbone.config.pad_token_id,
            discrete_vocab_size=self.discrete_vocab_size,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize trainable parameters using the configured initializer range."""
        std = getattr(self.backbone.config, "initializer_range", 0.02)
        for embedding in self.discrete_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=std)
        for lm_head in self.discrete_lm_heads:
            nn.init.normal_(lm_head.weight, mean=0.0, std=std)
        nn.init.normal_(self.continuous_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.continuous_prior_head.weight, mean=0.0, std=std)
        nn.init.normal_(self.token_type_embeddings, mean=0.0, std=std)
        nn.init.normal_(self.speech_stream_embeddings, mean=0.0, std=std)
        nn.init.zeros_(self.active_stream_count_embedding.weight)
        nn.init.normal_(self.masked_discrete_embeddings, mean=0.0, std=std)
        nn.init.normal_(self.masked_continuous_embedding, mean=0.0, std=std)
        if self.continuous_proj.bias is not None:
            nn.init.zeros_(self.continuous_proj.bias)
        if self.continuous_prior_head.bias is not None:
            nn.init.zeros_(self.continuous_prior_head.bias)

    @property
    def text_embedding(self) -> nn.Embedding:
        """Return the shared embedding table used for text tokens."""
        return self.backbone.embed_tokens

    def _build_inputs_embeds(
        self,
        flat: MU.FlatBatch,
        masked_target_blocks: torch.BoolTensor,
        active_discrete_stream_count: Optional[torch.Tensor | int] = None,
    ) -> tuple[torch.FloatTensor, torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
        """Build model input embeddings and masks for masked discrete/continuous targets."""
        is_speech = flat.token_type_ids != MU.TEXT_TOKEN_TYPE
        is_discrete = flat.token_type_ids == MU.SPEECH_DISCRETE_TOKEN_TYPE
        is_continuous = flat.token_type_ids == MU.SPEECH_CONTINUOUS_TOKEN_TYPE

        target_token_mask = flat.target_block_ids >= 0
        if masked_target_blocks.numel() == 0:
            masked_target_token_mask = torch.zeros_like(target_token_mask)
        else:
            clamped_target_ids = flat.target_block_ids.clamp(min=0)
            target_lookup = torch.gather(masked_target_blocks, 1, clamped_target_ids)
            masked_target_token_mask = target_token_mask & target_lookup

        masked_discrete_positions = masked_target_token_mask & is_discrete
        masked_continuous_positions = masked_target_token_mask & is_continuous

        token_embeds = self.text_embedding(flat.token_ids)
        if is_discrete.any():
            discrete_token_ids = flat.token_ids.clamp(min=0, max=self.discrete_vocab_size - 1)
            discrete_embeds = torch.zeros_like(token_embeds)
            discrete_stream_ids = flat.speech_stream_ids
            for stream_idx, embedding in enumerate(self.discrete_embeddings):
                stream_mask = is_discrete & (discrete_stream_ids == stream_idx)
                if stream_mask.any():
                    discrete_embeds[stream_mask] = embedding(discrete_token_ids[stream_mask])
            token_embeds = torch.where(
                is_discrete.unsqueeze(-1),
                discrete_embeds,
                token_embeds,
            )
        continuous_embeds = self.continuous_proj(flat.continuous_values)
        base_embeds = torch.where(
            is_continuous.unsqueeze(-1),
            continuous_embeds,
            token_embeds,
        )

        type_embeds = self.token_type_embeddings[flat.token_type_ids.clamp(min=0, max=2)]
        base_embeds = base_embeds + type_embeds

        stream_ids_clamped = flat.speech_stream_ids.clamp(min=0, max=self.speech_block_size - 1)
        stream_embeds = self.speech_stream_embeddings[stream_ids_clamped]
        base_embeds = base_embeds + stream_embeds * is_speech.unsqueeze(-1).to(dtype=base_embeds.dtype)

        if masked_discrete_positions.any():
            disc_stream_ids = flat.speech_stream_ids.clamp(min=0, max=self.num_discrete_tokens - 1)
            masked_discrete_embeds = self.masked_discrete_embeddings[disc_stream_ids]
            masked_discrete_embeds = (
                masked_discrete_embeds
                + self.token_type_embeddings[MU.SPEECH_DISCRETE_TOKEN_TYPE]
                + self.speech_stream_embeddings[disc_stream_ids]
            )
            base_embeds = torch.where(
                masked_discrete_positions.unsqueeze(-1),
                masked_discrete_embeds,
                base_embeds,
            )

        if masked_continuous_positions.any():
            cont_stream_ids = flat.speech_stream_ids.clamp(min=0, max=self.speech_block_size - 1)
            masked_cont_embeds = (
                self.masked_continuous_embedding.view(1, 1, self.hidden_size)
                + self.token_type_embeddings[MU.SPEECH_CONTINUOUS_TOKEN_TYPE].view(1, 1, self.hidden_size)
                + self.speech_stream_embeddings[cont_stream_ids]
            )
            base_embeds = torch.where(
                masked_continuous_positions.unsqueeze(-1),
                masked_cont_embeds,
                base_embeds,
            )

        active_stream_count_ids = self._normalize_active_discrete_stream_count(
            flat=flat,
            active_discrete_stream_count=active_discrete_stream_count,
        )
        active_stream_embeds = self.active_stream_count_embedding(active_stream_count_ids)
        base_embeds = base_embeds + active_stream_embeds.unsqueeze(1)

        return (
            base_embeds,
            masked_discrete_positions,
            masked_continuous_positions,
            masked_target_token_mask,
        )

    def _normalize_active_discrete_stream_count(
        self,
        *,
        flat: MU.FlatBatch,
        active_discrete_stream_count: Optional[torch.Tensor | int],
    ) -> torch.LongTensor:
        batch_size = int(flat.token_ids.shape[0])
        device = flat.token_ids.device
        if active_discrete_stream_count is None:
            discrete_mask = flat.token_type_ids == MU.SPEECH_DISCRETE_TOKEN_TYPE
            inferred = torch.full(
                (batch_size,),
                fill_value=self.num_discrete_tokens,
                dtype=torch.long,
                device=device,
            )
            if discrete_mask.any():
                observed_stream_ids = torch.where(
                    discrete_mask,
                    flat.speech_stream_ids,
                    flat.speech_stream_ids.new_full(flat.speech_stream_ids.shape, -1),
                )
                has_discrete = discrete_mask.any(dim=1)
                inferred_stream_count = observed_stream_ids.max(dim=1).values + 1
                inferred = torch.where(
                    has_discrete,
                    inferred_stream_count.clamp(min=1),
                    inferred,
                )
            return inferred

        active_count = torch.as_tensor(
            active_discrete_stream_count,
            dtype=torch.long,
            device=device,
        )
        if active_count.dim() == 0:
            active_count = active_count.repeat(batch_size)
        elif active_count.dim() == 1 and int(active_count.shape[0]) == 1 and batch_size != 1:
            active_count = active_count.repeat(batch_size)
        elif active_count.dim() != 1 or int(active_count.shape[0]) != batch_size:
            raise ValueError(
                "active_discrete_stream_count must be scalar or shape [batch]."
            )

        if ((active_count < 1) | (active_count > self.num_discrete_tokens)).any():
            raise ValueError(
                "active_discrete_stream_count must be in "
                f"[1, {self.num_discrete_tokens}]."
            )
        return active_count

    def _compute_discrete_loss(
        self,
        hidden_states: torch.FloatTensor,
        token_ids: torch.LongTensor,
        speech_stream_ids: torch.LongTensor,
        masked_discrete_positions: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.FloatTensor]:
        """Compute weighted CE on masked discrete tokens and return selected logits."""
        if not masked_discrete_positions.any():
            zero = hidden_states.new_zeros(())
            empty_logits = hidden_states.new_zeros((0, self.discrete_vocab_size))
            return zero, empty_logits

        selected_hidden = hidden_states[masked_discrete_positions]
        selected_targets = token_ids[masked_discrete_positions]
        selected_stream_ids = speech_stream_ids[masked_discrete_positions]
        selected_logits = self._project_discrete_logits(
            hidden_states=selected_hidden,
            discrete_stream_ids=selected_stream_ids,
        )
        per_token_loss = F.cross_entropy(
            selected_logits,
            selected_targets,
            reduction="none",
        )

        token_weights = torch.full_like(
            per_token_loss,
            fill_value=self.discrete_regular_token_loss_weight,
            dtype=per_token_loss.dtype,
        )
        if (
            self.discrete_special_token_loss_weight != self.discrete_regular_token_loss_weight
            and len(self.training_special_discrete_token_ids) > 0
        ):
            special_ids = torch.tensor(
                self.training_special_discrete_token_ids,
                dtype=selected_targets.dtype,
                device=selected_targets.device,
            )
            special_mask = torch.isin(selected_targets, special_ids)
            token_weights = torch.where(
                special_mask,
                token_weights.new_full(
                    token_weights.shape,
                    self.discrete_special_token_loss_weight,
                ),
                token_weights,
            )

        weighted_loss = per_token_loss * token_weights
        normalizer = token_weights.sum().clamp_min(1e-12)
        loss = weighted_loss.sum() / normalizer
        return loss, selected_logits

    def _project_discrete_logits(
        self,
        *,
        hidden_states: torch.FloatTensor,
        discrete_stream_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Project masked discrete hidden states with stream-specific LM heads."""
        if hidden_states.dim() != 2:
            raise ValueError("hidden_states must have shape [num_tokens, hidden_size].")
        if discrete_stream_ids.dim() != 1:
            raise ValueError("discrete_stream_ids must have shape [num_tokens].")
        if hidden_states.shape[0] != discrete_stream_ids.shape[0]:
            raise ValueError("hidden_states and discrete_stream_ids must align on token dimension.")
        if hidden_states.shape[0] == 0:
            return hidden_states.new_zeros((0, self.discrete_vocab_size))

        if (
            (discrete_stream_ids < 0).any()
            or (discrete_stream_ids >= self.num_discrete_tokens).any()
        ):
            raise ValueError("discrete_stream_ids contain out-of-range values.")

        logits = None
        for stream_idx, lm_head in enumerate(self.discrete_lm_heads):
            stream_mask = discrete_stream_ids == stream_idx
            if stream_mask.any():
                stream_logits = lm_head(hidden_states[stream_mask])
                if logits is None:
                    logits = stream_logits.new_empty(
                        (hidden_states.shape[0], self.discrete_vocab_size)
                    )
                logits[stream_mask] = stream_logits
        if logits is None:
            return hidden_states.new_empty((0, self.discrete_vocab_size))
        return logits

    def _compute_continuous_losses(
        self,
        hidden_states: torch.FloatTensor,
        continuous_values: torch.FloatTensor,
        token_type_ids: torch.LongTensor,
        masked_continuous_positions: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute masked continuous latent reconstruction loss from backbone hidden states."""
        if not masked_continuous_positions.any():
            return hidden_states.new_zeros(())

        valid_masked_continuous = (
            masked_continuous_positions
            & (token_type_ids == MU.SPEECH_CONTINUOUS_TOKEN_TYPE)
        )
        if not valid_masked_continuous.any():
            return hidden_states.new_zeros(())
        predicted_continuous_latents = self.continuous_prior_head(
            hidden_states[valid_masked_continuous]
        )
        return F.mse_loss(
            predicted_continuous_latents,
            continuous_values[valid_masked_continuous],
        )

    def _sample_discrete_ids(
        self,
        logits: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> torch.LongTensor:
        """Compatibility wrapper for utilities-backed discrete sampling."""
        return MU.sample_discrete_ids(
            logits=logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )

    def sample_continuous_latent(
        self,
        cond: torch.FloatTensor,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.FloatTensor:
        """Return backbone-predicted continuous latents for compatibility."""
        del num_steps, temperature
        if cond.dim() not in (2, 3):
            raise ValueError("cond must have shape [batch, channels] or [batch, seq, channels].")
        return cond

    def _encode(
        self,
        flat: MU.FlatBatch,
        masked_target_blocks: torch.BoolTensor,
        active_discrete_stream_count: Optional[torch.Tensor | int] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
    ) -> tuple[
        torch.FloatTensor,
        torch.BoolTensor,
        torch.BoolTensor,
        torch.BoolTensor,
    ]:
        """Encode flattened inputs with masking and two-level RoPE."""
        resolved_attention_mask = flat.attention_mask
        if attention_mask is not None:
            if (
                attention_mask.dim() != 2
                or attention_mask.shape[0] != flat.token_ids.shape[0]
                or attention_mask.shape[1] != flat.token_ids.shape[1]
            ):
                raise ValueError("attention_mask must have shape [batch, sequence].")
            resolved_attention_mask = attention_mask.to(
                device=flat.token_ids.device,
                dtype=torch.bool,
            )

        inputs_embeds, masked_discrete_positions, masked_continuous_positions, masked_target_token_mask = (
            self._build_inputs_embeds(
                flat=flat,
                masked_target_blocks=masked_target_blocks,
                active_discrete_stream_count=active_discrete_stream_count,
            )
        )
        position_embeddings = MU.build_two_level_rope_position_embeddings(
            inputs_embeds=inputs_embeds,
            speech_stream_ids=flat.speech_stream_ids,
            rotary_emb=self.backbone.rotary_emb,
        )
        backbone_outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=resolved_attention_mask,
            position_embeddings=position_embeddings,
            return_dict=True,
        )
        return (
            backbone_outputs.last_hidden_state,
            masked_discrete_positions,
            masked_continuous_positions,
            masked_target_token_mask,
        )

    def forward(
        self,
        flat_token_ids: torch.LongTensor,
        flat_continuous_values: torch.FloatTensor,
        flat_token_type_ids: torch.LongTensor,
        flat_speech_stream_ids: torch.LongTensor,
        flat_target_block_ids: torch.LongTensor,
        flat_target_block_counts: Optional[torch.LongTensor] = None,
        active_discrete_stream_count: Optional[torch.Tensor | int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mask_ratio: Optional[float] = None,
        masked_target_blocks: Optional[torch.BoolTensor] = None,
        return_dict: bool = True,
    ) -> MU.PrismTTSOutput | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run masked block reconstruction training from collate-preflattened tensors."""
        flat = MU.build_flat_batch_from_collate(
            flat_token_ids=flat_token_ids,
            flat_continuous_values=flat_continuous_values,
            flat_token_type_ids=flat_token_type_ids,
            flat_speech_stream_ids=flat_speech_stream_ids,
            flat_target_block_ids=flat_target_block_ids,
            flat_target_block_counts=flat_target_block_counts,
            attention_mask=attention_mask,
            continuous_latent_size=self.continuous_latent_size,
        )

        if mask_ratio is None:
            # Default training behavior: sample the eligible-region mask ratio uniformly.
            effective_mask_ratio = float(
                0.3 + 0.7 * torch.rand((), device=flat.token_ids.device).item()
            )
        else:
            effective_mask_ratio = float(mask_ratio)
            if not (0.3 <= effective_mask_ratio <= 1.0):
                raise ValueError("mask_ratio must be in [0.3, 1.0].")
        masked_blocks = MU.sample_masked_target_blocks(
            target_block_counts=flat.target_block_counts,
            mask_ratio=effective_mask_ratio,
            masked_target_blocks=masked_target_blocks,
        )

        (
            hidden_states,
            masked_discrete_positions,
            masked_continuous_positions,
            _,
        ) = self._encode(
            flat=flat,
            masked_target_blocks=masked_blocks,
            active_discrete_stream_count=active_discrete_stream_count,
        )

        discrete_loss, _ = self._compute_discrete_loss(
            hidden_states=hidden_states,
            token_ids=flat.token_ids,
            speech_stream_ids=flat.speech_stream_ids,
            masked_discrete_positions=masked_discrete_positions,
        )
        continuous_loss = self._compute_continuous_losses(
            hidden_states=hidden_states,
            continuous_values=flat.continuous_values,
            token_type_ids=flat.token_type_ids,
            masked_continuous_positions=masked_continuous_positions,
        )
        loss = discrete_loss + self.continuous_loss_weight * continuous_loss

        if not return_dict:
            return loss, discrete_loss, continuous_loss

        return MU.PrismTTSOutput(
            loss=loss,
            discrete_loss=discrete_loss,
            continuous_loss=continuous_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        text_prompt: Optional[torch.LongTensor] = None,
        discrete_prompt: Optional[torch.LongTensor] = None,
        continuous_prompt: Optional[torch.FloatTensor] = None,
        text_target: Optional[torch.LongTensor] = None,
        text_prompt_lengths: Optional[torch.Tensor | int] = None,
        speech_prompt_lengths: Optional[torch.Tensor | int] = None,
        text_target_lengths: Optional[torch.Tensor | int] = None,
        speech_target_lengths: Optional[torch.Tensor | int] = None,
        max_new_blocks: Optional[int] = 375,
        discrete_eos_token_id: Optional[int] = 2049,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        force_silent_special_tokens: bool = False,
        return_dict: bool = True,
        generation_method: str = "causal",
        parallel_num_steps: Optional[int] = None,
        raw_text_prompt: str | Sequence[str] | None = None,
        raw_speech_prompt: Any | list[Any] | None = None,
        raw_text_target: str | Sequence[str] | None = None,
        text_tokenizer: Optional[Callable[[str], Sequence[int]]] = None,
        speech_encoder: Optional[Callable[[Any], tuple[torch.Tensor, torch.Tensor]]] = None,
        speech_decoder: Optional[Callable[[torch.Tensor], Any]] = None,
        output_type: str = "tensor",
        active_discrete_streams: Optional[int] = None,
        mimi_model_name_or_path: str = "kyutai/mimi",
        mimi_revision: str = "main",
        mimi_token: str | bool | None = None,
        mimi_local_files_only: bool = False,
    ) -> (
        MU.PrismTTSGenerationOutput
        | tuple[torch.LongTensor, torch.FloatTensor]
        | torch.Tensor
    ):
        """
        Generate target speech from either:
        - tensorized prompt/target inputs (`text_prompt`, `discrete_prompt`, ...)
        - raw end-to-end inputs (`raw_text_prompt`, `raw_speech_prompt`, `raw_text_target`)
        """
        uses_raw_inputs = any(
            arg is not None
            for arg in (raw_text_prompt, raw_speech_prompt, raw_text_target)
        )
        if uses_raw_inputs:
            if (
                raw_text_prompt is None
                or raw_speech_prompt is None
                or raw_text_target is None
            ):
                raise ValueError(
                    "raw_text_prompt, raw_speech_prompt, and raw_text_target must all be provided together."
                )
            if text_prompt is not None or discrete_prompt is not None or continuous_prompt is not None:
                raise ValueError(
                    "Do not pass text_prompt/discrete_prompt/continuous_prompt when using raw_* inputs."
                )
            if text_prompt_lengths is not None:
                raise ValueError("Do not pass text_prompt_lengths when using raw_* inputs.")
            if speech_prompt_lengths is not None:
                raise ValueError("Do not pass speech_prompt_lengths when using raw_* inputs.")
            if text_target_lengths is not None:
                raise ValueError("Do not pass text_target_lengths when using raw_* inputs.")
            return e2e_generation.generate_e2e(
                model=self,
                raw_text_prompt=raw_text_prompt,
                raw_speech_prompt=raw_speech_prompt,
                raw_text_target=raw_text_target,
                text_tokenizer=text_tokenizer,
                speech_encoder=speech_encoder,
                speech_decoder=speech_decoder,
                output_type=output_type,
                return_dict=return_dict,
                mimi_model_name_or_path=mimi_model_name_or_path,
                mimi_revision=mimi_revision,
                mimi_token=mimi_token,
                mimi_local_files_only=mimi_local_files_only,
                speech_target_lengths=speech_target_lengths,
                max_new_blocks=max_new_blocks,
                discrete_eos_token_id=discrete_eos_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                force_silent_special_tokens=force_silent_special_tokens,
                generation_method=generation_method,
                parallel_num_steps=parallel_num_steps,
                active_discrete_streams=active_discrete_streams,
            )

        if (
            text_tokenizer is not None
            or speech_encoder is not None
            or speech_decoder is not None
        ):
            raise ValueError(
                "text_tokenizer/speech_encoder/speech_decoder are only valid with raw_* inputs."
            )
        output_type_normalized = str(output_type).strip().lower()
        if output_type_normalized not in ("tensor", "speech"):
            raise ValueError("output_type must be one of {'tensor', 'speech'}.")
        if output_type_normalized != "tensor":
            raise ValueError("output_type='speech' requires raw_* end-to-end inputs.")
        if text_prompt is None or discrete_prompt is None or continuous_prompt is None:
            raise ValueError(
                "text_prompt, discrete_prompt, and continuous_prompt are required when raw_* inputs are not used."
            )
        text_prompt = MU.normalize_text_tokens(text_prompt, "text_prompt")
        discrete_prompt = MU.normalize_discrete_tokens(
            discrete_prompt,
            "discrete_prompt",
            num_discrete_tokens=self.num_discrete_tokens,
            allow_fewer_streams=True,
        )
        prompt_discrete_streams = int(discrete_prompt.shape[-1])
        if active_discrete_streams is None:
            resolved_active_discrete_streams = prompt_discrete_streams
        else:
            resolved_active_discrete_streams = int(active_discrete_streams)
        if (
            resolved_active_discrete_streams < 1
            or resolved_active_discrete_streams > self.num_discrete_tokens
        ):
            raise ValueError(
                "active_discrete_streams must be in [1, num_discrete_tokens]."
            )
        if resolved_active_discrete_streams > prompt_discrete_streams:
            raise ValueError(
                "active_discrete_streams exceeds provided discrete_prompt stream count: "
                f"{resolved_active_discrete_streams} > {prompt_discrete_streams}."
            )
        discrete_prompt = discrete_prompt[:, :, :resolved_active_discrete_streams]
        continuous_prompt = MU.normalize_continuous_latents(
            continuous_prompt,
            expected_len=int(discrete_prompt.shape[1]),
            name="continuous_prompt",
            continuous_latent_size=self.continuous_latent_size,
        )
        batch_size = int(text_prompt.shape[0])

        if text_target is None:
            text_target = text_prompt.new_zeros((batch_size, 0))
        else:
            text_target = MU.normalize_text_tokens(text_target, "text_target")

        if max_new_blocks is not None and int(max_new_blocks) < 0:
            raise ValueError("max_new_blocks must be >= 0 when provided.")
        if parallel_num_steps is not None and int(parallel_num_steps) < 1:
            raise ValueError("parallel_num_steps must be >= 1 when provided.")
        resolved_parallel_num_steps = (
            self.parallel_sample_steps
            if parallel_num_steps is None
            else int(parallel_num_steps)
        )
        generation_method_normalized = str(generation_method).strip().lower()
        if generation_method_normalized not in ("causal", "parallel", "parallel_stable"):
            raise ValueError(
                "generation_method must be one of {'causal', 'parallel', 'parallel_stable'}."
            )

        text_prompt_lengths = MU.normalize_lengths(
            lengths=text_prompt_lengths,
            batch_size=batch_size,
            max_length=int(text_prompt.shape[1]),
            name="text_prompt_lengths",
            device=text_prompt.device,
            default_value=int(text_prompt.shape[1]),
        )
        speech_prompt_lengths = MU.normalize_lengths(
            lengths=speech_prompt_lengths,
            batch_size=batch_size,
            max_length=int(discrete_prompt.shape[1]),
            name="speech_prompt_lengths",
            device=text_prompt.device,
            default_value=int(discrete_prompt.shape[1]),
        )
        text_target_lengths = MU.normalize_lengths(
            lengths=text_target_lengths,
            batch_size=batch_size,
            max_length=int(text_target.shape[1]),
            name="text_target_lengths",
            device=text_prompt.device,
            default_value=int(text_target.shape[1]),
        )
        max_text_target = int(text_target_lengths.max().item()) if batch_size > 0 else 0
        text_ids_out = text_target[:, :max_text_target]

        if speech_target_lengths is None:
            if generation_method_normalized in ("parallel", "parallel_stable"):
                speech_target_lengths = MU.estimate_parallel_speech_target_lengths(
                    text_prompt_lengths=text_prompt_lengths,
                    speech_prompt_lengths=speech_prompt_lengths,
                    text_target_lengths=text_target_lengths,
                    max_new_blocks=max_new_blocks,
                )
            else:
                speech_target_max_length = (
                    int(max_new_blocks)
                    if max_new_blocks is not None
                    else int(text_target.shape[1])
                )
                default_speech_target_length = (
                    int(text_target.shape[1])
                    if max_new_blocks is None
                    else int(max_new_blocks)
                )
                speech_target_lengths = MU.normalize_lengths(
                    lengths=speech_target_lengths,
                    batch_size=batch_size,
                    max_length=speech_target_max_length,
                    name="speech_target_lengths",
                    device=text_prompt.device,
                    default_value=default_speech_target_length,
                )
        else:
            speech_target_max_length = int(
                torch.as_tensor(speech_target_lengths, device=text_prompt.device)
                .to(dtype=torch.long)
                .max()
                .item()
            )
            speech_target_lengths = MU.normalize_lengths(
                lengths=speech_target_lengths,
                batch_size=batch_size,
                max_length=speech_target_max_length,
                name="speech_target_lengths",
                device=text_prompt.device,
                default_value=None,
            )
        if max_new_blocks is not None:
            speech_target_lengths = torch.minimum(
                speech_target_lengths,
                torch.full_like(speech_target_lengths, int(max_new_blocks)),
            )

        max_target = int(speech_target_lengths.max().item()) if batch_size > 0 else 0
        if max_target <= 0:
            empty_disc = discrete_prompt.new_empty(
                (batch_size, resolved_active_discrete_streams, 0)
            )
            empty_cont = continuous_prompt.new_empty((batch_size, 0, self.continuous_latent_size))
            if not return_dict:
                return empty_disc, empty_cont
            return MU.PrismTTSGenerationOutput(
                text_ids=text_ids_out,
                discrete_ids=empty_disc,
                continuous_latents=empty_cont,
                prior_latents=empty_cont,
                discrete_logits=tuple(),
            )

        discrete_eos_id = MU.resolve_generation_discrete_eos_token_id(
            discrete_eos_token_id,
            backbone_eos_token_id=self.backbone.config.eos_token_id,
            discrete_vocab_size=self.discrete_vocab_size,
        )
        special_discrete_token_ids = (
            MU.infer_special_discrete_token_ids(
                discrete_eos_id,
                backbone_eos_token_id=self.backbone.config.eos_token_id,
                backbone_pad_token_id=self.backbone.config.pad_token_id,
                discrete_vocab_size=self.discrete_vocab_size,
            )
            if force_silent_special_tokens
            else tuple()
        )
        if generation_method_normalized == "parallel":
            generation_fn = parallel_generation.generate_parallel
        elif generation_method_normalized == "parallel_stable":
            generation_fn = parallel_stable_generation.generate_parallel_stable
        else:
            generation_fn = causal_generation.generate_causal
        (
            predicted_discrete,
            predicted_continuous,
            predicted_prior,
            generated_lengths,
            collected_logits,
        ) = generation_fn(
            model=self,
            text_prompt=text_prompt,
            discrete_prompt=discrete_prompt,
            continuous_prompt=continuous_prompt,
            text_target=text_target,
            text_prompt_lengths=text_prompt_lengths,
            speech_prompt_lengths=speech_prompt_lengths,
            text_target_lengths=text_target_lengths,
            speech_target_lengths=speech_target_lengths,
            discrete_eos_id=discrete_eos_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            parallel_num_steps=resolved_parallel_num_steps,
            special_discrete_token_ids=special_discrete_token_ids,
            active_discrete_streams=resolved_active_discrete_streams,
        )

        final_target = int(generated_lengths.max().item()) if batch_size > 0 else 0
        predicted_discrete = predicted_discrete[:, :final_target, :]
        predicted_continuous = predicted_continuous[:, :final_target, :]
        predicted_prior = predicted_prior[:, :final_target, :]

        if not return_dict:
            return predicted_discrete.transpose(1, 2).contiguous(), predicted_continuous
        return MU.PrismTTSGenerationOutput(
            text_ids=text_ids_out,
            discrete_ids=predicted_discrete.transpose(1, 2).contiguous(),
            continuous_latents=predicted_continuous,
            prior_latents=predicted_prior,
            discrete_logits=tuple(collected_logits),
        )
