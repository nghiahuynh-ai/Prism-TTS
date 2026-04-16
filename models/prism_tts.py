from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig

from models.flow_head import FlowHead
from models.llama_backbone import LlamaBackbone
from utils.model_utils import (
    SPEECH_CONTINUOUS_TOKEN_TYPE,
    SPEECH_DISCRETE_TOKEN_TYPE,
    TEXT_TOKEN_TYPE,
    FlatBatch,
    PrismTTSGenerationOutput,
    PrismTTSOutput,
    assemble_flat_batch,
    build_flat_batch_from_collate,
    build_special_block_mask,
    build_two_level_rope_position_embeddings,
    estimate_parallel_speech_target_lengths,
    infer_special_discrete_token_ids,
    normalize_continuous_latents,
    normalize_discrete_tokens,
    normalize_lengths,
    normalize_text_tokens,
    resolve_generation_discrete_eos_token_id,
    sample_masked_target_blocks,
    top_k_top_p_filter,
)


class PrismTTS(nn.Module):
    """
    Prism-TTS masked reconstruction model.

    Sequence layout (per sample):
    text_prompt -> EOT -> speech_prompt -> EOS -> text_target -> EOT -> speech_target -> EOS

    Speech is flattened block-wise. Each speech block has (N + 1) streams:
    N discrete streams + 1 continuous stream.
    """

    def __init__(
        self,
        llama_config: LlamaConfig,
        num_discrete_tokens: int,
        discrete_vocab_size: int,
        continuous_latent_size: int,
        flow_num_res_blocks: int = 4,
        flow_model_channels: Optional[int] = None,
        flow_loss_weight: float = 1.0,
        continuous_loss_weight: float = 1.0,
        discrete_regular_token_loss_weight: float = 1.0,
        discrete_special_token_loss_weight: float = 1.0,
        flow_sample_steps: int = 64,
        parallel_sample_steps: int = 64,
    ):
        """Initialize model modules, embeddings, loss weights, and special-token ids."""
        super().__init__()
        if num_discrete_tokens < 1:
            raise ValueError("num_discrete_tokens must be at least 1.")
        if discrete_vocab_size < 1:
            raise ValueError("discrete_vocab_size must be at least 1.")
        if discrete_vocab_size > llama_config.vocab_size:
            raise ValueError(
                "discrete_vocab_size must be <= llama_config.vocab_size when text/discrete embeddings are shared."
            )
        if continuous_latent_size < 1:
            raise ValueError("continuous_latent_size must be at least 1.")
        if flow_loss_weight < 0.0:
            raise ValueError("flow_loss_weight must be >= 0.")
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
        if flow_sample_steps < 1:
            raise ValueError("flow_sample_steps must be at least 1.")
        if parallel_sample_steps < 1:
            raise ValueError("parallel_sample_steps must be at least 1.")

        self.hidden_size = int(llama_config.hidden_size)
        self.num_discrete_tokens = int(num_discrete_tokens)
        self.discrete_vocab_size = int(discrete_vocab_size)
        self.continuous_latent_size = int(continuous_latent_size)
        self.speech_block_size = self.num_discrete_tokens + 1

        self.flow_loss_weight = float(flow_loss_weight)
        self.continuous_loss_weight = float(continuous_loss_weight)
        self.discrete_regular_token_loss_weight = float(discrete_regular_token_loss_weight)
        self.discrete_special_token_loss_weight = float(discrete_special_token_loss_weight)
        self.flow_sample_steps = int(flow_sample_steps)
        self.parallel_sample_steps = int(parallel_sample_steps)

        self.backbone = LlamaBackbone(llama_config)
        self.discrete_lm_head = nn.Linear(self.hidden_size, self.discrete_vocab_size, bias=False)
        self.continuous_proj = nn.Linear(self.continuous_latent_size, self.hidden_size)
        self.continuous_prior_head = nn.Linear(self.hidden_size, self.continuous_latent_size)

        self.token_type_embeddings = nn.Parameter(torch.empty(3, self.hidden_size))
        self.speech_stream_embeddings = nn.Parameter(
            torch.empty(self.speech_block_size, self.hidden_size)
        )
        self.masked_discrete_embeddings = nn.Parameter(
            torch.empty(self.num_discrete_tokens, self.hidden_size)
        )
        self.masked_continuous_embedding = nn.Parameter(torch.empty(self.hidden_size))

        # Flow head conditions on the continuous prior.
        self.flow_head = FlowHead(
            in_channels=self.continuous_latent_size,
            model_channels=flow_model_channels or self.hidden_size,
            out_channels=self.continuous_latent_size,
            z_channels=self.continuous_latent_size,
            num_res_blocks=flow_num_res_blocks,
        )

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

        self.training_special_discrete_token_ids = infer_special_discrete_token_ids(
            resolve_generation_discrete_eos_token_id(
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
        nn.init.normal_(self.discrete_lm_head.weight, mean=0.0, std=std)
        nn.init.normal_(self.continuous_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.continuous_prior_head.weight, mean=0.0, std=std)
        nn.init.normal_(self.token_type_embeddings, mean=0.0, std=std)
        nn.init.normal_(self.speech_stream_embeddings, mean=0.0, std=std)
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

    @property
    def discrete_embedding(self) -> nn.Embedding:
        """Return the shared embedding table used for discrete speech tokens."""
        return self.backbone.embed_tokens

    def _build_inputs_embeds(
        self,
        flat: FlatBatch,
        masked_target_blocks: torch.BoolTensor,
    ) -> tuple[torch.FloatTensor, torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
        """Build model input embeddings and masks for masked discrete/continuous targets."""
        batch_size, seq_len = flat.token_ids.shape
        device = flat.token_ids.device

        is_speech = flat.token_type_ids != TEXT_TOKEN_TYPE
        is_discrete = flat.token_type_ids == SPEECH_DISCRETE_TOKEN_TYPE
        is_continuous = flat.token_type_ids == SPEECH_CONTINUOUS_TOKEN_TYPE

        target_token_mask = flat.target_block_ids >= 0
        if masked_target_blocks.numel() == 0:
            masked_target_token_mask = torch.zeros_like(target_token_mask)
        else:
            clamped_target_ids = flat.target_block_ids.clamp(min=0)
            target_lookup = torch.gather(masked_target_blocks, 1, clamped_target_ids)
            masked_target_token_mask = target_token_mask & target_lookup

        masked_discrete_positions = masked_target_token_mask & is_discrete
        masked_continuous_positions = masked_target_token_mask & is_continuous

        token_embeds = self.discrete_embedding(flat.token_ids)
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
                + self.token_type_embeddings[SPEECH_DISCRETE_TOKEN_TYPE]
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
                + self.token_type_embeddings[SPEECH_CONTINUOUS_TOKEN_TYPE].view(1, 1, self.hidden_size)
                + self.speech_stream_embeddings[cont_stream_ids]
            )
            base_embeds = torch.where(
                masked_continuous_positions.unsqueeze(-1),
                masked_cont_embeds,
                base_embeds,
            )

        return (
            base_embeds,
            masked_discrete_positions,
            masked_continuous_positions,
            masked_target_token_mask,
        )

    def _sample_flow_training_inputs(
        self,
        continuous_targets: torch.FloatTensor,
        flow_timesteps: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample flow-matching mixtures and targets for continuous latent denoising."""
        if flow_timesteps is None:
            flow_timesteps = torch.rand(
                continuous_targets.shape[:-1],
                device=continuous_targets.device,
                dtype=continuous_targets.dtype,
            )
        if noise is None:
            noise = torch.randn_like(continuous_targets)

        t = flow_timesteps.unsqueeze(-1)
        flow_inputs = (1.0 - t) * noise + t * continuous_targets
        flow_target = continuous_targets - noise
        return flow_inputs, flow_target, flow_timesteps

    def _compute_discrete_loss(
        self,
        hidden_states: torch.FloatTensor,
        token_ids: torch.LongTensor,
        masked_discrete_positions: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.FloatTensor]:
        """Compute plain CE loss on masked discrete tokens and return selected logits."""
        if not masked_discrete_positions.any():
            zero = hidden_states.new_zeros(())
            empty_logits = hidden_states.new_zeros((0, self.discrete_vocab_size))
            return zero, empty_logits

        selected_hidden = hidden_states[masked_discrete_positions]
        selected_targets = token_ids[masked_discrete_positions]
        selected_logits = self.discrete_lm_head(selected_hidden)
        loss = F.cross_entropy(selected_logits, selected_targets)
        return loss, selected_logits

    def _compute_continuous_losses(
        self,
        hidden_states: torch.FloatTensor,
        continuous_values: torch.FloatTensor,
        target_block_ids: torch.LongTensor,
        masked_continuous_positions: torch.BoolTensor,
        flow_timesteps: Optional[torch.FloatTensor],
        noise: Optional[torch.FloatTensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute continuous prior MSE loss and flow-matching denoising loss."""
        if not masked_continuous_positions.any():
            zero = hidden_states.new_zeros(())
            return zero, zero

        selected_hidden = hidden_states[masked_continuous_positions]
        prior_prediction = self.continuous_prior_head(selected_hidden)
        target_continuous = continuous_values[masked_continuous_positions]
        reconstruction_loss = F.mse_loss(prior_prediction, target_continuous)

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        batch_indices = (
            torch.arange(batch_size, device=hidden_states.device)
            .unsqueeze(1)
            .expand(batch_size, seq_len)
        )[masked_continuous_positions]
        target_block_indices = target_block_ids[masked_continuous_positions]

        selected_flow_timesteps: Optional[torch.FloatTensor] = None
        if flow_timesteps is not None:
            if flow_timesteps.dim() != 2 or flow_timesteps.shape[0] != batch_size:
                raise ValueError("flow_timesteps must have shape [batch, target_blocks].")
            max_target_blocks = int(target_block_indices.max().item()) + 1
            if flow_timesteps.shape[1] < max_target_blocks:
                raise ValueError(
                    "flow_timesteps length must cover all target speech blocks."
                )
            selected_flow_timesteps = flow_timesteps[batch_indices, target_block_indices]

        selected_noise: Optional[torch.FloatTensor] = None
        if noise is not None:
            if noise.dim() != 3 or noise.shape[0] != batch_size:
                raise ValueError(
                    "noise must have shape [batch, target_blocks, continuous_latent_size]."
                )
            if noise.shape[2] != self.continuous_latent_size:
                raise ValueError(
                    "noise channel mismatch: expected "
                    f"{self.continuous_latent_size}, got {noise.shape[2]}."
                )
            max_target_blocks = int(target_block_indices.max().item()) + 1
            if noise.shape[1] < max_target_blocks:
                raise ValueError("noise must cover all target speech blocks.")
            selected_noise = noise[batch_indices, target_block_indices, :]

        flow_inputs, flow_target, sampled_timesteps = self._sample_flow_training_inputs(
            continuous_targets=target_continuous,
            flow_timesteps=selected_flow_timesteps,
            noise=selected_noise,
        )
        flow_prediction = self.flow_head(
            flow_inputs,
            sampled_timesteps,
            prior_prediction,
        )
        flow_loss = F.mse_loss(flow_prediction, flow_target)
        return reconstruction_loss, flow_loss

    def _sample_discrete_ids(
        self,
        logits: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> torch.LongTensor:
        """Sample discrete ids from logits with optional temperature/top-k/top-p."""
        if not do_sample or temperature <= 0.0:
            return torch.argmax(logits, dim=-1)

        scaled_logits = logits / temperature
        filtered_logits = top_k_top_p_filter(
            scaled_logits,
            top_k=top_k,
            top_p=top_p,
        )
        probs = torch.softmax(filtered_logits, dim=-1)
        sample_shape = probs.shape[:-1]
        samples = torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1)
        return samples.reshape(*sample_shape)

    def sample_continuous_latent(
        self,
        cond: torch.FloatTensor,
        num_steps: Optional[int] = None,
    ) -> torch.FloatTensor:
        """Generate continuous latents by integrating the flow field from Gaussian noise."""
        num_steps = num_steps or self.flow_sample_steps
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")

        cond_shape = cond.shape[:-1]
        flat_cond = cond.reshape(-1, cond.shape[-1])
        x = torch.randn(
            flat_cond.shape[0],
            self.continuous_latent_size,
            device=cond.device,
            dtype=cond.dtype,
        )
        dt = 1.0 / num_steps

        for step_idx in range(num_steps):
            t = torch.full(
                (flat_cond.shape[0],),
                step_idx / num_steps,
                device=cond.device,
                dtype=cond.dtype,
            )
            velocity = self.flow_head(x, t, flat_cond)
            x = x + dt * velocity

        return x.reshape(*cond_shape, self.continuous_latent_size)

    def _encode(
        self,
        flat: FlatBatch,
        masked_target_blocks: torch.BoolTensor,
    ) -> tuple[
        torch.FloatTensor,
        torch.BoolTensor,
        torch.BoolTensor,
        torch.BoolTensor,
    ]:
        """Encode flattened inputs with masking and two-level RoPE."""
        inputs_embeds, masked_discrete_positions, masked_continuous_positions, masked_target_token_mask = (
            self._build_inputs_embeds(
                flat=flat,
                masked_target_blocks=masked_target_blocks,
            )
        )
        position_embeddings = build_two_level_rope_position_embeddings(
            inputs_embeds=inputs_embeds,
            speech_stream_ids=flat.speech_stream_ids,
            rotary_emb=self.backbone.rotary_emb,
        )
        backbone_outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=flat.attention_mask,
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
        attention_mask: Optional[torch.Tensor] = None,
        flow_timesteps: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        mask_ratio: Optional[float] = None,
        masked_target_blocks: Optional[torch.BoolTensor] = None,
        return_dict: bool = True,
    ) -> PrismTTSOutput | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run masked block reconstruction training from collate-preflattened tensors."""
        flat = build_flat_batch_from_collate(
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
            # Default training behavior: sample mask ratio uniformly per forward pass.
            effective_mask_ratio = float(torch.rand((), device=flat.token_ids.device).item())
        else:
            effective_mask_ratio = float(mask_ratio)
            if not (0.0 <= effective_mask_ratio <= 1.0):
                raise ValueError("mask_ratio must be in [0, 1].")
        masked_blocks = sample_masked_target_blocks(
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
        )

        discrete_loss, _ = self._compute_discrete_loss(
            hidden_states=hidden_states,
            token_ids=flat.token_ids,
            masked_discrete_positions=masked_discrete_positions,
        )
        continuous_loss, flow_loss = self._compute_continuous_losses(
            hidden_states=hidden_states,
            continuous_values=flat.continuous_values,
            target_block_ids=flat.target_block_ids,
            masked_continuous_positions=masked_continuous_positions,
            flow_timesteps=flow_timesteps,
            noise=noise,
        )
        loss = (
            discrete_loss
            + self.continuous_loss_weight * continuous_loss
            + self.flow_loss_weight * flow_loss
        )

        if not return_dict:
            return loss, discrete_loss, continuous_loss, flow_loss

        return PrismTTSOutput(
            loss=loss,
            discrete_loss=discrete_loss,
            continuous_loss=continuous_loss,
            flow_loss=flow_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        continuous_prompt: torch.FloatTensor,
        text_target: Optional[torch.LongTensor] = None,
        text_prompt_lengths: Optional[torch.Tensor | int] = None,
        speech_prompt_lengths: Optional[torch.Tensor | int] = None,
        text_target_lengths: Optional[torch.Tensor | int] = None,
        speech_target_lengths: Optional[torch.Tensor | int] = None,
        max_new_blocks: Optional[int] = 128,
        discrete_eos_token_id: Optional[int] = 2049,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        flow_num_steps: Optional[int] = None,
        force_silent_special_tokens: bool = False,
        return_dict: bool = True,
        generation_method: str = "causal",
        parallel_num_steps: Optional[int] = None,
    ) -> PrismTTSGenerationOutput | tuple[torch.LongTensor, torch.FloatTensor]:
        """Generate target speech blocks conditioned on text/speech prompt and target text."""
        text_prompt = normalize_text_tokens(text_prompt, "text_prompt")
        discrete_prompt = normalize_discrete_tokens(
            discrete_prompt,
            "discrete_prompt",
            num_discrete_tokens=self.num_discrete_tokens,
        )
        continuous_prompt = normalize_continuous_latents(
            continuous_prompt,
            expected_len=int(discrete_prompt.shape[1]),
            name="continuous_prompt",
            continuous_latent_size=self.continuous_latent_size,
        )
        batch_size = int(text_prompt.shape[0])

        if text_target is None:
            text_target = text_prompt.new_zeros((batch_size, 0))
        else:
            text_target = normalize_text_tokens(text_target, "text_target")

        return self._generate_prepared(
            text_prompt=text_prompt,
            discrete_prompt=discrete_prompt,
            continuous_prompt=continuous_prompt,
            text_target=text_target,
            text_prompt_lengths=text_prompt_lengths,
            speech_prompt_lengths=speech_prompt_lengths,
            text_target_lengths=text_target_lengths,
            speech_target_lengths=speech_target_lengths,
            max_new_blocks=max_new_blocks,
            discrete_eos_token_id=discrete_eos_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            flow_num_steps=flow_num_steps,
            parallel_num_steps=parallel_num_steps,
            force_silent_special_tokens=force_silent_special_tokens,
            return_dict=return_dict,
            generation_method=generation_method,
        )

    @torch.no_grad()
    def _generate_prepared(
        self,
        *,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        continuous_prompt: torch.FloatTensor,
        text_target: torch.LongTensor,
        text_prompt_lengths: Optional[torch.Tensor | int] = None,
        speech_prompt_lengths: Optional[torch.Tensor | int] = None,
        text_target_lengths: Optional[torch.Tensor | int] = None,
        speech_target_lengths: Optional[torch.Tensor | int] = None,
        max_new_blocks: Optional[int] = 128,
        discrete_eos_token_id: Optional[int] = 2049,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        flow_num_steps: Optional[int] = None,
        force_silent_special_tokens: bool = False,
        return_dict: bool = True,
        generation_method: str = "causal",
        parallel_num_steps: Optional[int] = None,
    ) -> PrismTTSGenerationOutput | tuple[torch.LongTensor, torch.FloatTensor]:
        """Generate from already-prepared tensor inputs, routed by generation method."""
        batch_size = int(text_prompt.shape[0])

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
        if generation_method_normalized not in ("causal", "parallel"):
            raise ValueError("generation_method must be one of {'causal', 'parallel'}.")

        text_prompt_lengths = normalize_lengths(
            lengths=text_prompt_lengths,
            batch_size=batch_size,
            max_length=int(text_prompt.shape[1]),
            name="text_prompt_lengths",
            device=text_prompt.device,
            default_value=int(text_prompt.shape[1]),
        )
        speech_prompt_lengths = normalize_lengths(
            lengths=speech_prompt_lengths,
            batch_size=batch_size,
            max_length=int(discrete_prompt.shape[1]),
            name="speech_prompt_lengths",
            device=text_prompt.device,
            default_value=int(discrete_prompt.shape[1]),
        )
        text_target_lengths = normalize_lengths(
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
            if generation_method_normalized == "parallel":
                speech_target_lengths = estimate_parallel_speech_target_lengths(
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
                speech_target_lengths = normalize_lengths(
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
            speech_target_lengths = normalize_lengths(
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
            empty_disc = discrete_prompt.new_empty((batch_size, self.num_discrete_tokens, 0))
            empty_cont = continuous_prompt.new_empty((batch_size, 0, self.continuous_latent_size))
            if not return_dict:
                return empty_disc, empty_cont
            return PrismTTSGenerationOutput(
                text_ids=text_ids_out,
                discrete_ids=empty_disc,
                continuous_latents=empty_cont,
                discrete_logits=tuple(),
            )

        discrete_eos_id = resolve_generation_discrete_eos_token_id(
            discrete_eos_token_id,
            backbone_eos_token_id=self.backbone.config.eos_token_id,
            discrete_vocab_size=self.discrete_vocab_size,
        )
        special_discrete_token_ids = (
            infer_special_discrete_token_ids(
                discrete_eos_id,
                backbone_eos_token_id=self.backbone.config.eos_token_id,
                backbone_pad_token_id=self.backbone.config.pad_token_id,
                discrete_vocab_size=self.discrete_vocab_size,
            )
            if force_silent_special_tokens
            else tuple()
        )
        generation_fn = (
            self.generate_parallel
            if generation_method_normalized == "parallel"
            else self.generate_causal
        )
        (
            predicted_discrete,
            predicted_continuous,
            generated_lengths,
            collected_logits,
        ) = generation_fn(
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
            flow_num_steps=flow_num_steps,
            parallel_num_steps=resolved_parallel_num_steps,
            special_discrete_token_ids=special_discrete_token_ids,
        )

        final_target = int(generated_lengths.max().item()) if batch_size > 0 else 0
        predicted_discrete = predicted_discrete[:, :final_target, :]
        predicted_continuous = predicted_continuous[:, :final_target, :]

        if not return_dict:
            return predicted_discrete.transpose(1, 2).contiguous(), predicted_continuous
        return PrismTTSGenerationOutput(
            text_ids=text_ids_out,
            discrete_ids=predicted_discrete.transpose(1, 2).contiguous(),
            continuous_latents=predicted_continuous,
            discrete_logits=tuple(collected_logits),
        )

    @torch.no_grad()
    def generate_causal(
        self,
        *,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        continuous_prompt: torch.FloatTensor,
        text_target: torch.LongTensor,
        text_prompt_lengths: torch.LongTensor,
        speech_prompt_lengths: torch.LongTensor,
        text_target_lengths: torch.LongTensor,
        speech_target_lengths: torch.LongTensor,
        discrete_eos_id: int,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
        flow_num_steps: Optional[int],
        parallel_num_steps: int,
        special_discrete_token_ids: tuple[int, ...],
    ) -> tuple[
        torch.LongTensor,
        torch.FloatTensor,
        torch.LongTensor,
        list[torch.Tensor],
    ]:
        """Causal left-to-right generation with a fixed terminal EOS speech block."""
        del parallel_num_steps
        batch_size = int(text_prompt.shape[0])
        device = text_prompt.device
        max_target = int(speech_target_lengths.max().item()) if batch_size > 0 else 0
        safe_discrete_fill_id = (
            int(discrete_eos_id)
            if 0 <= int(discrete_eos_id) < self.discrete_vocab_size
            else self.pad_token_id
        )
        terminal_discrete_id = safe_discrete_fill_id

        predicted_discrete = discrete_prompt.new_full(
            (batch_size, max_target, self.num_discrete_tokens),
            safe_discrete_fill_id,
        )
        predicted_continuous = continuous_prompt.new_zeros(
            (batch_size, max_target, self.continuous_latent_size)
        )
        generated_lengths = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=device,
        )
        collected_logits: list[torch.Tensor] = []
        if max_target <= 0:
            return predicted_discrete, predicted_continuous, generated_lengths, collected_logits

        valid_target_mask = (
            torch.arange(max_target, device=device).unsqueeze(0)
            < speech_target_lengths.unsqueeze(1)
        )
        maskable_target_mask = valid_target_mask.clone()
        has_target = speech_target_lengths > 0
        if has_target.any():
            eos_sample_idx = torch.nonzero(has_target, as_tuple=False).squeeze(1)
            eos_block_idx = speech_target_lengths[eos_sample_idx] - 1
            predicted_discrete[eos_sample_idx, eos_block_idx, :] = terminal_discrete_id
            predicted_continuous[eos_sample_idx, eos_block_idx, :] = 0.0
            maskable_target_mask[eos_sample_idx, eos_block_idx] = False

        step_indices = torch.arange(max_target, device=device).view(max_target, 1, 1)
        block_indices = torch.arange(max_target, device=device).view(1, 1, max_target)
        left_to_right_masks = block_indices >= step_indices
        mask_schedule = maskable_target_mask.unsqueeze(0) & left_to_right_masks

        maskable_lengths = torch.clamp(speech_target_lengths - 1, min=0)
        finished = maskable_lengths <= 0
        for step_idx in range(max_target):
            masked_blocks = mask_schedule[step_idx] & (~finished).unsqueeze(1)
            if not bool(masked_blocks.any().item()):
                break

            current_discrete = predicted_discrete.clone()
            current_continuous = predicted_continuous.clone()
            current_discrete[masked_blocks] = self.pad_token_id
            current_continuous[masked_blocks] = 0.0

            flat = assemble_flat_batch(
                text_prompt=text_prompt,
                discrete_prompt=discrete_prompt,
                continuous_prompt=continuous_prompt,
                text_target=text_target,
                discrete_target=current_discrete,
                continuous_target=current_continuous,
                text_prompt_lengths=text_prompt_lengths,
                speech_prompt_lengths=speech_prompt_lengths,
                text_target_lengths=text_target_lengths,
                speech_target_lengths=speech_target_lengths,
                attention_mask=None,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                eot_token_id=self.eot_token_id,
                continuous_latent_size=self.continuous_latent_size,
                num_discrete_tokens=self.num_discrete_tokens,
            )
            hidden_states, masked_discrete_positions, masked_continuous_positions, _ = self._encode(
                flat=flat,
                masked_target_blocks=masked_blocks,
            )
            batch_indices = (
                torch.arange(batch_size, device=device)
                .unsqueeze(1)
                .expand(batch_size, hidden_states.shape[1])
            )
            step_target_mask = flat.target_block_ids == step_idx

            step_logits = torch.full(
                (batch_size, self.num_discrete_tokens, self.discrete_vocab_size),
                fill_value=float("-inf"),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            active_step_samples = (~finished) & (maskable_lengths > step_idx)

            step_discrete_positions = masked_discrete_positions & step_target_mask
            if step_discrete_positions.any():
                discrete_hidden = hidden_states[step_discrete_positions]
                discrete_logits = self.discrete_lm_head(discrete_hidden)
                sampled_discrete = self._sample_discrete_ids(
                    discrete_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                )
                discrete_batch_idx = batch_indices[step_discrete_positions]
                discrete_stream_idx = flat.speech_stream_ids[step_discrete_positions]
                predicted_discrete[discrete_batch_idx, step_idx, discrete_stream_idx] = (
                    sampled_discrete
                )
                step_logits[discrete_batch_idx, discrete_stream_idx, :] = discrete_logits

            step_continuous_positions = masked_continuous_positions & step_target_mask
            if step_continuous_positions.any():
                continuous_hidden = hidden_states[step_continuous_positions]
                prior_prediction = self.continuous_prior_head(continuous_hidden)
                sampled_continuous = self.sample_continuous_latent(
                    cond=prior_prediction,
                    num_steps=flow_num_steps,
                )
                continuous_batch_idx = batch_indices[step_continuous_positions]
                predicted_continuous[continuous_batch_idx, step_idx, :] = sampled_continuous

                if len(special_discrete_token_ids) > 0:
                    step_discrete = predicted_discrete[continuous_batch_idx, step_idx, :]
                    special_step_mask = build_special_block_mask(
                        discrete_tokens=step_discrete,
                        special_token_ids=special_discrete_token_ids,
                    )
                    if special_step_mask.any():
                        predicted_continuous[
                            continuous_batch_idx[special_step_mask],
                            step_idx,
                            :,
                        ] = 0.0

            active_sample_idx = torch.nonzero(active_step_samples, as_tuple=False).squeeze(1)
            if active_sample_idx.numel() > 0:
                generated_lengths[active_sample_idx] = step_idx + 1
                finished[active_sample_idx] = (
                    generated_lengths[active_sample_idx] >= maskable_lengths[active_sample_idx]
                )
            collected_logits.append(step_logits)

        generated_lengths = speech_target_lengths.clone()
        return predicted_discrete, predicted_continuous, generated_lengths, collected_logits

    @torch.no_grad()
    def generate_parallel(
        self,
        *,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        continuous_prompt: torch.FloatTensor,
        text_target: torch.LongTensor,
        text_prompt_lengths: torch.LongTensor,
        speech_prompt_lengths: torch.LongTensor,
        text_target_lengths: torch.LongTensor,
        speech_target_lengths: torch.LongTensor,
        discrete_eos_id: int,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
        flow_num_steps: Optional[int],
        parallel_num_steps: int,
        special_discrete_token_ids: tuple[int, ...],
    ) -> tuple[
        torch.LongTensor,
        torch.FloatTensor,
        torch.LongTensor,
        list[torch.Tensor],
    ]:
        """
        Block-parallel generation with confidence remasking and fixed terminal EOS block.

        We decode masked non-terminal blocks in parallel, score each block by
        discrete-token confidence, and remask only low-confidence blocks according
        to a cosine schedule.
        """
        batch_size = int(text_prompt.shape[0])
        device = text_prompt.device
        max_target = int(speech_target_lengths.max().item()) if batch_size > 0 else 0
        safe_discrete_fill_id = (
            int(discrete_eos_id)
            if 0 <= int(discrete_eos_id) < self.discrete_vocab_size
            else self.pad_token_id
        )
        terminal_discrete_id = safe_discrete_fill_id

        predicted_discrete = discrete_prompt.new_full(
            (batch_size, max_target, self.num_discrete_tokens),
            safe_discrete_fill_id,
        )
        predicted_continuous = continuous_prompt.new_zeros(
            (batch_size, max_target, self.continuous_latent_size)
        )
        if max_target <= 0:
            return (
                predicted_discrete,
                predicted_continuous,
                speech_target_lengths.new_zeros(batch_size),
                [],
            )

        valid_target_mask = (
            torch.arange(max_target, device=device).unsqueeze(0)
            < speech_target_lengths.unsqueeze(1)
        )
        maskable_target_mask = valid_target_mask.clone()
        has_target = speech_target_lengths > 0
        if has_target.any():
            eos_sample_idx = torch.nonzero(has_target, as_tuple=False).squeeze(1)
            eos_block_idx = speech_target_lengths[eos_sample_idx] - 1
            predicted_discrete[eos_sample_idx, eos_block_idx, :] = terminal_discrete_id
            predicted_continuous[eos_sample_idx, eos_block_idx, :] = 0.0
            maskable_target_mask[eos_sample_idx, eos_block_idx] = False

        masked_blocks = maskable_target_mask.clone()
        block_confidence = torch.full(
            (batch_size, max_target),
            fill_value=-1e9,
            dtype=torch.float32,
            device=device,
        )
        collected_logits: list[torch.Tensor] = []

        # Schedule: progressively shrink masked set to zero.
        num_parallel_steps = int(parallel_num_steps)
        for step_idx in range(num_parallel_steps):
            if not bool(masked_blocks.any().item()):
                break

            current_discrete = predicted_discrete.clone()
            current_continuous = predicted_continuous.clone()
            if masked_blocks.any():
                current_discrete[masked_blocks] = self.pad_token_id
                current_continuous[masked_blocks] = 0.0

            flat = assemble_flat_batch(
                text_prompt=text_prompt,
                discrete_prompt=discrete_prompt,
                continuous_prompt=continuous_prompt,
                text_target=text_target,
                discrete_target=current_discrete,
                continuous_target=current_continuous,
                text_prompt_lengths=text_prompt_lengths,
                speech_prompt_lengths=speech_prompt_lengths,
                text_target_lengths=text_target_lengths,
                speech_target_lengths=speech_target_lengths,
                attention_mask=None,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                eot_token_id=self.eot_token_id,
                continuous_latent_size=self.continuous_latent_size,
                num_discrete_tokens=self.num_discrete_tokens,
            )
            hidden_states, masked_discrete_positions, masked_continuous_positions, _ = self._encode(
                flat=flat,
                masked_target_blocks=masked_blocks,
            )
            batch_indices = (
                torch.arange(batch_size, device=device)
                .unsqueeze(1)
                .expand(batch_size, hidden_states.shape[1])
            )

            if masked_discrete_positions.any():
                discrete_hidden = hidden_states[masked_discrete_positions]
                discrete_logits = self.discrete_lm_head(discrete_hidden)
                sampled_discrete = self._sample_discrete_ids(
                    discrete_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                )
                discrete_batch_idx = batch_indices[masked_discrete_positions]
                discrete_block_idx = flat.target_block_ids[masked_discrete_positions]
                discrete_stream_idx = flat.speech_stream_ids[masked_discrete_positions]
                predicted_discrete[discrete_batch_idx, discrete_block_idx, discrete_stream_idx] = sampled_discrete

                sampled_log_probs = F.log_softmax(discrete_logits, dim=-1).gather(
                    -1,
                    sampled_discrete.unsqueeze(-1),
                ).squeeze(-1)
                step_conf_sum = torch.zeros(
                    (batch_size, max_target),
                    dtype=torch.float32,
                    device=device,
                )
                step_conf_count = torch.zeros(
                    (batch_size, max_target),
                    dtype=torch.float32,
                    device=device,
                )
                step_conf_sum.index_put_(
                    (discrete_batch_idx, discrete_block_idx),
                    sampled_log_probs.to(dtype=torch.float32),
                    accumulate=True,
                )
                step_conf_count.index_put_(
                    (discrete_batch_idx, discrete_block_idx),
                    torch.ones_like(sampled_log_probs, dtype=torch.float32),
                    accumulate=True,
                )
                has_confidence = step_conf_count > 0
                averaged_confidence = torch.zeros_like(step_conf_sum)
                averaged_confidence[has_confidence] = (
                    step_conf_sum[has_confidence] / step_conf_count[has_confidence]
                )
                block_confidence = torch.where(
                    has_confidence,
                    averaged_confidence,
                    block_confidence,
                )
                collected_logits.append(discrete_logits)
            else:
                collected_logits.append(
                    torch.empty(
                        0,
                        self.discrete_vocab_size,
                        dtype=hidden_states.dtype,
                        device=device,
                    )
                )

            if masked_continuous_positions.any():
                continuous_hidden = hidden_states[masked_continuous_positions]
                prior_prediction = self.continuous_prior_head(continuous_hidden)
                sampled_continuous = self.sample_continuous_latent(
                    cond=prior_prediction,
                    num_steps=flow_num_steps,
                )
                continuous_batch_idx = batch_indices[masked_continuous_positions]
                continuous_block_idx = flat.target_block_ids[masked_continuous_positions]
                predicted_continuous[continuous_batch_idx, continuous_block_idx, :] = sampled_continuous

                if len(special_discrete_token_ids) > 0:
                    step_discrete = predicted_discrete[continuous_batch_idx, continuous_block_idx, :]
                    special_step_mask = build_special_block_mask(
                        discrete_tokens=step_discrete,
                        special_token_ids=special_discrete_token_ids,
                    )
                    if special_step_mask.any():
                        predicted_continuous[
                            continuous_batch_idx[special_step_mask],
                            continuous_block_idx[special_step_mask],
                            :,
                        ] = 0.0

            if step_idx >= num_parallel_steps - 1:
                break

            next_masked_blocks = torch.zeros_like(masked_blocks)
            remaining_ratio = math.cos(
                (math.pi * float(step_idx + 1)) / (2.0 * float(num_parallel_steps))
            )
            for sample_idx in range(batch_size):
                target_len = int(speech_target_lengths[sample_idx].item())
                maskable_len = max(0, target_len - 1)
                if maskable_len <= 0:
                    continue

                next_mask_count = int(round(remaining_ratio * float(maskable_len)))
                next_mask_count = min(maskable_len, max(0, next_mask_count))
                if next_mask_count <= 0:
                    continue

                sample_confidence = block_confidence[sample_idx, :maskable_len]
                low_confidence_blocks = torch.topk(
                    sample_confidence,
                    k=next_mask_count,
                    largest=False,
                    dim=-1,
                ).indices
                next_masked_blocks[sample_idx, low_confidence_blocks] = True

            masked_blocks = next_masked_blocks & maskable_target_mask

        generated_lengths = speech_target_lengths.clone()
        return predicted_discrete, predicted_continuous, generated_lengths, collected_logits


    @torch.no_grad()
    def generate_from_raw(
        self,
        raw_text_prompt: str | Sequence[str],
        raw_speech_prompt: Any | list[Any],
        raw_text_target: str | Sequence[str],
        *,
        text_tokenizer: Callable[[str], Sequence[int]],
        speech_prompt_encoder: Callable[[Any], tuple[torch.Tensor, torch.Tensor]],
        return_dict: bool = True,
        **generate_kwargs: Any,
    ) -> PrismTTSGenerationOutput | tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Convenience wrapper around `generate` that accepts raw text/audio-like inputs.

        `text_tokenizer` converts a raw string to token ids.
        `speech_prompt_encoder` converts one raw speech prompt object to:
        - discrete tokens with shape [L, N] or [N, L]
        - continuous latents with shape [L, D] (or [D, L], accepted and transposed)
        where N=num_discrete_tokens and D=continuous_latent_size.

        For batched inference, pass lists for all raw inputs with matching lengths.
        """

        def _normalize_raw_text_batch(
            value: str | Sequence[str],
            name: str,
        ) -> list[str]:
            if isinstance(value, str):
                return [value]
            if not isinstance(value, Sequence):
                raise ValueError(f"{name} must be a string or sequence of strings.")
            out = list(value)
            if len(out) == 0:
                raise ValueError(f"{name} must not be empty.")
            if not all(isinstance(item, str) for item in out):
                raise ValueError(f"{name} sequence must contain only strings.")
            return out

        prompt_text_list = _normalize_raw_text_batch(raw_text_prompt, "raw_text_prompt")
        target_text_list = _normalize_raw_text_batch(raw_text_target, "raw_text_target")

        if isinstance(raw_speech_prompt, list):
            speech_prompt_list = list(raw_speech_prompt)
        else:
            speech_prompt_list = [raw_speech_prompt]

        batch_size = len(prompt_text_list)
        if len(target_text_list) != batch_size:
            raise ValueError("raw_text_target batch size must match raw_text_prompt.")
        if len(speech_prompt_list) != batch_size:
            raise ValueError("raw_speech_prompt batch size must match raw_text_prompt.")

        params = list(self.parameters())
        if len(params) == 0:
            raise RuntimeError("Model has no parameters.")
        device = params[0].device
        continuous_dtype = self.continuous_proj.weight.dtype

        prompt_token_lists: list[list[int]] = []
        target_token_lists: list[list[int]] = []
        speech_discrete_list: list[torch.LongTensor] = []
        speech_continuous_list: list[torch.FloatTensor] = []

        for sample_idx in range(batch_size):
            prompt_tokens = [int(token) for token in text_tokenizer(prompt_text_list[sample_idx])]
            target_tokens = [int(token) for token in text_tokenizer(target_text_list[sample_idx])]
            prompt_token_lists.append(prompt_tokens)
            target_token_lists.append(target_tokens)

            encoded = speech_prompt_encoder(speech_prompt_list[sample_idx])
            if not isinstance(encoded, tuple) or len(encoded) != 2:
                raise ValueError(
                    "speech_prompt_encoder must return a tuple of "
                    "(discrete_tokens, continuous_latents)."
                )
            discrete_tokens, continuous_latents = encoded

            discrete = torch.as_tensor(discrete_tokens, device=device, dtype=torch.long)
            if discrete.dim() == 3 and discrete.shape[0] == 1:
                discrete = discrete.squeeze(0)
            if discrete.dim() != 2:
                raise ValueError(
                    "Encoded discrete prompt must have shape [L, N] or [N, L]."
                )
            if discrete.shape[1] == self.num_discrete_tokens:
                discrete = discrete.contiguous()
            elif discrete.shape[0] == self.num_discrete_tokens:
                discrete = discrete.transpose(0, 1).contiguous()
            else:
                raise ValueError(
                    "Encoded discrete prompt must contain one axis with "
                    f"num_discrete_tokens={self.num_discrete_tokens}, got {tuple(discrete.shape)}."
                )

            continuous = torch.as_tensor(
                continuous_latents,
                device=device,
                dtype=continuous_dtype,
            )
            if continuous.dim() == 3 and continuous.shape[0] == 1:
                continuous = continuous.squeeze(0)
            if continuous.dim() == 1:
                if self.continuous_latent_size != 1:
                    raise ValueError(
                        "Encoded continuous prompt with shape [L] is only valid when "
                        "continuous_latent_size == 1."
                    )
                continuous = continuous.unsqueeze(-1)
            if continuous.dim() != 2:
                raise ValueError(
                    "Encoded continuous prompt must have shape [L, D] or [D, L]."
                )
            if (
                continuous.shape[0] == self.continuous_latent_size
                and continuous.shape[1] == discrete.shape[0]
            ):
                continuous = continuous.transpose(0, 1).contiguous()
            if continuous.shape[0] != discrete.shape[0]:
                raise ValueError(
                    "Speech prompt length mismatch between discrete and continuous tensors: "
                    f"{discrete.shape[0]} vs {continuous.shape[0]}."
                )
            if continuous.shape[1] != self.continuous_latent_size:
                raise ValueError(
                    "Encoded continuous prompt channel mismatch: expected "
                    f"{self.continuous_latent_size}, got {continuous.shape[1]}."
                )

            speech_discrete_list.append(discrete)
            speech_continuous_list.append(continuous)

        max_prompt_text = max(len(tokens) for tokens in prompt_token_lists) if batch_size > 0 else 0
        max_target_text = max(len(tokens) for tokens in target_token_lists) if batch_size > 0 else 0
        max_speech_prompt = max(
            int(discrete.shape[0]) for discrete in speech_discrete_list
        ) if batch_size > 0 else 0

        text_prompt = torch.full(
            (batch_size, max_prompt_text),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        text_target = torch.full(
            (batch_size, max_target_text),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        discrete_prompt = torch.full(
            (batch_size, self.num_discrete_tokens, max_speech_prompt),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        continuous_prompt = torch.zeros(
            (batch_size, max_speech_prompt, self.continuous_latent_size),
            dtype=continuous_dtype,
            device=device,
        )

        text_prompt_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        speech_prompt_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        text_target_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

        for sample_idx in range(batch_size):
            prompt_tokens = prompt_token_lists[sample_idx]
            target_tokens = target_token_lists[sample_idx]
            speech_discrete = speech_discrete_list[sample_idx]
            speech_continuous = speech_continuous_list[sample_idx]

            text_prompt_lengths[sample_idx] = int(len(prompt_tokens))
            speech_prompt_lengths[sample_idx] = int(speech_discrete.shape[0])
            text_target_lengths[sample_idx] = int(len(target_tokens))

            if len(prompt_tokens) > 0:
                text_prompt[sample_idx, : len(prompt_tokens)] = torch.tensor(
                    prompt_tokens,
                    dtype=torch.long,
                    device=device,
                )
            if len(target_tokens) > 0:
                text_target[sample_idx, : len(target_tokens)] = torch.tensor(
                    target_tokens,
                    dtype=torch.long,
                    device=device,
                )
            if speech_discrete.shape[0] > 0:
                discrete_prompt[sample_idx, :, : speech_discrete.shape[0]] = speech_discrete.transpose(
                    0, 1
                )
                continuous_prompt[sample_idx, : speech_continuous.shape[0], :] = speech_continuous

        if "text_prompt_lengths" in generate_kwargs:
            raise ValueError("Do not pass text_prompt_lengths when using generate_from_raw.")
        if "speech_prompt_lengths" in generate_kwargs:
            raise ValueError("Do not pass speech_prompt_lengths when using generate_from_raw.")
        if "text_target_lengths" in generate_kwargs:
            raise ValueError("Do not pass text_target_lengths when using generate_from_raw.")
        if "return_dict" in generate_kwargs:
            raise ValueError("Use generate_from_raw(return_dict=...) instead of return_dict in kwargs.")

        return self._generate_prepared(
            text_prompt=text_prompt,
            discrete_prompt=normalize_discrete_tokens(
                discrete_prompt,
                "discrete_prompt",
                num_discrete_tokens=self.num_discrete_tokens,
            ),
            continuous_prompt=normalize_continuous_latents(
                continuous_prompt,
                expected_len=int(max_speech_prompt),
                name="continuous_prompt",
                continuous_latent_size=self.continuous_latent_size,
            ),
            text_target=normalize_text_tokens(text_target, "text_target"),
            text_prompt_lengths=text_prompt_lengths,
            speech_prompt_lengths=speech_prompt_lengths,
            text_target_lengths=text_target_lengths,
            return_dict=return_dict,
            **generate_kwargs,
        )
