from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig

from models.flow_head import FlowHead
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
        self.masked_discrete_embeddings = nn.Parameter(
            torch.empty(self.num_discrete_tokens, self.hidden_size)
        )
        self.masked_continuous_embedding = nn.Parameter(torch.empty(self.hidden_size))

        # Flow head is timestep-conditioned; prior enters via x0 construction.
        self.flow_head = FlowHead(
            in_channels=self.continuous_latent_size,
            model_channels=flow_model_channels or self.hidden_size,
            out_channels=self.continuous_latent_size,
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
        inject_continuous_noise: bool = False,
    ) -> tuple[torch.FloatTensor, torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
        """Build model input embeddings and masks for masked discrete/continuous targets."""
        batch_size, seq_len = flat.token_ids.shape
        device = flat.token_ids.device

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
        continuous_values = flat.continuous_values
        if inject_continuous_noise:
            continuous_values = MU.inject_continuous_backbone_noise(continuous_values)
        continuous_embeds = self.continuous_proj(continuous_values)
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

        return (
            base_embeds,
            masked_discrete_positions,
            masked_continuous_positions,
            masked_target_token_mask,
        )

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

        logits = hidden_states.new_empty((hidden_states.shape[0], self.discrete_vocab_size))
        for stream_idx, lm_head in enumerate(self.discrete_lm_heads):
            stream_mask = discrete_stream_ids == stream_idx
            if stream_mask.any():
                logits[stream_mask] = lm_head(hidden_states[stream_mask])
        return logits

    def _compute_continuous_losses(
        self,
        hidden_states: torch.FloatTensor,
        continuous_values: torch.FloatTensor,
        token_type_ids: torch.LongTensor,
        target_block_ids: torch.LongTensor,
        target_block_counts: torch.LongTensor,
        masked_continuous_positions: torch.BoolTensor,
        flow_timesteps: Optional[torch.FloatTensor],
        noise: Optional[torch.FloatTensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute continuous losses with sequence-level latent mixing.

        Training flow:
        1) Predict per-target-block priors with the backbone hidden states.
        2) Build x0 = prior + eps and xt = t * x1 + (1 - t) * x0, where x1 is clean latent.
        3) Mix clean and noised-prior latents by replacing only masked blocks with xt.
        4) Run DiT flow head over the mixed sequence and compute flow loss on masked blocks only.
        """
        if not masked_continuous_positions.any():
            zero = hidden_states.new_zeros(())
            return zero, zero

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        zero = hidden_states.new_zeros(())

        target_continuous_positions = (
            (target_block_ids >= 0)
            & (token_type_ids == MU.SPEECH_CONTINUOUS_TOKEN_TYPE)
        )
        if not target_continuous_positions.any():
            return zero, zero

        max_target_blocks = int(target_block_counts.max().item()) if batch_size > 0 else 0
        if max_target_blocks <= 0:
            return zero, zero

        batch_index_grid = (
            torch.arange(batch_size, device=hidden_states.device)
            .unsqueeze(1)
            .expand(batch_size, seq_len)
        )
        target_batch_indices = batch_index_grid[target_continuous_positions]
        target_block_indices = target_block_ids[target_continuous_positions]
        if target_block_indices.numel() > 0 and int(target_block_indices.max().item()) >= max_target_blocks:
            raise ValueError("target_block_ids exceed target_block_counts.")

        target_clean_latents = continuous_values[target_continuous_positions]
        predicted_continuous_latents = self.continuous_prior_head(hidden_states)
        target_predicted_continuous_latents = predicted_continuous_latents[
            target_continuous_positions
        ]
        target_masked_flags = masked_continuous_positions[target_continuous_positions]

        if target_masked_flags.any():
            reconstruction_loss = F.mse_loss(
                target_predicted_continuous_latents[target_masked_flags],
                target_clean_latents[target_masked_flags],
            )
        else:
            reconstruction_loss = zero

        clean_by_block = hidden_states.new_zeros(
            (batch_size, max_target_blocks, self.continuous_latent_size)
        )
        prior_by_block = hidden_states.new_zeros(
            (batch_size, max_target_blocks, self.continuous_latent_size)
        )
        valid_by_block = torch.zeros(batch_size, max_target_blocks, dtype=torch.bool, device=device)
        masked_by_block = torch.zeros(batch_size, max_target_blocks, dtype=torch.bool, device=device)

        clean_by_block[target_batch_indices, target_block_indices] = target_clean_latents
        prior_by_block[target_batch_indices, target_block_indices] = (
            target_predicted_continuous_latents
        )
        valid_by_block[target_batch_indices, target_block_indices] = True
        masked_by_block[target_batch_indices, target_block_indices] = target_masked_flags

        expected_valid = (
            torch.arange(max_target_blocks, device=device, dtype=torch.long).unsqueeze(0)
            < target_block_counts.to(device=device, dtype=torch.long).unsqueeze(1)
        )
        valid_by_block = valid_by_block & expected_valid
        masked_by_block = masked_by_block & valid_by_block
        if not masked_by_block.any():
            return reconstruction_loss, zero

        if flow_timesteps is not None:
            if flow_timesteps.dim() != 2 or flow_timesteps.shape[0] != batch_size:
                raise ValueError("flow_timesteps must have shape [batch, target_blocks].")
            if flow_timesteps.shape[1] < max_target_blocks:
                raise ValueError(
                    "flow_timesteps length must cover all target speech blocks."
                )
            sampled_timesteps = flow_timesteps[:, :max_target_blocks].to(
                device=device,
                dtype=clean_by_block.dtype,
            )
        else:
            sampled_timesteps = torch.rand(
                (batch_size, max_target_blocks),
                device=device,
                dtype=clean_by_block.dtype,
            )

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
            if noise.shape[1] < max_target_blocks:
                raise ValueError("noise must cover all target speech blocks.")
            sampled_noise = noise[:, :max_target_blocks, :].to(
                device=device,
                dtype=clean_by_block.dtype,
            )
        else:
            sampled_noise = torch.randn_like(clean_by_block)

        x1 = clean_by_block
        x0 = prior_by_block + sampled_noise
        t = sampled_timesteps.unsqueeze(-1)
        xt = t * x1 + (1.0 - t) * x0

        mixed_inputs = x1.clone()
        mixed_inputs[masked_by_block] = xt[masked_by_block]
        flow_target = x1 - x0

        flow_prediction = self.flow_head(
            mixed_inputs,
            sampled_timesteps,
            mask=valid_by_block,
        )
        flow_loss = F.mse_loss(
            flow_prediction[masked_by_block],
            flow_target[masked_by_block],
        )
        return reconstruction_loss, flow_loss

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
        """Generate continuous latents by integrating from x0 = prior + eps."""
        squeeze_seq_dim = False
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
            squeeze_seq_dim = True
        elif cond.dim() != 3:
            raise ValueError("cond must have shape [batch, channels] or [batch, seq, channels].")

        all_target_mask = torch.ones(
            (cond.shape[0], cond.shape[1]),
            dtype=torch.bool,
            device=cond.device,
        )
        empty_prompt = cond.new_empty((cond.shape[0], 0, self.continuous_latent_size))
        empty_prompt_lengths = torch.zeros(
            cond.shape[0],
            dtype=torch.long,
            device=cond.device,
        )
        sampled = self._sample_continuous_with_clean_context(
            prior_target=cond,
            clean_target=torch.zeros_like(cond),
            denoise_target_mask=all_target_mask,
            valid_target_mask=all_target_mask,
            prompt_latents=empty_prompt,
            prompt_lengths=empty_prompt_lengths,
            num_steps=num_steps,
            temperature=temperature,
        )

        if squeeze_seq_dim:
            return sampled.squeeze(1)
        return sampled

    def _sample_continuous_with_clean_context(
        self,
        *,
        prior_target: torch.FloatTensor,
        clean_target: torch.FloatTensor,
        denoise_target_mask: torch.BoolTensor,
        valid_target_mask: torch.BoolTensor,
        prompt_latents: torch.FloatTensor,
        prompt_lengths: torch.LongTensor,
        num_steps: Optional[int],
        temperature: float = 1.0,
    ) -> torch.FloatTensor:
        """
        Denoise selected target latents while keeping prompt/clean context fixed.

        This mirrors `_compute_continuous_losses` inference semantics:
        - initialize denoised positions from `prior + noise`
        - keep all non-denoised valid positions fixed as clean context
        - integrate velocity only on denoised positions
        """
        num_steps = num_steps or self.flow_sample_steps
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")
        if prior_target.dim() != 3:
            raise ValueError("prior_target must have shape [batch, target_len, channels].")
        if clean_target.shape != prior_target.shape:
            raise ValueError("clean_target must match prior_target shape.")
        if denoise_target_mask.shape != prior_target.shape[:2]:
            raise ValueError("denoise_target_mask must have shape [batch, target_len].")
        if valid_target_mask.shape != prior_target.shape[:2]:
            raise ValueError("valid_target_mask must have shape [batch, target_len].")
        if prompt_latents.dim() != 3:
            raise ValueError("prompt_latents must have shape [batch, prompt_len, channels].")
        if prompt_latents.shape[0] != prior_target.shape[0]:
            raise ValueError("prompt_latents batch size must match prior_target.")
        if prompt_latents.shape[2] != prior_target.shape[2]:
            raise ValueError("prompt_latents channel size must match prior_target.")
        if prompt_lengths.dim() != 1 or prompt_lengths.shape[0] != prior_target.shape[0]:
            raise ValueError("prompt_lengths must have shape [batch].")
        noise_temperature = max(0.0, float(temperature))

        batch_size, target_len, latent_size = prior_target.shape
        prompt_len = int(prompt_latents.shape[1])
        device = prior_target.device

        resolved_valid_target_mask = valid_target_mask.to(device=device, dtype=torch.bool)
        resolved_denoise_target_mask = denoise_target_mask.to(device=device, dtype=torch.bool)
        resolved_denoise_target_mask = resolved_denoise_target_mask & resolved_valid_target_mask
        if not resolved_denoise_target_mask.any():
            return clean_target

        full_len = prompt_len + target_len
        clean_sequence = prior_target.new_zeros((batch_size, full_len, latent_size))
        if prompt_len > 0:
            clean_sequence[:, :prompt_len, :] = prompt_latents
        clean_sequence[:, prompt_len:, :] = clean_target

        full_valid_mask = torch.zeros((batch_size, full_len), dtype=torch.bool, device=device)
        if prompt_len > 0:
            clamped_prompt_lengths = prompt_lengths.to(
                device=device,
                dtype=torch.long,
            ).clamp(min=0, max=prompt_len)
            prompt_positions = torch.arange(prompt_len, device=device).unsqueeze(0)
            full_valid_mask[:, :prompt_len] = prompt_positions < clamped_prompt_lengths.unsqueeze(1)
        full_valid_mask[:, prompt_len:] = resolved_valid_target_mask

        full_denoise_mask = torch.zeros_like(full_valid_mask)
        full_denoise_mask[:, prompt_len:] = resolved_denoise_target_mask
        fixed_clean_mask = full_valid_mask & (~full_denoise_mask)

        x = clean_sequence.clone()
        noisy_target = prior_target + (noise_temperature * torch.randn_like(prior_target))
        target_x = x[:, prompt_len:, :]
        target_x[resolved_denoise_target_mask] = noisy_target[resolved_denoise_target_mask]
        x[:, prompt_len:, :] = target_x

        dt = 1.0 / num_steps
        has_invalid_positions = not bool(full_valid_mask.all().item())
        for step_idx in range(num_steps):
            t = torch.full(
                (batch_size,),
                step_idx / num_steps,
                device=device,
                dtype=prior_target.dtype,
            )
            velocity = self.flow_head(x, t, mask=full_valid_mask)
            x[full_denoise_mask] = x[full_denoise_mask] + dt * velocity[full_denoise_mask]
            x[fixed_clean_mask] = clean_sequence[fixed_clean_mask]
            if has_invalid_positions:
                x = x.masked_fill(~full_valid_mask.unsqueeze(-1), 0.0)

        denoised_target = clean_target.clone()
        final_target = x[:, prompt_len:, :]
        denoised_target[resolved_denoise_target_mask] = final_target[resolved_denoise_target_mask]
        return denoised_target

    def _encode(
        self,
        flat: MU.FlatBatch,
        masked_target_blocks: torch.BoolTensor,
        attention_mask: Optional[torch.BoolTensor] = None,
        inject_continuous_noise: bool = False,
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
                inject_continuous_noise=inject_continuous_noise,
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
        attention_mask: Optional[torch.Tensor] = None,
        flow_timesteps: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        mask_ratio: Optional[float] = None,
        masked_target_blocks: Optional[torch.BoolTensor] = None,
        return_dict: bool = True,
    ) -> MU.PrismTTSOutput | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            # Default training behavior: sample mask ratio uniformly per forward pass.
            effective_mask_ratio = 1 - float(torch.rand((), device=flat.token_ids.device).item())**2
        else:
            effective_mask_ratio = float(mask_ratio)
            if not (0.0 <= effective_mask_ratio <= 1.0):
                raise ValueError("mask_ratio must be in [0, 1].")
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
            inject_continuous_noise=True,
        )

        discrete_loss, _ = self._compute_discrete_loss(
            hidden_states=hidden_states,
            token_ids=flat.token_ids,
            speech_stream_ids=flat.speech_stream_ids,
            masked_discrete_positions=masked_discrete_positions,
        )
        continuous_loss, flow_loss = self._compute_continuous_losses(
            hidden_states=hidden_states,
            continuous_values=flat.continuous_values,
            token_type_ids=flat.token_type_ids,
            target_block_ids=flat.target_block_ids,
            target_block_counts=flat.target_block_counts,
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

        return MU.PrismTTSOutput(
            loss=loss,
            discrete_loss=discrete_loss,
            continuous_loss=continuous_loss,
            flow_loss=flow_loss,
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
        flow_num_steps: Optional[int] = None,
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
                flow_num_steps=flow_num_steps,
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
            flow_num_steps=flow_num_steps,
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
