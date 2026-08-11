from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig

from models.flow_head import FlowHead
from models.llama_backbone import LlamaBackbone
from utils import model_utils as MU


class PrismTTS(nn.Module):
    """Autoregressive fused-frame Prism-TTS model.

    Speech is represented by one token per codec frame. Each input frame fuses
    every discrete stream with its continuous latent additively in the model
    hidden space.
    Target frame ``t`` is predicted from the preceding position, which makes
    teacher-forced training and autoregressive rollout use the same context.
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
        normalize_continuous_latents: bool | int | str = False,
        continuous_latent_mean: float | Sequence[float] | torch.Tensor = 0.0,
        continuous_latent_std: float | Sequence[float] | torch.Tensor = 1.0,
        continuous_latent_std_eps: float = 1e-6,
    ) -> None:
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
        if llama_config.hidden_size < 1:
            raise ValueError("llama_config.hidden_size must be positive.")
        if flow_loss_weight < 0.0 or continuous_loss_weight < 0.0:
            raise ValueError("continuous and flow loss weights must be >= 0.")
        if discrete_regular_token_loss_weight < 0.0 or discrete_special_token_loss_weight < 0.0:
            raise ValueError("discrete loss weights must be >= 0.")
        if discrete_regular_token_loss_weight == 0.0 and discrete_special_token_loss_weight == 0.0:
            raise ValueError("At least one discrete loss weight must be > 0.")
        if flow_sample_steps < 1:
            raise ValueError("flow_sample_steps must be at least 1.")
        if continuous_latent_std_eps <= 0.0:
            raise ValueError("continuous_latent_std_eps must be > 0.")

        self.hidden_size = int(llama_config.hidden_size)
        self.num_discrete_tokens = int(num_discrete_tokens)
        self.discrete_vocab_size = int(discrete_vocab_size)
        self.continuous_latent_size = int(continuous_latent_size)
        self.flow_loss_weight = float(flow_loss_weight)
        self.continuous_loss_weight = float(continuous_loss_weight)
        self.discrete_regular_token_loss_weight = float(discrete_regular_token_loss_weight)
        self.discrete_special_token_loss_weight = float(discrete_special_token_loss_weight)
        self.flow_sample_steps = int(flow_sample_steps)
        self.normalize_continuous_latents = self._coerce_bool(
            normalize_continuous_latents,
            name="normalize_continuous_latents",
        )
        self.continuous_latent_std_eps = float(continuous_latent_std_eps)

        self.backbone = LlamaBackbone(llama_config)
        # Both modalities must occupy the full backbone width before they can
        # be fused by addition. Both output heads consume that full fused state.
        self.discrete_frame_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.continuous_proj = nn.Linear(self.continuous_latent_size, self.hidden_size)
        self.discrete_lm_head = nn.Linear(
            self.hidden_size,
            self.num_discrete_tokens * self.discrete_vocab_size,
            bias=False,
        )
        self.continuous_prior_head = nn.Linear(
            self.hidden_size,
            self.continuous_latent_size,
        )

        self.discrete_stream_embeddings = nn.Parameter(
            torch.empty(self.num_discrete_tokens, self.hidden_size)
        )
        self.flow_head = FlowHead(
            in_channels=self.continuous_latent_size,
            model_channels=flow_model_channels or self.hidden_size,
            out_channels=self.continuous_latent_size,
            z_channels=self.continuous_latent_size,
            num_res_blocks=flow_num_res_blocks,
        )

        vocab_size = int(self.backbone.config.vocab_size)
        self.pad_token_id = int(self.backbone.config.pad_token_id or 0)
        if not 0 <= self.pad_token_id < vocab_size:
            self.pad_token_id = 0
        self.eos_token_id = int(self.backbone.config.eos_token_id or 0)
        if not 0 <= self.eos_token_id < vocab_size:
            self.eos_token_id = 0
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
        mean = self._coerce_continuous_latent_stat(
            continuous_latent_mean,
            name="continuous_latent_mean",
            continuous_latent_size=self.continuous_latent_size,
        )
        std = self._coerce_continuous_latent_stat(
            continuous_latent_std,
            name="continuous_latent_std",
            continuous_latent_size=self.continuous_latent_size,
        )
        if not torch.isfinite(mean).all():
            raise ValueError("continuous_latent_mean must contain only finite values.")
        if not torch.isfinite(std).all():
            raise ValueError("continuous_latent_std must contain only finite values.")
        if torch.le(std, self.continuous_latent_std_eps).any():
            raise ValueError(
                "continuous_latent_std values must be greater than continuous_latent_std_eps."
            )
        stat_device = self.continuous_proj.weight.device
        self.register_buffer(
            "continuous_latent_mean",
            mean.to(device=stat_device),
            persistent=False,
        )
        self.register_buffer(
            "continuous_latent_std",
            std.to(device=stat_device),
            persistent=False,
        )
        self.reset_parameters()

    @staticmethod
    def _coerce_bool(value: bool | int | str, *, name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        raise ValueError(f"{name} must be a boolean value.")

    @staticmethod
    def _coerce_continuous_latent_stat(
        value: float | Sequence[float] | torch.Tensor,
        *,
        name: str,
        continuous_latent_size: int,
    ) -> torch.FloatTensor:
        stat = torch.as_tensor(value, dtype=torch.float32, device="cpu")
        if stat.dim() == 0 or stat.numel() == 1:
            return stat.reshape(1).repeat(int(continuous_latent_size)).contiguous()
        if stat.dim() == 1 and int(stat.numel()) == int(continuous_latent_size):
            return stat.contiguous()
        raise ValueError(
            f"{name} must be a scalar or a 1D sequence with "
            f"continuous_latent_size={continuous_latent_size} values."
        )

    def reset_parameters(self) -> None:
        """Initialize fused projections and task heads without resetting the backbone."""
        std = float(getattr(self.backbone.config, "initializer_range", 0.02))
        for layer in (
            self.discrete_frame_proj,
            self.continuous_proj,
            self.discrete_lm_head,
            self.continuous_prior_head,
        ):
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.normal_(self.discrete_stream_embeddings, mean=0.0, std=std)

    @property
    def text_embedding(self) -> nn.Embedding:
        return self.backbone.embed_tokens

    @property
    def discrete_embedding(self) -> nn.Embedding:
        return self.backbone.embed_tokens

    def _continuous_stat_view(self, stat: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        view_shape = (1,) * (latents.dim() - 1) + (self.continuous_latent_size,)
        return stat.to(device=latents.device, dtype=latents.dtype).view(view_shape)

    def _normalize_continuous_latent_values(
        self,
        latents: torch.FloatTensor,
        *,
        payload_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        if not self.normalize_continuous_latents:
            return latents
        mean = self._continuous_stat_view(self.continuous_latent_mean, latents)
        std = self._continuous_stat_view(self.continuous_latent_std, latents)
        normalized = (latents - mean) / std
        if payload_mask is None:
            return normalized
        return torch.where(payload_mask.unsqueeze(-1), normalized, latents)

    def _denormalize_continuous_latent_values(
        self,
        latents: torch.FloatTensor,
        *,
        payload_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        if not self.normalize_continuous_latents:
            return latents
        mean = self._continuous_stat_view(self.continuous_latent_mean, latents)
        std = self._continuous_stat_view(self.continuous_latent_std, latents)
        denormalized = latents * std + mean
        if payload_mask is None:
            return denormalized
        return torch.where(payload_mask.unsqueeze(-1), denormalized, latents)

    def _non_special_discrete_block_mask(
        self,
        discrete_values: torch.LongTensor,
    ) -> torch.BoolTensor:
        if not self.training_special_discrete_token_ids:
            return torch.ones(
                discrete_values.shape[:-1],
                dtype=torch.bool,
                device=discrete_values.device,
            )
        return ~MU.build_special_block_mask(
            discrete_values,
            self.training_special_discrete_token_ids,
        )

    def _continuous_input_payload_mask(
        self,
        *,
        discrete_values: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.BoolTensor:
        mask = token_type_ids == MU.SPEECH_FRAME_TOKEN_TYPE
        mask = mask & self._non_special_discrete_block_mask(discrete_values)
        if attention_mask is not None:
            mask = mask & attention_mask.to(device=mask.device, dtype=torch.bool)
        return mask

    def _normalize_autoregressive_batch(
        self,
        flat: MU.AutoregressiveBatch,
    ) -> MU.AutoregressiveBatch:
        if not self.normalize_continuous_latents:
            return flat
        input_payload_mask = self._continuous_input_payload_mask(
            discrete_values=flat.discrete_values,
            token_type_ids=flat.token_type_ids,
            attention_mask=flat.attention_mask,
        )
        target_payload_mask = (
            flat.attention_mask
            & (flat.target_block_ids >= 0)
            & self._non_special_discrete_block_mask(flat.target_discrete_values)
        )
        return replace(
            flat,
            continuous_values=self._normalize_continuous_latent_values(
                flat.continuous_values,
                payload_mask=input_payload_mask,
            ),
            target_continuous_values=self._normalize_continuous_latent_values(
                flat.target_continuous_values,
                payload_mask=target_payload_mask,
            ),
        )

    def _build_inputs_embeds(
        self,
        *,
        token_ids: torch.LongTensor,
        discrete_values: torch.LongTensor,
        continuous_values: torch.FloatTensor,
        token_type_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Build text and fused speech-frame embeddings."""
        is_frame = token_type_ids == MU.SPEECH_FRAME_TOKEN_TYPE
        text_embeds = self.text_embedding(token_ids)

        safe_discrete = discrete_values.clamp(min=0, max=self.backbone.config.vocab_size - 1)
        per_stream = self.discrete_embedding(safe_discrete)
        per_stream = per_stream + self.discrete_stream_embeddings.view(
            1,
            1,
            self.num_discrete_tokens,
            self.hidden_size,
        )
        discrete_frame = per_stream.sum(dim=2) / math.sqrt(float(self.num_discrete_tokens))
        discrete_frame = self.discrete_frame_proj(discrete_frame)
        continuous_frame = self.continuous_proj(continuous_values)
        fused_frame = discrete_frame + continuous_frame

        return torch.where(is_frame.unsqueeze(-1), fused_frame, text_embeds)

    def _run_backbone(
        self,
        *,
        token_ids: torch.LongTensor,
        discrete_values: torch.LongTensor,
        continuous_values: torch.FloatTensor,
        token_type_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Any = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> Any:
        inputs_embeds = self._build_inputs_embeds(
            token_ids=token_ids,
            discrete_values=discrete_values,
            continuous_values=continuous_values,
            token_type_ids=token_type_ids,
        )
        batch_size, seq_len, _ = inputs_embeds.shape
        position_ids = torch.arange(
            position_offset,
            position_offset + seq_len,
            device=inputs_embeds.device,
        ).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.backbone.rotary_emb(inputs_embeds, position_ids=position_ids)
        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )

    def _encode(self, flat: MU.AutoregressiveBatch) -> torch.FloatTensor:
        inputs_embeds = self._build_inputs_embeds(
            token_ids=flat.token_ids,
            discrete_values=flat.discrete_values,
            continuous_values=flat.continuous_values,
            token_type_ids=flat.token_type_ids,
        )
        causal_mask = MU.build_causal_attention_mask(
            flat.attention_mask,
            dtype=inputs_embeds.dtype,
        )
        batch_size, seq_len, _ = inputs_embeds.shape
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.backbone.rotary_emb(inputs_embeds, position_ids=position_ids)
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def _discrete_logits(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        logits = self.discrete_lm_head(hidden_states)
        return logits.view(*logits.shape[:-1], self.num_discrete_tokens, self.discrete_vocab_size)

    def _compute_discrete_loss(
        self,
        *,
        hidden_states: torch.FloatTensor,
        target_discrete: torch.LongTensor,
        prediction_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.FloatTensor]:
        if not prediction_mask.any():
            zero = hidden_states.new_zeros(())
            return zero, hidden_states.new_zeros(
                (0, self.num_discrete_tokens, self.discrete_vocab_size)
            )

        selected_hidden = hidden_states[prediction_mask]
        selected_targets = target_discrete[prediction_mask]
        logits = self._discrete_logits(selected_hidden)
        flat_logits = logits.flatten(0, 1)
        flat_targets = selected_targets.reshape(-1)
        per_token_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        weights = torch.full_like(
            per_token_loss,
            self.discrete_regular_token_loss_weight,
        )
        if (
            self.discrete_special_token_loss_weight != self.discrete_regular_token_loss_weight
            and self.training_special_discrete_token_ids
        ):
            special_ids = torch.tensor(
                self.training_special_discrete_token_ids,
                dtype=flat_targets.dtype,
                device=flat_targets.device,
            )
            weights = torch.where(
                torch.isin(flat_targets, special_ids),
                weights.new_full(weights.shape, self.discrete_special_token_loss_weight),
                weights,
            )
        return (per_token_loss * weights).sum() / weights.sum().clamp_min(1e-12), logits

    def _compute_continuous_losses(
        self,
        *,
        hidden_states: torch.FloatTensor,
        target_continuous: torch.FloatTensor,
        target_block_ids: torch.LongTensor,
        prediction_mask: torch.BoolTensor,
        flow_timesteps: Optional[torch.FloatTensor],
        noise: Optional[torch.FloatTensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not prediction_mask.any():
            zero = hidden_states.new_zeros(())
            return zero, zero

        selected_hidden = hidden_states[prediction_mask]
        prior_prediction = self.continuous_prior_head(selected_hidden)
        selected_target = target_continuous[prediction_mask]
        reconstruction_loss = F.mse_loss(prior_prediction, selected_target)

        batch_size, seq_len = prediction_mask.shape
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device).view(-1, 1)
        batch_indices = batch_indices.expand(batch_size, seq_len)[prediction_mask]
        block_indices = target_block_ids[prediction_mask]

        selected_timesteps = None
        if flow_timesteps is not None:
            if flow_timesteps.dim() != 2 or flow_timesteps.shape[0] != batch_size:
                raise ValueError("flow_timesteps must have shape [batch, target_blocks].")
            selected_timesteps = flow_timesteps[batch_indices, block_indices]

        selected_noise = None
        if noise is not None:
            if noise.dim() != 3 or noise.shape[0] != batch_size:
                raise ValueError("noise must have shape [batch, target_blocks, continuous_latent_size].")
            if noise.shape[2] != self.continuous_latent_size:
                raise ValueError("noise channel mismatch for continuous latent size.")
            selected_noise = noise[batch_indices, block_indices, :]

        flow_inputs, flow_target, timesteps = MU.sample_flow_training_inputs(
            continuous_targets=selected_target,
            flow_timesteps=selected_timesteps,
            noise=selected_noise,
        )
        flow_prediction = self.flow_head(flow_inputs, timesteps, prior_prediction)
        return reconstruction_loss, F.mse_loss(flow_prediction, flow_target)

    def forward(
        self,
        flat_token_ids: torch.LongTensor,
        flat_discrete_values: torch.LongTensor,
        flat_continuous_values: torch.FloatTensor,
        flat_token_type_ids: torch.LongTensor,
        flat_target_discrete_values: torch.LongTensor,
        flat_target_continuous_values: torch.FloatTensor,
        flat_target_block_ids: torch.LongTensor,
        flat_target_block_counts: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        flow_timesteps: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        **legacy_masking_kwargs: Any,
    ) -> MU.PrismTTSOutput | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute shifted autoregressive CE, latent prior, and flow-matching losses."""
        if legacy_masking_kwargs:
            unsupported = ", ".join(sorted(legacy_masking_kwargs))
            raise TypeError(
                "Masked-model arguments are not supported by the autoregressive PrismTTS model: "
                f"{unsupported}."
            )
        flat = MU.build_autoregressive_batch_from_collate(
            flat_token_ids=flat_token_ids,
            flat_discrete_values=flat_discrete_values,
            flat_continuous_values=flat_continuous_values,
            flat_token_type_ids=flat_token_type_ids,
            flat_target_discrete_values=flat_target_discrete_values,
            flat_target_continuous_values=flat_target_continuous_values,
            flat_target_block_ids=flat_target_block_ids,
            flat_target_block_counts=flat_target_block_counts,
            attention_mask=attention_mask,
            num_discrete_tokens=self.num_discrete_tokens,
            continuous_latent_size=self.continuous_latent_size,
        )
        flat = self._normalize_autoregressive_batch(flat)
        hidden_states = self._encode(flat)
        prediction_mask = flat.attention_mask & (flat.target_block_ids >= 0)
        discrete_loss, _ = self._compute_discrete_loss(
            hidden_states=hidden_states,
            target_discrete=flat.target_discrete_values,
            prediction_mask=prediction_mask,
        )
        continuous_loss, flow_loss = self._compute_continuous_losses(
            hidden_states=hidden_states,
            target_continuous=flat.target_continuous_values,
            target_block_ids=flat.target_block_ids,
            prediction_mask=prediction_mask,
            flow_timesteps=flow_timesteps,
            noise=noise,
        )
        loss = discrete_loss + self.continuous_loss_weight * continuous_loss + self.flow_loss_weight * flow_loss
        if not return_dict:
            return loss, discrete_loss, continuous_loss, flow_loss
        return MU.PrismTTSOutput(
            loss=loss,
            discrete_loss=discrete_loss,
            continuous_loss=continuous_loss,
            flow_loss=flow_loss,
        )

    def _sample_discrete_ids(
        self,
        logits: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> torch.LongTensor:
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
    ) -> torch.FloatTensor:
        """Integrate the conditional flow field from Gaussian noise."""
        steps = self.flow_sample_steps if num_steps is None else int(num_steps)
        if steps < 1:
            raise ValueError("num_steps must be at least 1.")
        cond_shape = cond.shape[:-1]
        flat_cond = cond.reshape(-1, cond.shape[-1])
        x = torch.randn(
            flat_cond.shape[0],
            self.continuous_latent_size,
            dtype=cond.dtype,
            device=cond.device,
        )
        dt = 1.0 / steps
        for step_idx in range(steps):
            t = torch.full(
                (flat_cond.shape[0],),
                step_idx / steps,
                dtype=cond.dtype,
                device=cond.device,
            )
            x = x + dt * self.flow_head(x, t, flat_cond)
        return x.reshape(*cond_shape, self.continuous_latent_size)

    def _build_generation_prefix(
        self,
        *,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        continuous_prompt: torch.FloatTensor,
        text_target: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor]:
        """Build one unpadded prefix ending at the target-speech prediction anchor."""
        prompt_speech_len = int(discrete_prompt.shape[0])
        total = int(text_prompt.shape[0]) + 1 + prompt_speech_len + 1 + int(text_target.shape[0]) + 1
        device = text_prompt.device
        token_ids = torch.full((1, total), self.pad_token_id, dtype=torch.long, device=device)
        discrete_values = torch.full(
            (1, total, self.num_discrete_tokens),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        continuous_values = torch.zeros(
            (1, total, self.continuous_latent_size),
            dtype=continuous_prompt.dtype,
            device=device,
        )
        token_type_ids = torch.full((1, total), MU.TEXT_TOKEN_TYPE, dtype=torch.long, device=device)

        cursor = 0
        prompt_text_len = int(text_prompt.shape[0])
        token_ids[0, cursor : cursor + prompt_text_len] = text_prompt
        cursor += prompt_text_len
        token_ids[0, cursor] = self.eot_token_id
        cursor += 1
        if prompt_speech_len:
            token_type_ids[0, cursor : cursor + prompt_speech_len] = MU.SPEECH_FRAME_TOKEN_TYPE
            discrete_values[0, cursor : cursor + prompt_speech_len] = discrete_prompt
            continuous_values[0, cursor : cursor + prompt_speech_len] = continuous_prompt
            cursor += prompt_speech_len
        token_ids[0, cursor] = self.eos_token_id
        cursor += 1
        target_text_len = int(text_target.shape[0])
        token_ids[0, cursor : cursor + target_text_len] = text_target
        cursor += target_text_len
        token_ids[0, cursor] = self.eot_token_id
        return token_ids, discrete_values, continuous_values, token_type_ids

    def _is_terminal_block(self, discrete_ids: torch.LongTensor, eos_id: int) -> bool:
        return bool(torch.eq(discrete_ids, int(eos_id)).all().item())

    @torch.no_grad()
    def _generate_one(
        self,
        *,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        continuous_prompt: torch.FloatTensor,
        text_target: torch.LongTensor,
        max_new_blocks: int,
        discrete_eos_id: int,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
        flow_num_steps: Optional[int],
        force_silent_special_tokens: bool,
    ) -> tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor, list[torch.Tensor]]:
        token_ids, discrete_values, continuous_values, token_type_ids = self._build_generation_prefix(
            text_prompt=text_prompt,
            discrete_prompt=discrete_prompt,
            continuous_prompt=continuous_prompt,
            text_target=text_target,
        )
        prefix_len = int(token_ids.shape[1])
        prefix_attention = MU.build_causal_attention_mask(
            torch.ones((1, prefix_len), dtype=torch.bool, device=token_ids.device),
            dtype=continuous_values.dtype,
        )
        outputs = self._run_backbone(
            token_ids=token_ids,
            discrete_values=discrete_values,
            continuous_values=continuous_values,
            token_type_ids=token_type_ids,
            attention_mask=prefix_attention,
            use_cache=True,
        )
        hidden = outputs.last_hidden_state[:, -1, :]
        cache = outputs.past_key_values

        generated_discrete: list[torch.Tensor] = []
        generated_continuous: list[torch.Tensor] = []
        generated_prior: list[torch.Tensor] = []
        generated_logits: list[torch.Tensor] = []
        special_ids = MU.infer_special_discrete_token_ids(
            discrete_eos_id,
            backbone_eos_token_id=self.backbone.config.eos_token_id,
            backbone_pad_token_id=self.backbone.config.pad_token_id,
            discrete_vocab_size=self.discrete_vocab_size,
        )

        for step_idx in range(max_new_blocks):
            logits = self._discrete_logits(hidden)[0]
            sampled_discrete = self._sample_discrete_ids(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            prior = self.continuous_prior_head(hidden)[0]
            is_terminal = self._is_terminal_block(sampled_discrete, discrete_eos_id)
            is_special = bool(
                MU.build_special_block_mask(
                    sampled_discrete.unsqueeze(0),
                    special_ids,
                )[0].item()
            )
            if is_terminal or (force_silent_special_tokens and is_special):
                sampled_continuous_model = prior.new_zeros((self.continuous_latent_size,))
                sampled_continuous_out = sampled_continuous_model
            else:
                sampled_continuous_model = self.sample_continuous_latent(
                    prior.unsqueeze(0),
                    num_steps=flow_num_steps,
                )[0]
                sampled_continuous_out = self._denormalize_continuous_latent_values(
                    sampled_continuous_model,
                )
            prior_out = self._denormalize_continuous_latent_values(prior)

            generated_discrete.append(sampled_discrete)
            generated_continuous.append(sampled_continuous_out)
            generated_prior.append(prior_out)
            generated_logits.append(logits)
            if is_terminal:
                break

            next_token_ids = torch.full(
                (1, 1), self.pad_token_id, dtype=torch.long, device=hidden.device
            )
            next_discrete = sampled_discrete.view(1, 1, self.num_discrete_tokens)
            next_continuous = sampled_continuous_model.view(1, 1, self.continuous_latent_size)
            next_types = torch.full(
                (1, 1), MU.SPEECH_FRAME_TOKEN_TYPE, dtype=torch.long, device=hidden.device
            )
            outputs = self._run_backbone(
                token_ids=next_token_ids,
                discrete_values=next_discrete,
                continuous_values=next_continuous,
                token_type_ids=next_types,
                attention_mask=None,
                past_key_values=cache,
                use_cache=True,
                position_offset=prefix_len + step_idx,
            )
            hidden = outputs.last_hidden_state[:, -1, :]
            cache = outputs.past_key_values

        if not generated_discrete:
            empty_discrete = torch.empty(
                (0, self.num_discrete_tokens), dtype=torch.long, device=token_ids.device
            )
            empty_continuous = continuous_values.new_empty((0, self.continuous_latent_size))
            return empty_discrete, empty_continuous, empty_continuous, generated_logits
        return (
            torch.stack(generated_discrete, dim=0),
            torch.stack(generated_continuous, dim=0),
            torch.stack(generated_prior, dim=0),
            generated_logits,
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
        max_new_blocks: Optional[int] = None,
        discrete_eos_token_id: Optional[int] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        flow_num_steps: Optional[int] = None,
        force_silent_special_tokens: bool = False,
        return_dict: bool = True,
        generation_method: str = "ar",
    ) -> MU.PrismTTSGenerationOutput | tuple[torch.LongTensor, torch.FloatTensor]:
        """Autoregressively sample speech frames until EOS or ``max_new_blocks``."""
        method = str(generation_method).strip().lower()
        if method not in ("ar", "causal"):
            raise ValueError("generation_method must be one of {'ar', 'causal'} for the AR model.")
        if max_new_blocks is None:
            if speech_target_lengths is not None:
                max_new_blocks = int(torch.as_tensor(speech_target_lengths).max().item())
            else:
                max_new_blocks = 128
        if int(max_new_blocks) < 0:
            raise ValueError("max_new_blocks must be >= 0.")

        text_prompt = MU.normalize_text_tokens(text_prompt, "text_prompt")
        discrete_prompt = MU.normalize_discrete_tokens(
            discrete_prompt,
            "discrete_prompt",
            num_discrete_tokens=self.num_discrete_tokens,
        )
        continuous_prompt = MU.normalize_continuous_latents(
            continuous_prompt,
            expected_len=int(discrete_prompt.shape[1]),
            name="continuous_prompt",
            continuous_latent_size=self.continuous_latent_size,
        )
        prompt_payload_mask = self._non_special_discrete_block_mask(discrete_prompt)
        continuous_prompt = self._normalize_continuous_latent_values(
            continuous_prompt,
            payload_mask=prompt_payload_mask,
        )
        batch_size = int(text_prompt.shape[0])
        if text_target is None:
            text_target = text_prompt.new_empty((batch_size, 0))
        else:
            text_target = MU.normalize_text_tokens(text_target, "text_target")
        if text_target.shape[0] != batch_size:
            raise ValueError("text_target batch size must match text_prompt.")

        text_prompt_lengths = MU.normalize_lengths(
            text_prompt_lengths,
            batch_size,
            int(text_prompt.shape[1]),
            "text_prompt_lengths",
            text_prompt.device,
            default_value=int(text_prompt.shape[1]),
        )
        speech_prompt_lengths = MU.normalize_lengths(
            speech_prompt_lengths,
            batch_size,
            int(discrete_prompt.shape[1]),
            "speech_prompt_lengths",
            text_prompt.device,
            default_value=int(discrete_prompt.shape[1]),
        )
        text_target_lengths = MU.normalize_lengths(
            text_target_lengths,
            batch_size,
            int(text_target.shape[1]),
            "text_target_lengths",
            text_prompt.device,
            default_value=int(text_target.shape[1]),
        )
        discrete_eos_id = MU.resolve_generation_discrete_eos_token_id(
            discrete_eos_token_id,
            backbone_eos_token_id=self.backbone.config.eos_token_id,
            discrete_vocab_size=self.discrete_vocab_size,
        )

        per_sample: list[tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor, list[torch.Tensor]]] = []
        for sample_idx in range(batch_size):
            prompt_text_len = int(text_prompt_lengths[sample_idx].item())
            prompt_speech_len = int(speech_prompt_lengths[sample_idx].item())
            target_text_len = int(text_target_lengths[sample_idx].item())
            per_sample.append(
                self._generate_one(
                    text_prompt=text_prompt[sample_idx, :prompt_text_len],
                    discrete_prompt=discrete_prompt[sample_idx, :prompt_speech_len, :],
                    continuous_prompt=continuous_prompt[sample_idx, :prompt_speech_len, :],
                    text_target=text_target[sample_idx, :target_text_len],
                    max_new_blocks=int(max_new_blocks),
                    discrete_eos_id=discrete_eos_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    flow_num_steps=flow_num_steps,
                    force_silent_special_tokens=force_silent_special_tokens,
                )
            )

        max_generated = max((int(item[0].shape[0]) for item in per_sample), default=0)
        generated_discrete = torch.full(
            (batch_size, max_generated, self.num_discrete_tokens),
            self.pad_token_id,
            dtype=torch.long,
            device=text_prompt.device,
        )
        generated_continuous = continuous_prompt.new_zeros(
            (batch_size, max_generated, self.continuous_latent_size)
        )
        generated_prior = continuous_prompt.new_zeros(
            (batch_size, max_generated, self.continuous_latent_size)
        )
        logits_by_step: list[torch.Tensor] = []
        for sample_idx, (sample_discrete, sample_continuous, sample_prior, sample_logits) in enumerate(per_sample):
            length = int(sample_discrete.shape[0])
            if length:
                generated_discrete[sample_idx, :length] = sample_discrete
                generated_continuous[sample_idx, :length] = sample_continuous
                generated_prior[sample_idx, :length] = sample_prior
            for step_idx, logits in enumerate(sample_logits):
                while len(logits_by_step) <= step_idx:
                    logits_by_step.append(
                        torch.full(
                            (batch_size, self.num_discrete_tokens, self.discrete_vocab_size),
                            float("-inf"),
                            dtype=logits.dtype,
                            device=logits.device,
                        )
                    )
                logits_by_step[step_idx][sample_idx] = logits

        max_text_target = int(text_target_lengths.max().item()) if batch_size else 0
        text_ids_out = text_target[:, :max_text_target]
        if not return_dict:
            return generated_discrete.transpose(1, 2).contiguous(), generated_continuous
        return MU.PrismTTSGenerationOutput(
            text_ids=text_ids_out,
            discrete_ids=generated_discrete.transpose(1, 2).contiguous(),
            continuous_latents=generated_continuous,
            prior_latents=generated_prior,
            discrete_logits=tuple(logits_by_step),
        )

    @torch.no_grad()
    def generate_e2e(
        self,
        raw_text_prompt: str | Sequence[str],
        raw_speech_prompt: Any | list[Any],
        raw_text_target: str | Sequence[str],
        *,
        text_tokenizer: Optional[Callable[[str], Sequence[int]]] = None,
        speech_encoder: Optional[Callable[[Any], tuple[torch.Tensor, torch.Tensor]]] = None,
        speech_decoder: Optional[Callable[[torch.Tensor], Any]] = None,
        output_type: str = "tensor",
        return_dict: bool = True,
        mimi_model_name_or_path: str = "kyutai/mimi",
        mimi_revision: str = "main",
        mimi_token: str | bool | None = None,
        mimi_local_files_only: bool = False,
        **generate_kwargs: Any,
    ) -> MU.PrismTTSGenerationOutput | tuple[torch.LongTensor, torch.FloatTensor] | torch.Tensor:
        """Generate from raw text and speech prompts using optional Mimi adapters."""
        output_type = str(output_type).strip().lower()
        if output_type not in ("tensor", "speech"):
            raise ValueError("output_type must be one of {'tensor', 'speech'}.")
        prompt_text_list = MU.normalize_raw_text_batch(raw_text_prompt, "raw_text_prompt")
        target_text_list = MU.normalize_raw_text_batch(raw_text_target, "raw_text_target")
        speech_prompt_list = list(raw_speech_prompt) if isinstance(raw_speech_prompt, list) else [raw_speech_prompt]
        batch_size = len(prompt_text_list)
        if len(target_text_list) != batch_size or len(speech_prompt_list) != batch_size:
            raise ValueError("Raw prompt and target inputs must have matching batch sizes.")

        device = next(self.parameters()).device
        continuous_dtype = self.continuous_proj.weight.dtype
        if text_tokenizer is None:
            from dataset.dataset import SharedVocabTokenizer, build_shared_token_layout

            discrete_count = self.discrete_vocab_size - 3
            _, eos_id, _, text_offset = build_shared_token_layout(discrete_count)
            text_tokenizer = SharedVocabTokenizer(
                vocab_path=Path(__file__).resolve().parents[1] / "dataset" / "vocab.txt",
                text_token_offset=text_offset,
                eos_token_id=eos_id,
                append_eos=False,
            )
        if speech_encoder is None:
            speech_encoder = MU.build_default_mimi_speech_encoder(
                num_discrete_tokens=self.num_discrete_tokens,
                device=device,
                continuous_dtype=continuous_dtype,
                mimi_model_name_or_path=mimi_model_name_or_path,
                mimi_revision=mimi_revision,
                mimi_token=mimi_token,
                mimi_local_files_only=mimi_local_files_only,
            )
        if speech_decoder is None:
            speech_decoder = MU.build_lazy_mimi_speech_decoder(
                device=device,
                continuous_dtype=continuous_dtype,
                mimi_model_name_or_path=mimi_model_name_or_path,
                mimi_revision=mimi_revision,
                mimi_token=mimi_token,
                mimi_local_files_only=mimi_local_files_only,
            )

        prompt_text_tokens = [[int(token) for token in text_tokenizer(value)] for value in prompt_text_list]
        target_text_tokens = [[int(token) for token in text_tokenizer(value)] for value in target_text_list]
        encoded_prompts: list[tuple[torch.LongTensor, torch.FloatTensor]] = []
        for prompt in speech_prompt_list:
            discrete, continuous = speech_encoder(prompt)
            discrete = torch.as_tensor(discrete, dtype=torch.long, device=device)
            continuous = torch.as_tensor(continuous, dtype=continuous_dtype, device=device)
            if discrete.dim() == 3 and discrete.shape[0] == 1:
                discrete = discrete[0]
            if continuous.dim() == 3 and continuous.shape[0] == 1:
                continuous = continuous[0]
            if discrete.dim() != 2 or continuous.dim() != 2:
                raise ValueError("speech_encoder outputs must be [frames, streams] and [frames, channels].")
            if discrete.shape[-1] != self.num_discrete_tokens:
                if discrete.shape[0] == self.num_discrete_tokens:
                    discrete = discrete.transpose(0, 1).contiguous()
                else:
                    raise ValueError("Encoded discrete prompt stream count does not match the model.")
            if continuous.shape[-1] != self.continuous_latent_size:
                if continuous.shape[0] == self.continuous_latent_size:
                    continuous = continuous.transpose(0, 1).contiguous()
                else:
                    raise ValueError("Encoded continuous prompt channel count does not match the model.")
            if discrete.shape[0] != continuous.shape[0]:
                raise ValueError("Encoded discrete and continuous prompt lengths must match.")
            encoded_prompts.append((discrete, continuous))

        max_prompt_text = max(map(len, prompt_text_tokens), default=0)
        max_target_text = max(map(len, target_text_tokens), default=0)
        max_prompt_speech = max((int(discrete.shape[0]) for discrete, _ in encoded_prompts), default=0)
        text_prompt = torch.full((batch_size, max_prompt_text), self.pad_token_id, dtype=torch.long, device=device)
        text_target = torch.full((batch_size, max_target_text), self.pad_token_id, dtype=torch.long, device=device)
        discrete_prompt = torch.full(
            (batch_size, self.num_discrete_tokens, max_prompt_speech),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        continuous_prompt = torch.zeros(
            (batch_size, max_prompt_speech, self.continuous_latent_size),
            dtype=continuous_dtype,
            device=device,
        )
        text_prompt_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        text_target_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        speech_prompt_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        for idx, (discrete, continuous) in enumerate(encoded_prompts):
            prompt_len = len(prompt_text_tokens[idx])
            target_len = len(target_text_tokens[idx])
            speech_len = int(discrete.shape[0])
            text_prompt_lengths[idx] = prompt_len
            text_target_lengths[idx] = target_len
            speech_prompt_lengths[idx] = speech_len
            if prompt_len:
                text_prompt[idx, :prompt_len] = torch.tensor(prompt_text_tokens[idx], dtype=torch.long, device=device)
            if target_len:
                text_target[idx, :target_len] = torch.tensor(target_text_tokens[idx], dtype=torch.long, device=device)
            if speech_len:
                discrete_prompt[idx, :, :speech_len] = discrete.transpose(0, 1)
                continuous_prompt[idx, :speech_len] = continuous
        for name in ("text_prompt_lengths", "speech_prompt_lengths", "text_target_lengths", "return_dict"):
            if name in generate_kwargs:
                raise ValueError(f"Do not pass {name} when using generate_e2e.")
        generation = self.generate(
            text_prompt=text_prompt,
            discrete_prompt=discrete_prompt,
            continuous_prompt=continuous_prompt,
            text_target=text_target,
            text_prompt_lengths=text_prompt_lengths,
            speech_prompt_lengths=speech_prompt_lengths,
            text_target_lengths=text_target_lengths,
            return_dict=True,
            **generate_kwargs,
        )
        if output_type == "tensor":
            if return_dict:
                return generation
            if generation.discrete_ids is None or generation.continuous_latents is None:
                raise RuntimeError("Generation did not produce tensors.")
            return generation.discrete_ids, generation.continuous_latents
        if generation.continuous_latents is None:
            raise RuntimeError("Generation did not produce continuous latents.")
        decoded = speech_decoder(generation.continuous_latents)
        speech = decoded if torch.is_tensor(decoded) else torch.as_tensor(decoded)
        if speech.dim() == 3 and speech.shape[1] == 1:
            speech = speech[:, 0]
        return speech.unsqueeze(0) if speech.dim() == 1 else speech
