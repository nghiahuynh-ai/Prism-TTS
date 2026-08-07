from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig

from models.flow_head import TimestepEmbedder
from models.llama_backbone import LlamaBackbone
from utils import model_utils as MU


class PrismContinuousMeanFlowTTS(nn.Module):
    """Causal continuous-latent generator used by Prism-TTS stage 2.

    A target patch is represented by its discrete codec tokens, independently
    noised continuous latents, and one noise level per frame.  Causal attention
    makes velocity at frame ``i`` depend only on modalities at positions
    ``<= i``.  Clean history is supplied at ``t=1``; the current patch is the
    only portion denoised by the flow objective.
    """

    stage = "continuous_meanflow"

    def __init__(
        self,
        *,
        llama_config: LlamaConfig,
        num_discrete_tokens: int,
        discrete_vocab_size: int,
        continuous_latent_size: int,
        min_train_window: int = 1,
        max_train_window: int = 16,
        time_epsilon: float = 1e-4,
        normalize_continuous_latents: bool | int | str = False,
        continuous_latent_mean: float | Sequence[float] | torch.Tensor = 0.0,
        continuous_latent_std: float | Sequence[float] | torch.Tensor = 1.0,
        continuous_latent_std_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_discrete_tokens < 1:
            raise ValueError("num_discrete_tokens must be >= 1.")
        if discrete_vocab_size < 1 or discrete_vocab_size > llama_config.vocab_size:
            raise ValueError("discrete_vocab_size must be in [1, llama_config.vocab_size].")
        if continuous_latent_size < 1:
            raise ValueError("continuous_latent_size must be >= 1.")
        if min_train_window < 1 or max_train_window < min_train_window:
            raise ValueError("Require 1 <= min_train_window <= max_train_window.")
        if not 0.0 <= time_epsilon < 0.5:
            raise ValueError("time_epsilon must be in [0, 0.5).")
        if continuous_latent_std_eps <= 0:
            raise ValueError("continuous_latent_std_eps must be > 0.")

        self.backbone = LlamaBackbone(llama_config)
        self.hidden_size = int(llama_config.hidden_size)
        self.num_discrete_tokens = int(num_discrete_tokens)
        self.discrete_vocab_size = int(discrete_vocab_size)
        self.continuous_latent_size = int(continuous_latent_size)
        self.min_train_window = int(min_train_window)
        self.max_train_window = int(max_train_window)
        self.time_epsilon = float(time_epsilon)
        self.normalize_continuous_latents = self._coerce_bool(
            normalize_continuous_latents, name="normalize_continuous_latents"
        )
        self.continuous_latent_std_eps = float(continuous_latent_std_eps)

        self.discrete_frame_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.continuous_proj = nn.Linear(self.continuous_latent_size, self.hidden_size)
        self.time_embed = TimestepEmbedder(self.hidden_size)
        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.velocity_head = nn.Linear(self.hidden_size, self.continuous_latent_size)
        self.discrete_stream_embeddings = nn.Parameter(
            torch.empty(self.num_discrete_tokens, self.hidden_size)
        )

        vocab_size = int(self.backbone.config.vocab_size)
        self.pad_token_id = int(self.backbone.config.pad_token_id or 0)
        if not 0 <= self.pad_token_id < vocab_size:
            self.pad_token_id = 0
        self.eos_token_id = int(self.backbone.config.eos_token_id or 0)
        if not 0 <= self.eos_token_id < vocab_size:
            self.eos_token_id = 0
        self.eot_token_id = max(0, self.eos_token_id - 1)
        self.special_discrete_token_ids = MU.infer_special_discrete_token_ids(
            self.eos_token_id,
            backbone_eos_token_id=self.backbone.config.eos_token_id,
            backbone_pad_token_id=self.backbone.config.pad_token_id,
            discrete_vocab_size=self.discrete_vocab_size,
        )

        mean = self._coerce_stat(
            continuous_latent_mean, name="continuous_latent_mean", size=self.continuous_latent_size
        )
        std = self._coerce_stat(
            continuous_latent_std, name="continuous_latent_std", size=self.continuous_latent_size
        )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Continuous normalization statistics must be finite.")
        if torch.le(std, self.continuous_latent_std_eps).any():
            raise ValueError("continuous_latent_std must be above continuous_latent_std_eps.")
        self.register_buffer("continuous_latent_mean", mean, persistent=False)
        self.register_buffer("continuous_latent_std", std, persistent=False)
        self.reset_parameters()

    @staticmethod
    def _coerce_bool(value: bool | int | str, *, name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            parsed = value.strip().lower()
            if parsed in {"1", "true", "yes", "y", "on"}:
                return True
            if parsed in {"0", "false", "no", "n", "off"}:
                return False
        raise ValueError(f"{name} must be a boolean value.")

    @staticmethod
    def _coerce_stat(
        value: float | Sequence[float] | torch.Tensor, *, name: str, size: int
    ) -> torch.FloatTensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
        if tensor.dim() == 0 or tensor.numel() == 1:
            return tensor.reshape(1).repeat(size).contiguous()
        if tensor.dim() == 1 and tensor.numel() == size:
            return tensor.contiguous()
        raise ValueError(f"{name} must be a scalar or a {size}-value vector.")

    def reset_parameters(self) -> None:
        std = float(getattr(self.backbone.config, "initializer_range", 0.02))
        for layer in (self.discrete_frame_proj, self.continuous_proj, self.velocity_head):
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

    def _stat_view(self, stat: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return stat.to(device=values.device, dtype=values.dtype).view(
            (1,) * (values.dim() - 1) + (self.continuous_latent_size,)
        )

    def _normalize(self, values: torch.FloatTensor) -> torch.FloatTensor:
        if not self.normalize_continuous_latents:
            return values
        return (values - self._stat_view(self.continuous_latent_mean, values)) / self._stat_view(
            self.continuous_latent_std, values
        )

    def _denormalize(self, values: torch.FloatTensor) -> torch.FloatTensor:
        if not self.normalize_continuous_latents:
            return values
        return values * self._stat_view(self.continuous_latent_std, values) + self._stat_view(
            self.continuous_latent_mean, values
        )

    def _is_special_block(self, discrete: torch.LongTensor) -> torch.BoolTensor:
        if not self.special_discrete_token_ids:
            return torch.zeros(discrete.shape[:-1], dtype=torch.bool, device=discrete.device)
        ids = torch.tensor(self.special_discrete_token_ids, dtype=discrete.dtype, device=discrete.device)
        return torch.isin(discrete, ids).any(dim=-1)

    def _build_inputs_embeds(
        self,
        *,
        token_ids: torch.LongTensor,
        discrete_values: torch.LongTensor,
        continuous_values: torch.FloatTensor,
        timesteps: torch.FloatTensor,
        token_type_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        is_frame = token_type_ids == MU.SPEECH_FRAME_TOKEN_TYPE
        text_embeds = self.text_embedding(token_ids)
        safe_discrete = discrete_values.clamp(min=0, max=self.backbone.config.vocab_size - 1)
        per_stream = self.discrete_embedding(safe_discrete)
        per_stream = per_stream + self.discrete_stream_embeddings.view(
            1, 1, self.num_discrete_tokens, self.hidden_size
        )
        discrete_frame = self.discrete_frame_proj(
            per_stream.sum(dim=2) / math.sqrt(float(self.num_discrete_tokens))
        )
        continuous_frame = self.continuous_proj(continuous_values)
        time_frame = self.time_embed(timesteps.reshape(-1)).reshape(
            *timesteps.shape, self.hidden_size
        ).to(dtype=continuous_frame.dtype)
        frame = self.input_norm(discrete_frame + continuous_frame + time_frame)
        return torch.where(is_frame.unsqueeze(-1), frame, text_embeds)

    def _velocity_for_sequence(
        self,
        *,
        token_ids: torch.LongTensor,
        discrete_values: torch.LongTensor,
        continuous_values: torch.FloatTensor,
        timesteps: torch.FloatTensor,
        token_type_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        inputs_embeds = self._build_inputs_embeds(
            token_ids=token_ids,
            discrete_values=discrete_values,
            continuous_values=continuous_values,
            timesteps=timesteps,
            token_type_ids=token_type_ids,
        )
        batch_size, seq_len, _ = inputs_embeds.shape
        causal_mask = MU.build_causal_attention_mask(attention_mask, dtype=inputs_embeds.dtype)
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.backbone.rotary_emb(inputs_embeds, position_ids=position_ids)
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            return_dict=True,
        )
        return self.velocity_head(outputs.last_hidden_state)

    def _sample_window(self, length: int) -> tuple[int, int]:
        window_cap = min(self.max_train_window, length)
        window_min = min(self.min_train_window, window_cap)
        if self.training:
            width = int(torch.randint(window_min, window_cap + 1, (1,)).item())
            start = int(torch.randint(0, length - width + 1, (1,)).item())
            return start, width
        return 0, window_cap

    def _build_training_sequence(
        self,
        *,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        continuous_prompt: torch.FloatTensor,
        text_target: torch.LongTensor,
        discrete_target: torch.LongTensor,
        continuous_target: torch.FloatTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.BoolTensor, torch.FloatTensor, torch.BoolTensor]:
        """Build one causal sequence with a clean history and an active noisy patch."""
        valid_targets = ~self._is_special_block(discrete_target)
        real_length = int(valid_targets.to(dtype=torch.long).cumprod(dim=0).sum().item())
        if real_length < 1:
            raise ValueError("A continuous MeanFlow sample requires at least one non-special target frame.")
        start, width = self._sample_window(real_length)
        target = self._normalize(continuous_target[:real_length])
        noise = torch.randn_like(target[start : start + width])
        timesteps_patch = torch.empty(width, dtype=target.dtype, device=target.device).uniform_(
            self.time_epsilon, 1.0 - self.time_epsilon
        )
        noisy_patch = (1.0 - timesteps_patch.unsqueeze(-1)) * noise + timesteps_patch.unsqueeze(-1) * target[
            start : start + width
        ]
        velocity_target = target[start : start + width] - noise

        token_ids: list[int] = []
        discrete_values: list[torch.Tensor] = []
        continuous_values: list[torch.Tensor] = []
        timesteps: list[torch.Tensor] = []
        token_types: list[int] = []
        velocity_values: list[torch.Tensor] = []
        velocity_mask: list[bool] = []
        zero_cont = target.new_zeros((self.continuous_latent_size,))
        pad_discrete = torch.full(
            (self.num_discrete_tokens,), self.pad_token_id, dtype=torch.long, device=target.device
        )

        def add_text(token: int) -> None:
            token_ids.append(int(token))
            discrete_values.append(pad_discrete)
            continuous_values.append(zero_cont)
            timesteps.append(target.new_zeros(()))
            token_types.append(MU.TEXT_TOKEN_TYPE)
            velocity_values.append(zero_cont)
            velocity_mask.append(False)

        def add_frame(discrete: torch.Tensor, continuous: torch.Tensor, timestep: torch.Tensor, *, train: bool, velocity: torch.Tensor | None = None) -> None:
            token_ids.append(self.pad_token_id)
            discrete_values.append(discrete.to(dtype=torch.long))
            continuous_values.append(continuous)
            timesteps.append(timestep)
            token_types.append(MU.SPEECH_FRAME_TOKEN_TYPE)
            velocity_values.append(zero_cont if velocity is None else velocity)
            velocity_mask.append(train)

        for token in text_prompt.tolist():
            add_text(token)
        add_text(self.eot_token_id)
        normalized_prompt = self._normalize(continuous_prompt)
        for d, x in zip(discrete_prompt, normalized_prompt):
            add_frame(d, x, target.new_ones(()), train=False)
        add_text(self.eos_token_id)
        for token in text_target.tolist():
            add_text(token)
        add_text(self.eot_token_id)
        for index in range(start):
            add_frame(discrete_target[index], target[index], target.new_ones(()), train=False)
        for local_index in range(width):
            add_frame(
                discrete_target[start + local_index],
                noisy_patch[local_index],
                timesteps_patch[local_index],
                train=True,
                velocity=velocity_target[local_index],
            )
        return (
            torch.tensor(token_ids, dtype=torch.long, device=target.device),
            torch.stack(discrete_values),
            torch.stack(continuous_values),
            torch.stack(timesteps),
            torch.tensor(token_types, dtype=torch.long, device=target.device),
            torch.ones(len(token_ids), dtype=torch.bool, device=target.device),
            torch.stack(velocity_values),
            torch.tensor(velocity_mask, dtype=torch.bool, device=target.device),
        )

    def forward(
        self,
        *,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        continuous_prompt: torch.FloatTensor,
        text_target: torch.LongTensor,
        discrete_target: torch.LongTensor,
        continuous_target: torch.FloatTensor,
        text_prompt_lengths: Optional[torch.Tensor] = None,
        speech_prompt_lengths: Optional[torch.Tensor] = None,
        text_target_lengths: Optional[torch.Tensor] = None,
        speech_target_lengths: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> MU.PrismTTSOutput | tuple[torch.Tensor, torch.Tensor]:
        text_prompt = MU.normalize_text_tokens(text_prompt, "text_prompt")
        discrete_prompt = MU.normalize_discrete_tokens(
            discrete_prompt, "discrete_prompt", num_discrete_tokens=self.num_discrete_tokens
        )
        discrete_target = MU.normalize_discrete_tokens(
            discrete_target, "discrete_target", num_discrete_tokens=self.num_discrete_tokens
        )
        continuous_prompt = MU.normalize_continuous_latents(
            continuous_prompt, discrete_prompt.shape[1], "continuous_prompt",
            continuous_latent_size=self.continuous_latent_size,
        )
        continuous_target = MU.normalize_continuous_latents(
            continuous_target, discrete_target.shape[1], "continuous_target",
            continuous_latent_size=self.continuous_latent_size,
        )
        text_target = MU.normalize_text_tokens(text_target, "text_target")
        batch_size = int(text_prompt.shape[0])
        if not all(tensor.shape[0] == batch_size for tensor in (discrete_prompt, continuous_prompt, text_target, discrete_target, continuous_target)):
            raise ValueError("All continuous-stage inputs must share a batch size.")
        text_prompt_lengths = MU.normalize_lengths(
            text_prompt_lengths, batch_size, text_prompt.shape[1], "text_prompt_lengths", text_prompt.device,
            default_value=text_prompt.shape[1],
        )
        speech_prompt_lengths = MU.normalize_lengths(
            speech_prompt_lengths, batch_size, discrete_prompt.shape[1], "speech_prompt_lengths", text_prompt.device,
            default_value=discrete_prompt.shape[1],
        )
        text_target_lengths = MU.normalize_lengths(
            text_target_lengths, batch_size, text_target.shape[1], "text_target_lengths", text_prompt.device,
            default_value=text_target.shape[1],
        )
        speech_target_lengths = MU.normalize_lengths(
            speech_target_lengths, batch_size, discrete_target.shape[1], "speech_target_lengths", text_prompt.device,
            default_value=discrete_target.shape[1],
        )

        sequences = []
        for index in range(batch_size):
            sequences.append(self._build_training_sequence(
                text_prompt=text_prompt[index, : int(text_prompt_lengths[index])],
                discrete_prompt=discrete_prompt[index, : int(speech_prompt_lengths[index])],
                continuous_prompt=continuous_prompt[index, : int(speech_prompt_lengths[index])],
                text_target=text_target[index, : int(text_target_lengths[index])],
                discrete_target=discrete_target[index, : int(speech_target_lengths[index])],
                continuous_target=continuous_target[index, : int(speech_target_lengths[index])],
            ))
        max_length = max(item[0].shape[0] for item in sequences)
        device = text_prompt.device
        dtype = continuous_target.dtype
        token_ids = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long, device=device)
        discrete_values = torch.full(
            (batch_size, max_length, self.num_discrete_tokens), self.pad_token_id, dtype=torch.long, device=device
        )
        continuous_values = torch.zeros(
            (batch_size, max_length, self.continuous_latent_size), dtype=dtype, device=device
        )
        timesteps = torch.zeros((batch_size, max_length), dtype=dtype, device=device)
        token_types = torch.full((batch_size, max_length), MU.TEXT_TOKEN_TYPE, dtype=torch.long, device=device)
        attention = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
        velocity_targets = torch.zeros_like(continuous_values)
        velocity_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
        for index, item in enumerate(sequences):
            length = item[0].shape[0]
            token_ids[index, :length] = item[0]
            discrete_values[index, :length] = item[1]
            continuous_values[index, :length] = item[2]
            timesteps[index, :length] = item[3]
            token_types[index, :length] = item[4]
            attention[index, :length] = item[5]
            velocity_targets[index, :length] = item[6]
            velocity_mask[index, :length] = item[7]
        velocity = self._velocity_for_sequence(
            token_ids=token_ids,
            discrete_values=discrete_values,
            continuous_values=continuous_values,
            timesteps=timesteps,
            token_type_ids=token_types,
            attention_mask=attention,
        )
        loss = F.mse_loss(velocity[velocity_mask], velocity_targets[velocity_mask])
        if not return_dict:
            return loss, loss
        zero = loss.new_zeros(())
        return MU.PrismTTSOutput(loss=loss, discrete_loss=zero, continuous_loss=loss, flow_loss=loss)

    def _resolve_generation_length(self, discrete: torch.LongTensor, requested: int | None) -> int:
        maximum = int(discrete.shape[0]) if requested is None else min(int(requested), int(discrete.shape[0]))
        for index in range(maximum):
            block = discrete[index]
            if bool(torch.eq(block, self.eos_token_id).all().item()) or bool(torch.eq(block, self.pad_token_id).all().item()):
                return index
        return maximum

    @staticmethod
    def _resolve_windows(length: int, window_size: int | Sequence[int]) -> list[int]:
        if isinstance(window_size, int):
            if window_size < 1:
                raise ValueError("window_size must be >= 1.")
            return [min(window_size, length - offset) for offset in range(0, length, window_size)]
        windows = [int(value) for value in window_size]
        if any(value < 1 for value in windows):
            raise ValueError("All user-selected window sizes must be >= 1.")
        result: list[int] = []
        remaining = length
        for width in windows:
            if remaining <= 0:
                break
            actual = min(width, remaining)
            result.append(actual)
            remaining -= actual
        if remaining:
            raise ValueError("The supplied window-size list does not cover all discrete target frames.")
        return result

    def _build_generation_sequence(
        self,
        *,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        continuous_prompt: torch.FloatTensor,
        text_target: torch.LongTensor,
        discrete_history: torch.LongTensor,
        continuous_history: torch.FloatTensor,
        discrete_patch: torch.LongTensor,
        continuous_patch: torch.FloatTensor,
        patch_timesteps: torch.FloatTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.BoolTensor]:
        values: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
        zero = continuous_prompt.new_zeros((self.continuous_latent_size,))
        pad = torch.full((self.num_discrete_tokens,), self.pad_token_id, dtype=torch.long, device=zero.device)
        for token in text_prompt.tolist():
            values.append((int(token), pad, zero, zero.new_zeros(()), MU.TEXT_TOKEN_TYPE))
        values.append((self.eot_token_id, pad, zero, zero.new_zeros(()), MU.TEXT_TOKEN_TYPE))
        for d, x in zip(discrete_prompt, continuous_prompt):
            values.append((self.pad_token_id, d, x, zero.new_ones(()), MU.SPEECH_FRAME_TOKEN_TYPE))
        values.append((self.eos_token_id, pad, zero, zero.new_zeros(()), MU.TEXT_TOKEN_TYPE))
        for token in text_target.tolist():
            values.append((int(token), pad, zero, zero.new_zeros(()), MU.TEXT_TOKEN_TYPE))
        values.append((self.eot_token_id, pad, zero, zero.new_zeros(()), MU.TEXT_TOKEN_TYPE))
        for d, x in zip(discrete_history, continuous_history):
            values.append((self.pad_token_id, d, x, zero.new_ones(()), MU.SPEECH_FRAME_TOKEN_TYPE))
        for d, x, t in zip(discrete_patch, continuous_patch, patch_timesteps):
            values.append((self.pad_token_id, d, x, t, MU.SPEECH_FRAME_TOKEN_TYPE))
        return (
            torch.tensor([item[0] for item in values], dtype=torch.long, device=zero.device).unsqueeze(0),
            torch.stack([item[1] for item in values]).unsqueeze(0),
            torch.stack([item[2] for item in values]).unsqueeze(0),
            torch.stack([item[3] for item in values]).unsqueeze(0),
            torch.tensor([item[4] for item in values], dtype=torch.long, device=zero.device).unsqueeze(0),
            torch.ones((1, len(values)), dtype=torch.bool, device=zero.device),
        )

    @torch.no_grad()
    def _generate_one(
        self,
        *,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        continuous_prompt: torch.FloatTensor,
        text_target: torch.LongTensor,
        discrete_target: torch.LongTensor,
        window_size: int | Sequence[int],
        flow_num_steps: int,
        time_stagger: float,
    ) -> torch.FloatTensor:
        if flow_num_steps < 1:
            raise ValueError("flow_num_steps must be >= 1.")
        if not 0.0 <= time_stagger < 1.0:
            raise ValueError("time_stagger must be in [0, 1).")
        length = self._resolve_generation_length(discrete_target, None)
        if length == 0:
            return continuous_prompt.new_empty((0, self.continuous_latent_size))
        windows = self._resolve_windows(length, window_size)
        prompt = self._normalize(continuous_prompt)
        history_d = discrete_target.new_empty((0, self.num_discrete_tokens))
        history_x = prompt.new_empty((0, self.continuous_latent_size))
        cursor = 0
        for width in windows:
            patch_d = discrete_target[cursor : cursor + width]
            patch_x = torch.randn(
                (width, self.continuous_latent_size), dtype=prompt.dtype, device=prompt.device
            )
            offsets = torch.linspace(-time_stagger, time_stagger, width, dtype=prompt.dtype, device=prompt.device)
            for step in range(flow_num_steps):
                s = float(step) / float(flow_num_steps)
                base = patch_x.new_full((width,), s)
                # t_i(s)=s+a_i*s*(1-s) gives t_i(0)=0, t_i(1)=1, and distinct,
                # monotonic per-frame time levels throughout the solver trajectory.
                times = base + offsets * s * (1.0 - s)
                dt_ds = 1.0 + offsets * (1.0 - 2.0 * s)
                sequence = self._build_generation_sequence(
                    text_prompt=text_prompt,
                    discrete_prompt=discrete_prompt,
                    continuous_prompt=prompt,
                    text_target=text_target,
                    discrete_history=history_d,
                    continuous_history=history_x,
                    discrete_patch=patch_d,
                    continuous_patch=patch_x,
                    patch_timesteps=times,
                )
                velocity = self._velocity_for_sequence(
                    token_ids=sequence[0],
                    discrete_values=sequence[1],
                    continuous_values=sequence[2],
                    timesteps=sequence[3],
                    token_type_ids=sequence[4],
                    attention_mask=sequence[5],
                )[0, -width:]
                patch_x = patch_x + velocity * dt_ds.unsqueeze(-1) / float(flow_num_steps)
            history_d = torch.cat((history_d, patch_d), dim=0)
            history_x = torch.cat((history_x, patch_x), dim=0)
            cursor += width
        return self._denormalize(history_x)

    @torch.no_grad()
    def generate(
        self,
        *,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        continuous_prompt: torch.FloatTensor,
        text_target: torch.LongTensor,
        discrete_target: torch.LongTensor,
        text_prompt_lengths: Optional[torch.Tensor | int] = None,
        speech_prompt_lengths: Optional[torch.Tensor | int] = None,
        text_target_lengths: Optional[torch.Tensor | int] = None,
        discrete_target_lengths: Optional[torch.Tensor | int] = None,
        window_size: int | Sequence[int] = 1,
        flow_num_steps: int = 64,
        time_stagger: float = 0.5,
        return_dict: bool = True,
    ) -> MU.PrismTTSGenerationOutput | torch.FloatTensor:
        text_prompt = MU.normalize_text_tokens(text_prompt, "text_prompt")
        text_target = MU.normalize_text_tokens(text_target, "text_target")
        discrete_prompt = MU.normalize_discrete_tokens(
            discrete_prompt, "discrete_prompt", num_discrete_tokens=self.num_discrete_tokens
        )
        discrete_target = MU.normalize_discrete_tokens(
            discrete_target, "discrete_target", num_discrete_tokens=self.num_discrete_tokens
        )
        continuous_prompt = MU.normalize_continuous_latents(
            continuous_prompt, discrete_prompt.shape[1], "continuous_prompt",
            continuous_latent_size=self.continuous_latent_size,
        )
        batch_size = int(text_prompt.shape[0])
        if not all(tensor.shape[0] == batch_size for tensor in (text_target, discrete_prompt, continuous_prompt, discrete_target)):
            raise ValueError("All continuous-stage generation inputs must share a batch size.")
        text_prompt_lengths = MU.normalize_lengths(
            text_prompt_lengths, batch_size, text_prompt.shape[1], "text_prompt_lengths", text_prompt.device,
            default_value=text_prompt.shape[1],
        )
        speech_prompt_lengths = MU.normalize_lengths(
            speech_prompt_lengths, batch_size, discrete_prompt.shape[1], "speech_prompt_lengths", text_prompt.device,
            default_value=discrete_prompt.shape[1],
        )
        text_target_lengths = MU.normalize_lengths(
            text_target_lengths, batch_size, text_target.shape[1], "text_target_lengths", text_prompt.device,
            default_value=text_target.shape[1],
        )
        discrete_target_lengths = MU.normalize_lengths(
            discrete_target_lengths, batch_size, discrete_target.shape[1], "discrete_target_lengths", text_prompt.device,
            default_value=discrete_target.shape[1],
        )
        generated: list[torch.FloatTensor] = []
        for index in range(batch_size):
            generated.append(self._generate_one(
                text_prompt=text_prompt[index, : int(text_prompt_lengths[index])],
                discrete_prompt=discrete_prompt[index, : int(speech_prompt_lengths[index])],
                continuous_prompt=continuous_prompt[index, : int(speech_prompt_lengths[index])],
                text_target=text_target[index, : int(text_target_lengths[index])],
                discrete_target=discrete_target[index, : int(discrete_target_lengths[index])],
                window_size=window_size,
                flow_num_steps=flow_num_steps,
                time_stagger=time_stagger,
            ))
        max_length = max((item.shape[0] for item in generated), default=0)
        output = continuous_prompt.new_zeros((batch_size, max_length, self.continuous_latent_size))
        for index, values in enumerate(generated):
            if values.numel():
                output[index, : values.shape[0]] = values
        if not return_dict:
            return output
        return MU.PrismTTSGenerationOutput(
            text_ids=text_target,
            discrete_ids=discrete_target.transpose(1, 2).contiguous(),
            continuous_latents=output,
        )
