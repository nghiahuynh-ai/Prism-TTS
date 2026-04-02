from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.utils import ModelOutput

from models.flow_head import FlowHead
from models.llama_backbone import LlamaBackbone


@dataclass
class PrismTTSOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    discrete_loss: Optional[torch.Tensor] = None
    flow_loss: Optional[torch.Tensor] = None
    text_loss: Optional[torch.Tensor] = None


@dataclass
class PrismTTSGenerationOutput(ModelOutput):
    text_ids: Optional[torch.LongTensor] = None
    discrete_ids: Optional[torch.LongTensor] = None
    continuous_latents: Optional[torch.FloatTensor] = None
    discrete_logits: Optional[tuple[torch.Tensor, ...]] = None


class PrismTTS(nn.Module):
    """
    Block structure per time step:
    [text, discrete_1, ..., discrete_N, continuous]
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
        text_loss_weight: float = 0.1,
        flow_sample_steps: int = 16,
    ):
        super().__init__()
        if num_discrete_tokens < 1:
            raise ValueError("num_discrete_tokens must be at least 1.")
        if discrete_vocab_size < 1:
            raise ValueError("discrete_vocab_size must be at least 1.")
        if discrete_vocab_size > llama_config.vocab_size:
            raise ValueError(
                "discrete_vocab_size must be <= llama_config.vocab_size when "
                "text and discrete embeddings are unified."
            )
        if continuous_latent_size < 1:
            raise ValueError("continuous_latent_size must be at least 1.")
        if text_loss_weight < 0.0:
            raise ValueError("text_loss_weight must be >= 0.")
        if flow_sample_steps < 1:
            raise ValueError("flow_sample_steps must be at least 1.")

        self.hidden_size = llama_config.hidden_size
        self.num_discrete_tokens = num_discrete_tokens
        self.discrete_vocab_size = discrete_vocab_size
        self.continuous_latent_size = continuous_latent_size
        self.block_size = num_discrete_tokens + 2  # text + N discrete + continuous
        self.flow_loss_weight = flow_loss_weight
        self.text_loss_weight = text_loss_weight
        self.flow_sample_steps = flow_sample_steps

        self.backbone = LlamaBackbone(llama_config)
        # Text and discrete streams share one token embedding table.
        self.discrete_lm_head = nn.Linear(self.hidden_size, discrete_vocab_size, bias=False)
        self.continuous_proj = nn.Linear(continuous_latent_size, self.hidden_size)

        # Stream type ids in block order: [text, d1, ..., dN, continuous].
        self.stream_type_embeddings = nn.Parameter(
            torch.empty(self.block_size, self.hidden_size)
        )

        self.flow_head = FlowHead(
            in_channels=continuous_latent_size,
            model_channels=flow_model_channels or self.hidden_size,
            out_channels=continuous_latent_size,
            z_channels=self.hidden_size,
            num_res_blocks=flow_num_res_blocks,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = getattr(self.backbone.config, "initializer_range", 0.02)
        nn.init.normal_(self.discrete_lm_head.weight, mean=0.0, std=std)
        nn.init.normal_(self.continuous_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.stream_type_embeddings, mean=0.0, std=std)
        if self.continuous_proj.bias is not None:
            nn.init.zeros_(self.continuous_proj.bias)

    @property
    def text_embedding(self) -> nn.Embedding:
        return self.backbone.embed_tokens

    @property
    def discrete_embedding(self) -> nn.Embedding:
        return self.backbone.embed_tokens

    def _normalize_text_tokens(
        self,
        text_tokens: torch.LongTensor,
        name: str,
    ) -> torch.LongTensor:
        if text_tokens.dim() != 2:
            raise ValueError(f"{name} must have shape [batch, length].")
        return text_tokens

    def _normalize_discrete_tokens(
        self,
        discrete_tokens: torch.LongTensor,
        name: str,
    ) -> torch.LongTensor:
        if discrete_tokens.dim() != 3:
            raise ValueError(
                f"{name} must have shape [batch, num_discrete_tokens, length] "
                f"or [batch, length, num_discrete_tokens]."
            )
        if discrete_tokens.shape[1] == self.num_discrete_tokens:
            return discrete_tokens.transpose(1, 2).contiguous()
        if discrete_tokens.shape[-1] == self.num_discrete_tokens:
            return discrete_tokens
        raise ValueError(
            f"{name} must contain one axis with size num_discrete_tokens="
            f"{self.num_discrete_tokens}, got shape={tuple(discrete_tokens.shape)}."
        )

    def _normalize_continuous_latents(
        self,
        continuous_latents: torch.FloatTensor,
        expected_len: int,
        name: str,
    ) -> torch.FloatTensor:
        if continuous_latents.dim() == 2:
            if self.continuous_latent_size != 1:
                raise ValueError(
                    f"{name} with shape [batch, length] is only valid when "
                    "continuous_latent_size == 1."
                )
            continuous_latents = continuous_latents.unsqueeze(-1)
        elif continuous_latents.dim() != 3:
            raise ValueError(
                f"{name} must have shape [batch, length] or "
                f"[batch, length, {self.continuous_latent_size}]."
            )

        if continuous_latents.shape[1] != expected_len:
            raise ValueError(
                f"{name} length mismatch: expected {expected_len}, got "
                f"{continuous_latents.shape[1]}."
            )
        if continuous_latents.shape[-1] != self.continuous_latent_size:
            raise ValueError(
                f"{name} channel mismatch: expected {self.continuous_latent_size}, "
                f"got {continuous_latents.shape[-1]}."
            )
        return continuous_latents

    def _normalize_lengths(
        self,
        lengths: Optional[torch.Tensor | int],
        batch_size: int,
        max_length: int,
        name: str,
        device: torch.device,
        default_value: Optional[int] = None,
    ) -> torch.LongTensor:
        if lengths is None:
            if default_value is None:
                raise ValueError(f"{name} must be provided.")
            lengths_tensor = torch.full(
                (batch_size,),
                int(default_value),
                dtype=torch.long,
                device=device,
            )
        elif isinstance(lengths, int):
            lengths_tensor = torch.full(
                (batch_size,),
                int(lengths),
                dtype=torch.long,
                device=device,
            )
        else:
            lengths_tensor = torch.as_tensor(lengths, dtype=torch.long, device=device)
            if lengths_tensor.dim() == 0:
                lengths_tensor = lengths_tensor.repeat(batch_size)
            if lengths_tensor.dim() != 1 or lengths_tensor.shape[0] != batch_size:
                raise ValueError(f"{name} must have shape [batch].")

        if lengths_tensor.numel() == 0:
            raise ValueError(f"{name} must not be empty.")
        if (lengths_tensor < 0).any() or (lengths_tensor > max_length).any():
            raise ValueError(
                f"{name} values must be in [0, {max_length}], got "
                f"min={int(lengths_tensor.min().item())}, max={int(lengths_tensor.max().item())}."
            )
        return lengths_tensor

    def _build_target_block_mask(
        self,
        total_blocks: int,
        prompt_lens: torch.LongTensor,
        target_lens: torch.LongTensor,
        device: torch.device,
    ) -> torch.BoolTensor:
        block_ids = torch.arange(total_blocks, device=device).unsqueeze(0)
        target_starts = prompt_lens.unsqueeze(1)
        target_ends = (prompt_lens + target_lens).clamp(max=total_blocks).unsqueeze(1)
        # True only on generated target blocks; prompt and tail blocks are excluded.
        return (block_ids >= target_starts) & (block_ids < target_ends)

    def _prepare_attention_masks(
        self,
        attention_mask: Optional[torch.Tensor],
        prompt_lens: torch.LongTensor,
        total_blocks: int,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must be a 2D tensor.")
        if attention_mask.shape[0] != batch_size:
            raise ValueError("attention_mask batch size must match inputs.")

        # Non-zero entries are treated as visible/valid positions.
        mask = attention_mask.to(device=device)
        if mask.shape[-1] == total_blocks:
            # Block-level mask: directly marks valid blocks.
            return mask.to(dtype=torch.bool)
        if mask.shape[-1] == total_blocks * self.block_size:
            # Token-level mask collapsed to blocks: any masked token invalidates its block.
            return mask.reshape(batch_size, total_blocks, self.block_size).all(dim=-1)

        unique_prompt_lens = torch.unique(prompt_lens)
        if unique_prompt_lens.numel() != 1:
            raise ValueError(
                "attention_mask with generated-only shape requires identical prompt lengths "
                "across the batch."
            )
        prompt_len = int(unique_prompt_lens.item())
        generated_len = total_blocks - prompt_len

        if mask.shape[-1] == generated_len:
            prompt_prefix = torch.ones(
                batch_size,
                prompt_len,
                device=device,
                dtype=mask.dtype,
            )
            # Generated-only block mask: prompt blocks stay visible for conditioning.
            block_mask = torch.cat([prompt_prefix, mask], dim=-1).to(dtype=torch.bool)
            return block_mask

        if mask.shape[-1] == generated_len * self.block_size:
            prompt_prefix = torch.ones(
                batch_size,
                prompt_len * self.block_size,
                device=device,
                dtype=mask.dtype,
            )
            # Generated-only token mask: prepend visible prompt tokens, then collapse to blocks.
            flat_mask = torch.cat([prompt_prefix, mask], dim=-1)
            block_mask = flat_mask.reshape(batch_size, total_blocks, self.block_size).all(dim=-1)
            return block_mask

        raise ValueError(
            "attention_mask must be one of: "
            f"[B, {generated_len}], [B, {generated_len * self.block_size}], "
            f"[B, {total_blocks}], or [B, {total_blocks * self.block_size}]."
        )

    def _flatten_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        num_blocks: int,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must be a 2D tensor.")
        if attention_mask.shape[-1] == num_blocks:
            # Expand each block flag to all streams/tokens in that block.
            return attention_mask.repeat_interleave(self.block_size, dim=-1)
        if attention_mask.shape[-1] == num_blocks * self.block_size:
            return attention_mask
        raise ValueError(
            "attention_mask must be block-level [batch, num_blocks] or token-level "
            "[batch, num_blocks * block_size]."
        )

    def _block_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        num_blocks: int,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.shape[-1] == num_blocks:
            return attention_mask.to(dtype=torch.bool)
        flat_mask = self._flatten_attention_mask(attention_mask, num_blocks)
        # Aggregate back to block visibility for loss masking and target selection.
        return flat_mask.reshape(flat_mask.shape[0], num_blocks, self.block_size).all(dim=-1)

    def _build_dual_attention_masks(
        self,
        batch_size: int,
        device: torch.device,
        total_blocks: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if total_blocks is None:
            if total_tokens is None:
                raise ValueError("Specify either total_blocks or total_tokens.")
            if total_tokens % self.block_size != 0:
                raise ValueError("total_tokens must be divisible by block_size.")
            total_blocks = total_tokens // self.block_size
        elif total_tokens is not None and total_tokens != total_blocks * self.block_size:
            raise ValueError("total_tokens and total_blocks are inconsistent.")

        total_tokens = total_blocks * self.block_size
        positions = torch.arange(total_tokens, device=device)
        query_positions = positions[:, None]
        key_positions = positions[None, :]
        # Base autoregressive mask: disallow looking ahead in token time.
        causal_mask = key_positions <= query_positions

        query_blocks = query_positions // self.block_size
        key_blocks = key_positions // self.block_size

        # Stream-wise mask: token attends only to the same stream slot across past/current blocks.
        stream_mask = causal_mask & (
            (query_positions % self.block_size) == (key_positions % self.block_size)
        )
        # Block-wise mask: token attends to all streams from past/current blocks.
        block_mask = key_blocks <= query_blocks

        stream_mask = stream_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        block_mask = block_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        return stream_mask, block_mask

    def _build_inputs_embeds(
        self,
        text_tokens: torch.LongTensor,
        discrete_tokens: torch.LongTensor,
        continuous_latents: torch.FloatTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_embeds = self.text_embedding(text_tokens).unsqueeze(2)  # [B, L, 1, H]
        discrete_embeds = self.discrete_embedding(discrete_tokens)  # [B, L, N, H]
        continuous_embeds = self.continuous_proj(continuous_latents).unsqueeze(2)  # [B, L, 1, H]

        block_embeds = torch.cat([text_embeds, discrete_embeds, continuous_embeds], dim=2)
        block_embeds = block_embeds + self.stream_type_embeddings.view(
            1,
            1,
            self.block_size,
            self.hidden_size,
        )
        inputs_embeds = block_embeds.reshape(
            block_embeds.shape[0],
            block_embeds.shape[1] * self.block_size,
            self.hidden_size,
        )
        return inputs_embeds, discrete_embeds

    def _build_two_level_rope_position_embeddings(
        self,
        inputs_embeds: torch.FloatTensor,
        total_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device

        block_ids = torch.arange(total_blocks, device=device).repeat_interleave(self.block_size)
        stream_ids = torch.arange(self.block_size, device=device).repeat(total_blocks)
        block_ids = block_ids.unsqueeze(0).expand(batch_size, -1)
        stream_ids = stream_ids.unsqueeze(0).expand(batch_size, -1)

        block_cos, block_sin = self.backbone.rotary_emb(inputs_embeds, position_ids=block_ids)
        stream_cos, stream_sin = self.backbone.rotary_emb(inputs_embeds, position_ids=stream_ids)

        # Applying RoPE twice is equivalent to composing the two rotations.
        cos = block_cos * stream_cos - block_sin * stream_sin
        sin = block_sin * stream_cos + block_cos * stream_sin
        return cos, sin

    def _build_step_attention_masks(
        self,
        batch_size: int,
        total_blocks: int,
        block_idx: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stream_full, block_full = self._build_dual_attention_masks(
            batch_size=batch_size,
            device=device,
            total_blocks=total_blocks,
        )
        start = block_idx * self.block_size
        end = start + self.block_size
        # Keep only current-block queries; keys include all tokens up to current block.
        stream_step = stream_full[:, :, start:end, :end]
        block_step = block_full[:, :, start:end, :end]
        return stream_step, block_step

    def _build_step_rope_position_embeddings(
        self,
        inputs_embeds: torch.FloatTensor,
        block_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device
        block_ids = torch.full(
            (batch_size, self.block_size),
            fill_value=block_idx,
            dtype=torch.long,
            device=device,
        )
        stream_ids = torch.arange(self.block_size, device=device).unsqueeze(0).expand(
            batch_size,
            -1,
        )
        block_cos, block_sin = self.backbone.rotary_emb(inputs_embeds, position_ids=block_ids)
        stream_cos, stream_sin = self.backbone.rotary_emb(inputs_embeds, position_ids=stream_ids)
        cos = block_cos * stream_cos - block_sin * stream_sin
        sin = block_sin * stream_cos + block_cos * stream_sin
        return cos, sin

    def _discrete_condition(self, discrete_tokens: torch.LongTensor) -> torch.Tensor:
        return self.discrete_embedding(discrete_tokens).sum(dim=-2)

    def _sample_flow_training_inputs(
        self,
        continuous_targets: torch.FloatTensor,
        flow_timesteps: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        hidden_states: torch.Tensor,
        full_discrete_tokens: torch.LongTensor,
        target_block_mask: torch.BoolTensor,
        block_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, total_tokens, _ = hidden_states.shape
        total_blocks = total_tokens // self.block_size
        if total_blocks < 2:
            return hidden_states.new_zeros(())

        block_hidden = hidden_states.reshape(
            batch_size,
            total_blocks,
            self.block_size,
            self.hidden_size,
        )

        source_discrete_states = block_hidden[:, :-1, 1 : 1 + self.num_discrete_tokens, :]
        discrete_logits = self.discrete_lm_head(source_discrete_states)  # [B, L-1, N, V]
        target_discrete_tokens = full_discrete_tokens[:, 1:, :]  # [B, L-1, N]

        if target_block_mask.shape != (batch_size, total_blocks):
            raise ValueError(
                "target_block_mask must have shape "
                f"[{batch_size}, {total_blocks}], got {tuple(target_block_mask.shape)}."
            )
        # Train only on generated target blocks (shifted by one block for AR prediction).
        valid_targets = target_block_mask[:, 1:].unsqueeze(-1).expand(
            batch_size,
            -1,
            self.num_discrete_tokens,
        )
        if block_attention_mask is not None:
            # Further drop blocks marked invalid by user-provided attention masking.
            valid_targets = valid_targets & block_attention_mask[:, 1:].unsqueeze(-1)

        if not valid_targets.any():
            return hidden_states.new_zeros(())
        return F.cross_entropy(discrete_logits[valid_targets], target_discrete_tokens[valid_targets])

    def _compute_text_loss(
        self,
        hidden_states: torch.Tensor,
        full_text_tokens: torch.LongTensor,
        target_block_mask: torch.BoolTensor,
        block_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, total_tokens, _ = hidden_states.shape
        total_blocks = total_tokens // self.block_size
        if total_blocks < 2:
            return hidden_states.new_zeros(())

        if full_text_tokens.shape != (batch_size, total_blocks):
            raise ValueError(
                "full_text_tokens must have shape "
                f"[{batch_size}, {total_blocks}], got {tuple(full_text_tokens.shape)}."
            )
        if target_block_mask.shape != (batch_size, total_blocks):
            raise ValueError(
                "target_block_mask must have shape "
                f"[{batch_size}, {total_blocks}], got {tuple(target_block_mask.shape)}."
            )

        block_hidden = hidden_states.reshape(
            batch_size,
            total_blocks,
            self.block_size,
            self.hidden_size,
        )
        source_text_states = block_hidden[:, :-1, 0, :]  # [B, L-1, H]
        text_logits = F.linear(source_text_states, self.text_embedding.weight)  # [B, L-1, V]
        target_text_tokens = full_text_tokens[:, 1:]  # [B, L-1]

        # Same generated-target gating used for text auxiliary loss.
        valid_targets = target_block_mask[:, 1:].clone()
        if block_attention_mask is not None:
            valid_targets = valid_targets & block_attention_mask[:, 1:]
        pad_token_id = self.backbone.config.pad_token_id
        if pad_token_id is not None:
            valid_targets = valid_targets & (target_text_tokens != int(pad_token_id))

        if not valid_targets.any():
            return hidden_states.new_zeros(())
        return F.cross_entropy(text_logits[valid_targets], target_text_tokens[valid_targets])

    def _compute_flow_loss(
        self,
        full_discrete_tokens: torch.LongTensor,
        full_continuous_latents: torch.FloatTensor,
        target_block_mask: torch.BoolTensor,
        block_attention_mask: Optional[torch.Tensor],
        flow_timesteps: Optional[torch.FloatTensor],
        noise: Optional[torch.FloatTensor],
    ) -> torch.Tensor:
        batch_size, total_blocks, channels = full_continuous_latents.shape
        if target_block_mask.shape != (batch_size, total_blocks):
            raise ValueError(
                "target_block_mask must have shape "
                f"[{batch_size}, {total_blocks}], got {tuple(target_block_mask.shape)}."
            )

        # Flow loss only uses generated blocks; prompt blocks are excluded.
        valid_blocks = target_block_mask
        if block_attention_mask is not None:
            # External attention mask can additionally suppress specific blocks.
            valid_blocks = valid_blocks & block_attention_mask
        if not valid_blocks.any():
            return full_continuous_latents.new_zeros(())

        def _expand_generated_timesteps(
            timesteps: torch.FloatTensor,
        ) -> torch.FloatTensor:
            if timesteps.dim() != 2 or timesteps.shape[0] != batch_size:
                raise ValueError("flow_timesteps must have shape [batch, length].")
            if timesteps.shape[1] == total_blocks:
                return timesteps

            target_counts = target_block_mask.sum(dim=1)
            max_target = int(target_counts.max().item()) if target_counts.numel() > 0 else 0
            if timesteps.shape[1] != max_target:
                raise ValueError(
                    "flow_timesteps must have shape [batch, total_blocks] or "
                    "[batch, max_target_blocks]."
                )
            expanded = torch.zeros(
                batch_size,
                total_blocks,
                dtype=timesteps.dtype,
                device=timesteps.device,
            )
            for sample_idx in range(batch_size):
                target_positions = torch.nonzero(
                    target_block_mask[sample_idx],
                    as_tuple=False,
                ).squeeze(-1)
                if target_positions.numel() == 0:
                    continue
                expanded[sample_idx, target_positions] = timesteps[
                    sample_idx,
                    : target_positions.numel(),
                ]
            return expanded

        def _expand_generated_noise(noise_tensor: torch.FloatTensor) -> torch.FloatTensor:
            if noise_tensor.dim() != 3 or noise_tensor.shape[0] != batch_size:
                raise ValueError(
                    "noise must have shape [batch, length, continuous_latent_size]."
                )
            if noise_tensor.shape[2] != channels:
                raise ValueError(
                    "noise channel mismatch: expected "
                    f"{channels}, got {noise_tensor.shape[2]}."
                )
            if noise_tensor.shape[1] == total_blocks:
                return noise_tensor

            target_counts = target_block_mask.sum(dim=1)
            max_target = int(target_counts.max().item()) if target_counts.numel() > 0 else 0
            if noise_tensor.shape[1] != max_target:
                raise ValueError(
                    "noise must have shape [batch, total_blocks, continuous_latent_size] "
                    "or [batch, max_target_blocks, continuous_latent_size]."
                )
            expanded = torch.zeros(
                batch_size,
                total_blocks,
                channels,
                dtype=noise_tensor.dtype,
                device=noise_tensor.device,
            )
            for sample_idx in range(batch_size):
                target_positions = torch.nonzero(
                    target_block_mask[sample_idx],
                    as_tuple=False,
                ).squeeze(-1)
                if target_positions.numel() == 0:
                    continue
                expanded[sample_idx, target_positions, :] = noise_tensor[
                    sample_idx,
                    : target_positions.numel(),
                    :,
                ]
            return expanded

        expanded_flow_timesteps = None
        if flow_timesteps is not None:
            expanded_flow_timesteps = _expand_generated_timesteps(
                flow_timesteps.to(
                    device=full_continuous_latents.device,
                    dtype=full_continuous_latents.dtype,
                )
            )

        expanded_noise = None
        if noise is not None:
            expanded_noise = _expand_generated_noise(
                noise.to(
                    device=full_continuous_latents.device,
                    dtype=full_continuous_latents.dtype,
                )
            )

        generated_continuous = full_continuous_latents[valid_blocks]
        generated_discrete = full_discrete_tokens[valid_blocks]
        cond = self._discrete_condition(generated_discrete)

        selected_flow_timesteps = (
            None if expanded_flow_timesteps is None else expanded_flow_timesteps[valid_blocks]
        )
        selected_noise = None if expanded_noise is None else expanded_noise[valid_blocks]

        flow_inputs, flow_target, sampled_timesteps = self._sample_flow_training_inputs(
            continuous_targets=generated_continuous,
            flow_timesteps=selected_flow_timesteps,
            noise=selected_noise,
        )
        flow_prediction = self.flow_head(
            flow_inputs,
            sampled_timesteps,
            cond,
        )
        return F.mse_loss(flow_prediction, flow_target)

    def _top_k_top_p_filter(
        self,
        logits: torch.Tensor,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Mask tail tokens outside nucleus so sampling keeps only top-p probability mass.
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(
                -1,
                sorted_indices,
                sorted_logits,
            )
        return logits

    def _sample_discrete_ids(
        self,
        logits: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> torch.LongTensor:
        if not do_sample or temperature <= 0.0:
            return torch.argmax(logits, dim=-1)

        scaled_logits = logits / temperature
        filtered_logits = self._top_k_top_p_filter(
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

    def forward(
        self,
        text: torch.LongTensor,
        discrete: torch.LongTensor,
        continuous: torch.FloatTensor,
        prompt_lengths: torch.LongTensor,
        target_lengths: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        flow_timesteps: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> PrismTTSOutput | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_text = self._normalize_text_tokens(text, "text")
        full_discrete = self._normalize_discrete_tokens(discrete, "discrete")
        total_blocks = int(full_text.shape[1])
        if full_discrete.shape[1] != total_blocks:
            raise ValueError("text and discrete must have the same length.")
        full_continuous = self._normalize_continuous_latents(
            continuous,
            expected_len=total_blocks,
            name="continuous",
        )
        batch_size = int(full_text.shape[0])
        if full_discrete.shape[0] != batch_size or full_continuous.shape[0] != batch_size:
            raise ValueError("All input tensors must have the same batch size.")

        prompt_lens = self._normalize_lengths(
            lengths=prompt_lengths,
            batch_size=batch_size,
            max_length=total_blocks,
            name="prompt_lengths",
            device=full_text.device,
        )
        if target_lengths is None:
            if attention_mask is None:
                raise ValueError(
                    "target_lengths is required when using concatenated inputs without "
                    "attention_mask."
                )
            block_mask_for_lengths = self._prepare_attention_masks(
                attention_mask=attention_mask,
                prompt_lens=prompt_lens,
                total_blocks=total_blocks,
                batch_size=batch_size,
                device=full_text.device,
            )
            if block_mask_for_lengths is None:
                raise ValueError("Failed to derive target_lengths from attention_mask.")
            target_lens = (block_mask_for_lengths.sum(dim=-1) - prompt_lens).clamp(min=0)
        else:
            target_lens = self._normalize_lengths(
                lengths=target_lengths,
                batch_size=batch_size,
                max_length=total_blocks,
                name="target_lengths",
                device=full_text.device,
            )

        target_block_mask = self._build_target_block_mask(
            total_blocks=total_blocks,
            prompt_lens=prompt_lens,
            target_lens=target_lens,
            device=full_text.device,
        )
        # target_block_mask: True for blocks that contribute training losses.

        block_attention_mask = self._prepare_attention_masks(
            attention_mask=attention_mask,
            prompt_lens=prompt_lens,
            total_blocks=total_blocks,
            batch_size=batch_size,
            device=full_text.device,
        )
        # block_attention_mask: optional validity gate from input mask (applied on top of targets).
        inputs_embeds, _ = self._build_inputs_embeds(
            text_tokens=full_text,
            discrete_tokens=full_discrete,
            continuous_latents=full_continuous,
        )
        # inputs_embeds: [B, (L1 + L2) * block_size, H]
        stream_mask, block_mask = self._build_dual_attention_masks(
            total_blocks=total_blocks,
            batch_size=batch_size,
            device=inputs_embeds.device,
        )
        # stream_mask/block_mask: stream-only history vs full-block history visibility.
        # Keep target text tokens visible as keys during training so discrete prediction
        # conditioning better matches generation-time behavior.
        position_embeddings = self._build_two_level_rope_position_embeddings(
            inputs_embeds=inputs_embeds,
            total_blocks=total_blocks,
        )
        # position_embeddings = (cos, sin), each [B, T, head_dim]

        backbone_outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            streamwise_attention_mask=stream_mask,
            blockwise_attention_mask=block_mask,
            position_embeddings=position_embeddings,
            return_dict=True,
        )
        # backbone_outputs.last_hidden_state: [B, T, H]

        # AR over discrete only: target is shifted by one whole block.
        discrete_loss = self._compute_discrete_loss(
            hidden_states=backbone_outputs.last_hidden_state,
            full_discrete_tokens=full_discrete,
            target_block_mask=target_block_mask,
            block_attention_mask=block_attention_mask,
        )
        # Auxiliary AR over text to directly supervise text conditioning.
        text_loss = self._compute_text_loss(
            hidden_states=backbone_outputs.last_hidden_state,
            full_text_tokens=full_text,
            target_block_mask=target_block_mask,
            block_attention_mask=block_attention_mask,
        )
        # Flow over continuous only: conditioned by current block discrete sum.
        flow_loss = self._compute_flow_loss(
            full_discrete_tokens=full_discrete,
            full_continuous_latents=full_continuous,
            target_block_mask=target_block_mask,
            block_attention_mask=block_attention_mask,
            flow_timesteps=flow_timesteps,
            noise=noise,
        )
        loss = (
            discrete_loss
            + self.flow_loss_weight * flow_loss
            + self.text_loss_weight * text_loss
        )

        if not return_dict:
            return loss, discrete_loss, flow_loss, text_loss

        return PrismTTSOutput(
            loss=loss,
            discrete_loss=discrete_loss,
            flow_loss=flow_loss,
            text_loss=text_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        continuous_prompt: torch.FloatTensor,
        text_target: Optional[torch.LongTensor] = None,
        max_new_blocks: Optional[int] = 128,
        text_eos_token_id: Optional[int] = None,
        discrete_eos_token_id: int = 0,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        flow_num_steps: Optional[int] = None,
        return_dict: bool = True,
    ) -> PrismTTSGenerationOutput | tuple[torch.LongTensor, torch.FloatTensor]:
        # text_prompt: [B, L1]
        text_prompt = self._normalize_text_tokens(text_prompt, "text_prompt")
        # discrete_prompt normalized to [B, L, N]
        discrete_prompt = self._normalize_discrete_tokens(discrete_prompt, "discrete_prompt")
        prompt_len = text_prompt.shape[1]
        continuous_prompt = self._normalize_continuous_latents(
            continuous_prompt,
            expected_len=prompt_len,
            name="continuous_prompt",
        )
        # continuous_prompt: [B, L, C]

        batch_size = text_prompt.shape[0]
        if (
            discrete_prompt.shape[0] != batch_size
            or continuous_prompt.shape[0] != batch_size
        ):
            raise ValueError("All prompt tensors must have matching batch size.")

        if text_target is None:
            text_target = text_prompt.new_zeros((batch_size, 0))
        else:
            text_target = self._normalize_text_tokens(text_target, "text_target")
            if text_target.shape[0] != batch_size:
                raise ValueError("text_target must have the same batch size as prompts.")

        target_len = text_target.shape[1]
        if max_new_blocks is None:
            max_new_blocks = max(1, target_len)
        if max_new_blocks < 1:
            raise ValueError("max_new_blocks must be at least 1.")

        if text_eos_token_id is None:
            text_eos_token_id = self.backbone.config.eos_token_id
            if text_eos_token_id is None:
                text_eos_token_id = self.backbone.config.pad_token_id
            if text_eos_token_id is None:
                text_eos_token_id = 0
        discrete_draft_token_id = (
            discrete_eos_token_id
            if 0 <= discrete_eos_token_id < self.discrete_vocab_size
            else 0
        )

        cur_text = text_prompt  # [B, cur_L]
        cur_discrete = discrete_prompt  # [B, cur_L, N]
        cur_continuous = continuous_prompt  # [B, cur_L, C]

        generated_text = []
        generated_discrete = []
        generated_continuous = []
        collected_logits = []

        for step_idx in range(max_new_blocks):
            if step_idx < target_len:
                next_text = text_target[:, step_idx : step_idx + 1]
            else:
                next_text = torch.full(
                    (batch_size, 1),
                    fill_value=text_eos_token_id,
                    dtype=cur_text.dtype,
                    device=cur_text.device,
                )

            # Put text token into the block before sampling this block.
            draft_discrete = torch.full(
                (batch_size, 1, self.num_discrete_tokens),
                fill_value=discrete_draft_token_id,
                dtype=cur_discrete.dtype,
                device=cur_discrete.device,
            )
            draft_continuous = torch.zeros(
                batch_size,
                1,
                self.continuous_latent_size,
                dtype=cur_continuous.dtype,
                device=cur_continuous.device,
            )
            cur_text = torch.cat([cur_text, next_text], dim=1)  # [B, cur_L+1]
            cur_discrete = torch.cat([cur_discrete, draft_discrete], dim=1)  # [B, cur_L+1, N]
            cur_continuous = torch.cat([cur_continuous, draft_continuous], dim=1)  # [B, cur_L+1, C]

            total_blocks = cur_text.shape[1]
            inputs_embeds, _ = self._build_inputs_embeds(
                text_tokens=cur_text,
                discrete_tokens=cur_discrete,
                continuous_latents=cur_continuous,
            )
            # inputs_embeds: [B, cur_L * block_size, H]
            stream_mask, block_mask = self._build_dual_attention_masks(
                total_blocks=total_blocks,
                batch_size=batch_size,
                device=inputs_embeds.device,
            )
            # stream_mask enforces same-stream AR, block_mask lets each stream read full past blocks.
            position_embeddings = self._build_two_level_rope_position_embeddings(
                inputs_embeds=inputs_embeds,
                total_blocks=total_blocks,
            )
            # position_embeddings = (cos, sin), each [B, T, head_dim]
            backbone_outputs = self.backbone(
                inputs_embeds=inputs_embeds,
                streamwise_attention_mask=stream_mask,
                blockwise_attention_mask=block_mask,
                position_embeddings=position_embeddings,
                return_dict=True,
            )
            # backbone_outputs.last_hidden_state: [B, T, H]

            block_hidden = backbone_outputs.last_hidden_state.reshape(
                batch_size,
                total_blocks,
                self.block_size,
                self.hidden_size,
            )
            # block_hidden: [B, cur_L, block_size, H]
            # Predict this block's discrete tokens from the just-injected block.
            next_discrete_states = block_hidden[:, -1, 1 : 1 + self.num_discrete_tokens, :]
            # next_discrete_states: [B, N, H]
            next_logits = self.discrete_lm_head(next_discrete_states)  # [B, N, V]
            next_discrete = self._sample_discrete_ids(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            # next_discrete: [B, N]
            next_continuous = self.sample_continuous_latent(
                cond=self._discrete_condition(next_discrete),
                num_steps=flow_num_steps,
            )
            # next_continuous: [B, C]

            cur_discrete[:, -1, :] = next_discrete
            cur_continuous[:, -1, :] = next_continuous

            generated_text.append(next_text)
            generated_discrete.append(next_discrete.unsqueeze(1))  # each: [B, 1, N]
            generated_continuous.append(next_continuous.unsqueeze(1))  # each: [B, 1, C]
            collected_logits.append(next_logits)

            eos_block = (next_text.squeeze(-1) == text_eos_token_id) & (
                next_discrete == discrete_eos_token_id
            ).all(dim=-1)
            if eos_block.all():
                break

        generated_text = torch.cat(generated_text, dim=1)
        # generated_text: [B, Lgen]
        generated_discrete = torch.cat(generated_discrete, dim=1).transpose(1, 2).contiguous()
        # generated_discrete: [B, N, L2]
        generated_continuous = torch.cat(generated_continuous, dim=1)
        # generated_continuous: [B, L2, C]

        if not return_dict:
            return generated_discrete, generated_continuous

        return PrismTTSGenerationOutput(
            text_ids=generated_text,
            discrete_ids=generated_discrete,
            continuous_latents=generated_continuous,
            discrete_logits=tuple(collected_logits),
        )

    @torch.no_grad()
    def generate_with_kv_cache(
        self,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        continuous_prompt: torch.FloatTensor,
        text_target: Optional[torch.LongTensor] = None,
        max_new_blocks: Optional[int] = 128,
        text_eos_token_id: Optional[int] = None,
        discrete_eos_token_id: int = 0,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        flow_num_steps: Optional[int] = None,
        return_dict: bool = True,
    ) -> PrismTTSGenerationOutput | tuple[torch.LongTensor, torch.FloatTensor]:
        # text_prompt: [B, L1]
        text_prompt = self._normalize_text_tokens(text_prompt, "text_prompt")
        # discrete_prompt normalized to [B, L1, N]
        discrete_prompt = self._normalize_discrete_tokens(discrete_prompt, "discrete_prompt")
        prompt_len = text_prompt.shape[1]
        continuous_prompt = self._normalize_continuous_latents(
            continuous_prompt,
            expected_len=prompt_len,
            name="continuous_prompt",
        )
        # continuous_prompt: [B, L1, C]

        batch_size = text_prompt.shape[0]
        if (
            discrete_prompt.shape[0] != batch_size
            or continuous_prompt.shape[0] != batch_size
        ):
            raise ValueError("All prompt tensors must have matching batch size.")

        if text_target is None:
            text_target = text_prompt.new_zeros((batch_size, 0))
        else:
            text_target = self._normalize_text_tokens(text_target, "text_target")
            if text_target.shape[0] != batch_size:
                raise ValueError("text_target must have the same batch size as prompts.")

        target_len = text_target.shape[1]
        if max_new_blocks is None:
            max_new_blocks = max(1, target_len)
        if max_new_blocks < 1:
            raise ValueError("max_new_blocks must be at least 1.")

        if text_eos_token_id is None:
            text_eos_token_id = self.backbone.config.eos_token_id
            if text_eos_token_id is None:
                text_eos_token_id = self.backbone.config.pad_token_id
            if text_eos_token_id is None:
                text_eos_token_id = 0
        discrete_draft_token_id = (
            discrete_eos_token_id
            if 0 <= discrete_eos_token_id < self.discrete_vocab_size
            else 0
        )

        past_key_values = None
        if prompt_len > 0:
            prompt_embeds, _ = self._build_inputs_embeds(
                text_tokens=text_prompt,
                discrete_tokens=discrete_prompt,
                continuous_latents=continuous_prompt,
            )
            # prompt_embeds: [B, L1 * block_size, H]
            prompt_stream_mask, prompt_block_mask = self._build_dual_attention_masks(
                total_blocks=prompt_len,
                batch_size=batch_size,
                device=prompt_embeds.device,
            )
            # Prompt prefill uses the same stream/block visibility rules as full forward pass.
            prompt_position_embeddings = self._build_two_level_rope_position_embeddings(
                inputs_embeds=prompt_embeds,
                total_blocks=prompt_len,
            )
            # prompt_position_embeddings = (cos, sin), each [B, T1, head_dim]
            prompt_outputs = self.backbone(
                inputs_embeds=prompt_embeds,
                streamwise_attention_mask=prompt_stream_mask,
                blockwise_attention_mask=prompt_block_mask,
                position_embeddings=prompt_position_embeddings,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = prompt_outputs.past_key_values

        generated_text = []
        generated_discrete = []
        generated_continuous = []
        collected_logits = []

        for step_idx in range(max_new_blocks):
            if step_idx < target_len:
                next_text = text_target[:, step_idx : step_idx + 1]
            else:
                next_text = torch.full(
                    (batch_size, 1),
                    fill_value=text_eos_token_id,
                    dtype=text_prompt.dtype,
                    device=text_prompt.device,
                )
            # next_text: [B, 1]

            draft_discrete = torch.full(
                (batch_size, 1, self.num_discrete_tokens),
                fill_value=discrete_draft_token_id,
                dtype=discrete_prompt.dtype,
                device=discrete_prompt.device,
            )
            # draft_discrete: [B, 1, N]
            draft_continuous = torch.zeros(
                batch_size,
                1,
                self.continuous_latent_size,
                dtype=continuous_prompt.dtype,
                device=continuous_prompt.device,
            )
            # draft_continuous: [B, 1, C]

            # Proposal pass to sample discrete tokens for this block.
            proposal_embeds, _ = self._build_inputs_embeds(
                text_tokens=next_text,
                discrete_tokens=draft_discrete,
                continuous_latents=draft_continuous,
            )
            # proposal_embeds: [B, block_size, H]
            block_idx = prompt_len + step_idx
            total_blocks = block_idx + 1
            step_stream_mask, step_block_mask = self._build_step_attention_masks(
                batch_size=batch_size,
                total_blocks=total_blocks,
                block_idx=block_idx,
                device=proposal_embeds.device,
            )
            # Step masks limit queries to current block while preserving full key history.
            step_position_embeddings = self._build_step_rope_position_embeddings(
                inputs_embeds=proposal_embeds,
                block_idx=block_idx,
            )
            # step_position_embeddings = (cos, sin), each [B, block_size, head_dim]

            past_seq_len = 0 if past_key_values is None else past_key_values.get_seq_length()
            proposal_outputs = self.backbone(
                inputs_embeds=proposal_embeds,
                streamwise_attention_mask=step_stream_mask,
                blockwise_attention_mask=step_block_mask,
                position_embeddings=step_position_embeddings,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = proposal_outputs.past_key_values
            # proposal_outputs.last_hidden_state: [B, block_size, H]
            proposal_discrete_states = proposal_outputs.last_hidden_state[
                :,
                1 : 1 + self.num_discrete_tokens,
                :,
            ]
            # proposal_discrete_states: [B, N, H]
            proposal_logits = self.discrete_lm_head(proposal_discrete_states)  # [B, N, V]
            next_discrete = self._sample_discrete_ids(
                proposal_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            # next_discrete: [B, N]
            next_continuous = self.sample_continuous_latent(
                cond=self._discrete_condition(next_discrete),
                num_steps=flow_num_steps,
            )
            # next_continuous: [B, C]

            # Remove drafted KV states, then append finalized block KV states.
            if past_key_values is not None:
                past_key_values.crop(past_seq_len)

            finalized_discrete = next_discrete.unsqueeze(1)  # [B, 1, N]
            finalized_continuous = next_continuous.unsqueeze(1)  # [B, 1, C]
            finalized_embeds, _ = self._build_inputs_embeds(
                text_tokens=next_text,
                discrete_tokens=finalized_discrete,
                continuous_latents=finalized_continuous,
            )
            # finalized_embeds: [B, block_size, H]
            finalize_outputs = self.backbone(
                inputs_embeds=finalized_embeds,
                streamwise_attention_mask=step_stream_mask,
                blockwise_attention_mask=step_block_mask,
                position_embeddings=step_position_embeddings,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = finalize_outputs.past_key_values

            generated_text.append(next_text)
            generated_discrete.append(finalized_discrete)  # each: [B, 1, N]
            generated_continuous.append(finalized_continuous)  # each: [B, 1, C]
            collected_logits.append(proposal_logits)

            eos_block = (next_text.squeeze(-1) == text_eos_token_id) & (
                next_discrete == discrete_eos_token_id
            ).all(dim=-1)
            if eos_block.all():
                break

        generated_text = torch.cat(generated_text, dim=1)
        # generated_text: [B, Lgen]
        generated_discrete = torch.cat(generated_discrete, dim=1).transpose(1, 2).contiguous()
        # generated_discrete: [B, N, Lgen]
        generated_continuous = torch.cat(generated_continuous, dim=1)
        # generated_continuous: [B, Lgen, C]

        if not return_dict:
            return generated_discrete, generated_continuous

        return PrismTTSGenerationOutput(
            text_ids=generated_text,
            discrete_ids=generated_discrete,
            continuous_latents=generated_continuous,
            discrete_logits=tuple(collected_logits),
        )
