from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.utils import ModelOutput

from models.flow_head import FlowHead
from models.llama_backbone import LlamaBackbone


TEXT_TOKEN_TYPE = 0
SPEECH_DISCRETE_TOKEN_TYPE = 1
SPEECH_CONTINUOUS_TOKEN_TYPE = 2


@dataclass
class PrismTTSOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    discrete_loss: Optional[torch.Tensor] = None
    continuous_loss: Optional[torch.Tensor] = None
    flow_loss: Optional[torch.Tensor] = None


@dataclass
class PrismTTSGenerationOutput(ModelOutput):
    text_ids: Optional[torch.LongTensor] = None
    discrete_ids: Optional[torch.LongTensor] = None
    continuous_latents: Optional[torch.FloatTensor] = None
    discrete_logits: Optional[tuple[torch.Tensor, ...]] = None


@dataclass
class _FlatBatch:
    token_ids: torch.LongTensor
    continuous_values: torch.FloatTensor
    token_type_ids: torch.LongTensor
    speech_stream_ids: torch.LongTensor
    target_block_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
    target_block_counts: torch.LongTensor


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
        mask_ratio: float = 0.5,
        discrete_regular_token_loss_weight: float = 1.0,
        discrete_special_token_loss_weight: float = 1.0,
        flow_sample_steps: int = 16,
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
        if not (0.0 <= mask_ratio <= 1.0):
            raise ValueError("mask_ratio must be in [0, 1].")
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

        self.hidden_size = int(llama_config.hidden_size)
        self.num_discrete_tokens = int(num_discrete_tokens)
        self.discrete_vocab_size = int(discrete_vocab_size)
        self.continuous_latent_size = int(continuous_latent_size)
        self.speech_block_size = self.num_discrete_tokens + 1

        self.flow_loss_weight = float(flow_loss_weight)
        self.continuous_loss_weight = float(continuous_loss_weight)
        self.mask_ratio = float(mask_ratio)
        self.discrete_regular_token_loss_weight = float(discrete_regular_token_loss_weight)
        self.discrete_special_token_loss_weight = float(discrete_special_token_loss_weight)
        self.flow_sample_steps = int(flow_sample_steps)

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

        self.training_special_discrete_token_ids = self._infer_special_discrete_token_ids(
            self._resolve_generation_discrete_eos_token_id(None)
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

    def _normalize_text_tokens(
        self,
        text_tokens: torch.LongTensor,
        name: str,
    ) -> torch.LongTensor:
        """Validate text token tensor shape as [batch, length]."""
        if text_tokens.dim() != 2:
            raise ValueError(f"{name} must have shape [batch, length].")
        return text_tokens

    def _normalize_discrete_tokens(
        self,
        discrete_tokens: torch.LongTensor,
        name: str,
    ) -> torch.LongTensor:
        """Normalize discrete tokens to shape [batch, length, num_discrete_tokens]."""
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
        """Validate/reshape continuous latents to [batch, length, continuous_latent_size]."""
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
        """Convert optional length inputs to validated [batch] long tensors."""
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

    def _assemble_flat_batch(
        self,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        continuous_prompt: torch.FloatTensor,
        text_target: torch.LongTensor,
        discrete_target: torch.LongTensor,
        continuous_target: torch.FloatTensor,
        text_prompt_lengths: torch.LongTensor,
        speech_prompt_lengths: torch.LongTensor,
        text_target_lengths: torch.LongTensor,
        speech_target_lengths: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
    ) -> _FlatBatch:
        """Assemble split prompt/target tensors into one flattened sequence representation."""
        batch_size = int(text_prompt.shape[0])
        device = text_prompt.device
        cont_dtype = continuous_prompt.dtype

        token_ids_per_sample: list[torch.LongTensor] = []
        continuous_per_sample: list[torch.FloatTensor] = []
        token_types_per_sample: list[torch.LongTensor] = []
        stream_ids_per_sample: list[torch.LongTensor] = []
        target_block_ids_per_sample: list[torch.LongTensor] = []

        for sample_idx in range(batch_size):
            l1 = int(text_prompt_lengths[sample_idx].item())
            l2 = int(speech_prompt_lengths[sample_idx].item())
            l3 = int(text_target_lengths[sample_idx].item())
            l4 = int(speech_target_lengths[sample_idx].item())

            sample_token_ids: list[int] = []
            sample_types: list[int] = []
            sample_stream_ids: list[int] = []
            sample_target_block_ids: list[int] = []
            sample_continuous: list[torch.Tensor] = []

            def append_text_token(token_id: int) -> None:
                sample_token_ids.append(int(token_id))
                sample_types.append(TEXT_TOKEN_TYPE)
                sample_stream_ids.append(-1)
                sample_target_block_ids.append(-1)
                sample_continuous.append(
                    torch.zeros(
                        self.continuous_latent_size,
                        dtype=cont_dtype,
                        device=device,
                    )
                )

            def append_speech_discrete(token_id: int, stream_id: int, target_block_id: int) -> None:
                sample_token_ids.append(int(token_id))
                sample_types.append(SPEECH_DISCRETE_TOKEN_TYPE)
                sample_stream_ids.append(int(stream_id))
                sample_target_block_ids.append(int(target_block_id))
                sample_continuous.append(
                    torch.zeros(
                        self.continuous_latent_size,
                        dtype=cont_dtype,
                        device=device,
                    )
                )

            def append_speech_continuous(
                value: torch.Tensor,
                target_block_id: int,
            ) -> None:
                sample_token_ids.append(self.pad_token_id)
                sample_types.append(SPEECH_CONTINUOUS_TOKEN_TYPE)
                sample_stream_ids.append(self.num_discrete_tokens)
                sample_target_block_ids.append(int(target_block_id))
                sample_continuous.append(value)

            prompt_text = text_prompt[sample_idx, :l1]
            prompt_discrete = discrete_prompt[sample_idx, :l2, :]
            prompt_continuous = continuous_prompt[sample_idx, :l2, :]
            target_text = text_target[sample_idx, :l3]
            target_discrete = discrete_target[sample_idx, :l4, :]
            target_continuous = continuous_target[sample_idx, :l4, :]

            for token in prompt_text.tolist():
                append_text_token(token)
            append_text_token(self.eot_token_id)

            for block_idx in range(l2):
                for stream_idx in range(self.num_discrete_tokens):
                    append_speech_discrete(
                        token_id=int(prompt_discrete[block_idx, stream_idx].item()),
                        stream_id=stream_idx,
                        target_block_id=-1,
                    )
                append_speech_continuous(
                    value=prompt_continuous[block_idx],
                    target_block_id=-1,
                )
            append_text_token(self.eos_token_id)

            for token in target_text.tolist():
                append_text_token(token)
            append_text_token(self.eot_token_id)

            for block_idx in range(l4):
                for stream_idx in range(self.num_discrete_tokens):
                    append_speech_discrete(
                        token_id=int(target_discrete[block_idx, stream_idx].item()),
                        stream_id=stream_idx,
                        target_block_id=block_idx,
                    )
                append_speech_continuous(
                    value=target_continuous[block_idx],
                    target_block_id=block_idx,
                )
            append_text_token(self.eos_token_id)

            token_ids_per_sample.append(
                torch.tensor(sample_token_ids, dtype=torch.long, device=device)
            )
            token_types_per_sample.append(
                torch.tensor(sample_types, dtype=torch.long, device=device)
            )
            stream_ids_per_sample.append(
                torch.tensor(sample_stream_ids, dtype=torch.long, device=device)
            )
            target_block_ids_per_sample.append(
                torch.tensor(sample_target_block_ids, dtype=torch.long, device=device)
            )
            continuous_per_sample.append(torch.stack(sample_continuous, dim=0))

        max_seq_len = max(int(x.shape[0]) for x in token_ids_per_sample)
        token_ids = torch.full(
            (batch_size, max_seq_len),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        token_type_ids = torch.full(
            (batch_size, max_seq_len),
            TEXT_TOKEN_TYPE,
            dtype=torch.long,
            device=device,
        )
        speech_stream_ids = torch.full(
            (batch_size, max_seq_len),
            -1,
            dtype=torch.long,
            device=device,
        )
        target_block_ids = torch.full(
            (batch_size, max_seq_len),
            -1,
            dtype=torch.long,
            device=device,
        )
        continuous_values = torch.zeros(
            batch_size,
            max_seq_len,
            self.continuous_latent_size,
            dtype=cont_dtype,
            device=device,
        )
        derived_attention_mask = torch.zeros(
            batch_size,
            max_seq_len,
            dtype=torch.bool,
            device=device,
        )

        for sample_idx in range(batch_size):
            seq_len = int(token_ids_per_sample[sample_idx].shape[0])
            token_ids[sample_idx, :seq_len] = token_ids_per_sample[sample_idx]
            token_type_ids[sample_idx, :seq_len] = token_types_per_sample[sample_idx]
            speech_stream_ids[sample_idx, :seq_len] = stream_ids_per_sample[sample_idx]
            target_block_ids[sample_idx, :seq_len] = target_block_ids_per_sample[sample_idx]
            continuous_values[sample_idx, :seq_len, :] = continuous_per_sample[sample_idx]
            derived_attention_mask[sample_idx, :seq_len] = True

        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.shape[0] != batch_size:
                raise ValueError("attention_mask must have shape [batch, sequence].")
            if attention_mask.shape[1] < max_seq_len:
                raise ValueError(
                    "attention_mask length must be >= concatenated sequence length."
                )
            derived_attention_mask = derived_attention_mask & attention_mask[:, :max_seq_len].to(
                device=device,
                dtype=torch.bool,
            )

        return _FlatBatch(
            token_ids=token_ids,
            continuous_values=continuous_values,
            token_type_ids=token_type_ids,
            speech_stream_ids=speech_stream_ids,
            target_block_ids=target_block_ids,
            attention_mask=derived_attention_mask,
            target_block_counts=speech_target_lengths.to(device=device, dtype=torch.long),
        )

    def _sample_masked_target_blocks(
        self,
        target_block_counts: torch.LongTensor,
        mask_ratio: float,
        masked_target_blocks: Optional[torch.BoolTensor],
    ) -> torch.BoolTensor:
        """Sample (or validate provided) masked target-block indices for reconstruction."""
        batch_size = int(target_block_counts.shape[0])
        device = target_block_counts.device
        max_target_blocks = int(target_block_counts.max().item()) if batch_size > 0 else 0
        if max_target_blocks <= 0:
            return torch.zeros((batch_size, 0), dtype=torch.bool, device=device)

        if masked_target_blocks is not None:
            if masked_target_blocks.dim() != 2 or masked_target_blocks.shape[0] != batch_size:
                raise ValueError("masked_target_blocks must have shape [batch, target_blocks].")
            if masked_target_blocks.shape[1] < max_target_blocks:
                raise ValueError(
                    "masked_target_blocks must cover at least max(target_block_counts) columns."
                )
            out = masked_target_blocks[:, :max_target_blocks].to(device=device, dtype=torch.bool).clone()
            for sample_idx in range(batch_size):
                count = int(target_block_counts[sample_idx].item())
                if count < max_target_blocks:
                    out[sample_idx, count:] = False
            return out

        out = torch.zeros((batch_size, max_target_blocks), dtype=torch.bool, device=device)
        for sample_idx in range(batch_size):
            count = int(target_block_counts[sample_idx].item())
            if count <= 0:
                continue
            num_masked = int(round(mask_ratio * count))
            if mask_ratio > 0.0:
                num_masked = max(1, num_masked)
            num_masked = min(count, max(0, num_masked))
            if num_masked == 0:
                continue
            picked = torch.randperm(count, device=device)[:num_masked]
            out[sample_idx, picked] = True
        return out

    def _build_flat_batch_from_collate(
        self,
        *,
        flat_token_ids: torch.LongTensor,
        flat_continuous_values: torch.FloatTensor,
        flat_token_type_ids: torch.LongTensor,
        flat_speech_stream_ids: torch.LongTensor,
        flat_target_block_ids: torch.LongTensor,
        flat_target_block_counts: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
    ) -> _FlatBatch:
        """Validate collate-produced flat tensors and pack them into `_FlatBatch`."""
        if flat_token_ids.dim() != 2:
            raise ValueError("flat_token_ids must have shape [batch, sequence].")
        batch_size, seq_len = flat_token_ids.shape
        device = flat_token_ids.device

        def _require_shape(name: str, tensor: torch.Tensor, expected_last: Optional[int] = None) -> torch.Tensor:
            if expected_last is None and tensor.dim() != 2:
                raise ValueError(f"{name} must be 2D tensor.")
            if expected_last is not None and tensor.dim() != 3:
                raise ValueError(f"{name} must be 3D tensor.")
            if tensor.shape[0] != batch_size or tensor.shape[1] != seq_len:
                raise ValueError(
                    f"{name} must match flat_token_ids shape [batch, sequence], got {tuple(tensor.shape)}."
                )
            if expected_last is not None and tensor.shape[2] != expected_last:
                raise ValueError(
                    f"{name} channel mismatch: expected {expected_last}, got {tensor.shape[2]}."
                )
            return tensor

        flat_continuous_values = _require_shape(
            "flat_continuous_values",
            flat_continuous_values,
            expected_last=self.continuous_latent_size,
        )
        flat_token_type_ids = _require_shape("flat_token_type_ids", flat_token_type_ids)
        flat_speech_stream_ids = _require_shape("flat_speech_stream_ids", flat_speech_stream_ids)
        flat_target_block_ids = _require_shape("flat_target_block_ids", flat_target_block_ids)

        if attention_mask is None:
            resolved_attention = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        else:
            if attention_mask.dim() != 2 or attention_mask.shape[0] != batch_size:
                raise ValueError("attention_mask must have shape [batch, sequence].")
            if attention_mask.shape[1] < seq_len:
                raise ValueError("attention_mask length must be >= flat sequence length.")
            resolved_attention = attention_mask[:, :seq_len].to(device=device, dtype=torch.bool)

        if flat_target_block_counts is None:
            inferred = torch.zeros(batch_size, dtype=torch.long, device=device)
            for sample_idx in range(batch_size):
                valid_mask = resolved_attention[sample_idx] & (flat_target_block_ids[sample_idx] >= 0)
                if valid_mask.any():
                    inferred[sample_idx] = int(flat_target_block_ids[sample_idx][valid_mask].max().item()) + 1
            flat_target_block_counts = inferred
        else:
            flat_target_block_counts = torch.as_tensor(
                flat_target_block_counts,
                dtype=torch.long,
                device=device,
            )
            if flat_target_block_counts.dim() == 0:
                flat_target_block_counts = flat_target_block_counts.repeat(batch_size)
            if flat_target_block_counts.dim() != 1 or flat_target_block_counts.shape[0] != batch_size:
                raise ValueError("flat_target_block_counts must have shape [batch].")
            if (flat_target_block_counts < 0).any():
                raise ValueError("flat_target_block_counts must be non-negative.")

        return _FlatBatch(
            token_ids=flat_token_ids.to(dtype=torch.long, device=device),
            continuous_values=flat_continuous_values.to(device=device),
            token_type_ids=flat_token_type_ids.to(dtype=torch.long, device=device),
            speech_stream_ids=flat_speech_stream_ids.to(dtype=torch.long, device=device),
            target_block_ids=flat_target_block_ids.to(dtype=torch.long, device=device),
            attention_mask=resolved_attention,
            target_block_counts=flat_target_block_counts,
        )

    def _build_inputs_embeds(
        self,
        flat: _FlatBatch,
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

    def _build_two_level_rope_position_embeddings(
        self,
        inputs_embeds: torch.FloatTensor,
        speech_stream_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose global 1D RoPE with within-block stream-index RoPE."""
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        global_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        secondary_position_ids = speech_stream_ids.clamp(min=0)

        global_cos, global_sin = self.backbone.rotary_emb(inputs_embeds, position_ids=global_position_ids)
        secondary_cos, secondary_sin = self.backbone.rotary_emb(
            inputs_embeds,
            position_ids=secondary_position_ids,
        )

        cos = global_cos * secondary_cos - global_sin * secondary_sin
        sin = global_sin * secondary_cos + global_cos * secondary_sin
        return cos, sin

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

    def _top_k_top_p_filter(
        self,
        logits: torch.Tensor,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """Apply top-k and nucleus (top-p) filtering to sampling logits."""
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
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
        """Sample discrete ids from logits with optional temperature/top-k/top-p."""
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

    def _resolve_generation_discrete_eos_token_id(
        self,
        discrete_eos_token_id: Optional[int],
    ) -> int:
        """Resolve the discrete EOS id used during generation."""
        if discrete_eos_token_id is not None:
            return int(discrete_eos_token_id)
        candidate = self.backbone.config.eos_token_id
        if candidate is None or not (0 <= int(candidate) < self.discrete_vocab_size):
            return 0
        return int(candidate)

    def _infer_special_discrete_token_ids(
        self,
        discrete_eos_token_id: int,
    ) -> tuple[int, ...]:
        """Collect discrete token ids treated as special for loss/silence heuristics."""
        ids: set[int] = set()
        if 0 <= discrete_eos_token_id < self.discrete_vocab_size:
            ids.add(int(discrete_eos_token_id))

        cfg_eos = self.backbone.config.eos_token_id
        cfg_pad = self.backbone.config.pad_token_id
        if cfg_eos is not None and 0 <= int(cfg_eos) < self.discrete_vocab_size:
            ids.add(int(cfg_eos))
            if 0 <= int(cfg_eos) - 1 < self.discrete_vocab_size:
                ids.add(int(cfg_eos) - 1)
        if cfg_pad is not None and 0 <= int(cfg_pad) < self.discrete_vocab_size:
            ids.add(int(cfg_pad))
        return tuple(sorted(ids))

    def _build_special_block_mask(
        self,
        discrete_tokens: torch.LongTensor,
        special_token_ids: tuple[int, ...],
    ) -> torch.BoolTensor:
        """Mark blocks where all discrete streams are special tokens."""
        if len(special_token_ids) == 0:
            return torch.zeros(
                discrete_tokens.shape[0],
                dtype=torch.bool,
                device=discrete_tokens.device,
            )
        ids = torch.tensor(
            special_token_ids,
            dtype=discrete_tokens.dtype,
            device=discrete_tokens.device,
        )
        return torch.isin(discrete_tokens, ids).all(dim=-1)

    def _encode(
        self,
        flat: _FlatBatch,
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
        position_embeddings = self._build_two_level_rope_position_embeddings(
            inputs_embeds=inputs_embeds,
            speech_stream_ids=flat.speech_stream_ids,
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
        flat = self._build_flat_batch_from_collate(
            flat_token_ids=flat_token_ids,
            flat_continuous_values=flat_continuous_values,
            flat_token_type_ids=flat_token_type_ids,
            flat_speech_stream_ids=flat_speech_stream_ids,
            flat_target_block_ids=flat_target_block_ids,
            flat_target_block_counts=flat_target_block_counts,
            attention_mask=attention_mask,
        )

        effective_mask_ratio = self.mask_ratio if mask_ratio is None else float(mask_ratio)
        masked_blocks = self._sample_masked_target_blocks(
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
    ) -> PrismTTSGenerationOutput | tuple[torch.LongTensor, torch.FloatTensor]:
        """Generate target speech blocks conditioned on text/speech prompt and target text."""
        text_prompt = self._normalize_text_tokens(text_prompt, "text_prompt")
        discrete_prompt = self._normalize_discrete_tokens(discrete_prompt, "discrete_prompt")
        continuous_prompt = self._normalize_continuous_latents(
            continuous_prompt,
            expected_len=int(discrete_prompt.shape[1]),
            name="continuous_prompt",
        )
        batch_size = int(text_prompt.shape[0])

        if text_target is None:
            text_target = text_prompt.new_zeros((batch_size, 0))
        else:
            text_target = self._normalize_text_tokens(text_target, "text_target")

        if max_new_blocks is not None and int(max_new_blocks) < 0:
            raise ValueError("max_new_blocks must be >= 0 when provided.")

        text_prompt_lengths = self._normalize_lengths(
            lengths=text_prompt_lengths,
            batch_size=batch_size,
            max_length=int(text_prompt.shape[1]),
            name="text_prompt_lengths",
            device=text_prompt.device,
            default_value=int(text_prompt.shape[1]),
        )
        speech_prompt_lengths = self._normalize_lengths(
            lengths=speech_prompt_lengths,
            batch_size=batch_size,
            max_length=int(discrete_prompt.shape[1]),
            name="speech_prompt_lengths",
            device=text_prompt.device,
            default_value=int(discrete_prompt.shape[1]),
        )
        text_target_lengths = self._normalize_lengths(
            lengths=text_target_lengths,
            batch_size=batch_size,
            max_length=int(text_target.shape[1]),
            name="text_target_lengths",
            device=text_prompt.device,
            default_value=int(text_target.shape[1]),
        )
        max_text_target = int(text_target_lengths.max().item()) if batch_size > 0 else 0
        text_ids_out = text_target[:, :max_text_target]

        speech_target_max_length = (
            int(max_new_blocks)
            if max_new_blocks is not None
            else int(text_target.shape[1])
        )
        if speech_target_lengths is None:
            default_speech_target_length = (
                int(text_target.shape[1])
                if max_new_blocks is None
                else int(max_new_blocks)
            )
        else:
            default_speech_target_length = None
            speech_target_max_length = int(
                torch.as_tensor(speech_target_lengths, device=text_prompt.device)
                .to(dtype=torch.long)
                .max()
                .item()
            )

        speech_target_lengths = self._normalize_lengths(
            lengths=speech_target_lengths,
            batch_size=batch_size,
            max_length=speech_target_max_length,
            name="speech_target_lengths",
            device=text_prompt.device,
            default_value=default_speech_target_length,
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

        discrete_eos_id = self._resolve_generation_discrete_eos_token_id(discrete_eos_token_id)

        predicted_discrete = discrete_prompt.new_full(
            (batch_size, max_target, self.num_discrete_tokens),
            discrete_eos_id,
        )
        predicted_continuous = continuous_prompt.new_zeros(
            (batch_size, max_target, self.continuous_latent_size)
        )
        generated_lengths = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=text_prompt.device,
        )
        finished = speech_target_lengths <= 0

        special_discrete_token_ids = (
            self._infer_special_discrete_token_ids(discrete_eos_id)
            if force_silent_special_tokens
            else tuple()
        )
        collected_logits: list[torch.Tensor] = []

        while True:
            can_generate = (~finished) & (generated_lengths < speech_target_lengths)
            if not bool(can_generate.any().item()):
                break

            current_target_lengths = generated_lengths + can_generate.to(dtype=torch.long)
            max_current_target = int(current_target_lengths.max().item())
            if max_current_target <= 0:
                break

            current_discrete = discrete_prompt.new_full(
                (batch_size, max_current_target, self.num_discrete_tokens),
                self.pad_token_id,
            )
            current_continuous = continuous_prompt.new_zeros(
                (batch_size, max_current_target, self.continuous_latent_size)
            )
            masked_blocks = torch.zeros(
                (batch_size, max_current_target),
                dtype=torch.bool,
                device=text_prompt.device,
            )

            for sample_idx in range(batch_size):
                generated_count = int(generated_lengths[sample_idx].item())
                if generated_count > 0:
                    current_discrete[sample_idx, :generated_count, :] = predicted_discrete[
                        sample_idx, :generated_count, :
                    ]
                    current_continuous[sample_idx, :generated_count, :] = predicted_continuous[
                        sample_idx, :generated_count, :
                    ]
                if bool(can_generate[sample_idx].item()):
                    masked_blocks[sample_idx, generated_count] = True

            flat = self._assemble_flat_batch(
                text_prompt=text_prompt,
                discrete_prompt=discrete_prompt,
                continuous_prompt=continuous_prompt,
                text_target=text_target,
                discrete_target=current_discrete,
                continuous_target=current_continuous,
                text_prompt_lengths=text_prompt_lengths,
                speech_prompt_lengths=speech_prompt_lengths,
                text_target_lengths=text_target_lengths,
                speech_target_lengths=current_target_lengths,
                attention_mask=None,
            )
            hidden_states, masked_discrete_positions, masked_continuous_positions, _ = self._encode(
                flat=flat,
                masked_target_blocks=masked_blocks,
            )
            batch_indices = (
                torch.arange(batch_size, device=text_prompt.device)
                .unsqueeze(1)
                .expand(batch_size, hidden_states.shape[1])
            )

            step_logits = torch.full(
                (batch_size, self.num_discrete_tokens, self.discrete_vocab_size),
                fill_value=float("-inf"),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
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
                step_logits[discrete_batch_idx, discrete_stream_idx, :] = discrete_logits

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
                    special_step_mask = self._build_special_block_mask(
                        discrete_tokens=step_discrete,
                        special_token_ids=special_discrete_token_ids,
                    )
                    if special_step_mask.any():
                        predicted_continuous[
                            continuous_batch_idx[special_step_mask],
                            continuous_block_idx[special_step_mask],
                            :,
                        ] = 0.0

            active_sample_idx = torch.nonzero(can_generate, as_tuple=False).squeeze(1)
            active_block_idx = generated_lengths[active_sample_idx]
            new_discrete_blocks = predicted_discrete[active_sample_idx, active_block_idx, :]
            reached_eos = torch.eq(new_discrete_blocks, discrete_eos_id).all(dim=-1)

            generated_lengths[active_sample_idx] = active_block_idx + 1
            finished[active_sample_idx] = reached_eos | (
                generated_lengths[active_sample_idx] >= speech_target_lengths[active_sample_idx]
            )
            collected_logits.append(step_logits)

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
    def generate_from_raw_inputs(
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

        return self.generate(
            text_prompt=text_prompt,
            discrete_prompt=discrete_prompt,
            continuous_prompt=continuous_prompt,
            text_target=text_target,
            text_prompt_lengths=text_prompt_lengths,
            speech_prompt_lengths=speech_prompt_lengths,
            text_target_lengths=text_target_lengths,
            return_dict=return_dict,
            **generate_kwargs,
        )
