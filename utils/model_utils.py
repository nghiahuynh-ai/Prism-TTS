from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
from transformers.utils import ModelOutput

import torch


TEXT_TOKEN_TYPE = 0
SPEECH_DISCRETE_TOKEN_TYPE = 1
SPEECH_CONTINUOUS_TOKEN_TYPE = 2


@dataclass
class FlatBatch:
    token_ids: torch.LongTensor
    continuous_values: torch.FloatTensor
    token_type_ids: torch.LongTensor
    speech_stream_ids: torch.LongTensor
    target_block_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
    target_block_counts: torch.LongTensor


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


def normalize_text_tokens(
    text_tokens: torch.LongTensor,
    name: str,
) -> torch.LongTensor:
    """Validate text token tensor shape as [batch, length]."""
    if text_tokens.dim() != 2:
        raise ValueError(f"{name} must have shape [batch, length].")
    return text_tokens


def normalize_discrete_tokens(
    discrete_tokens: torch.LongTensor,
    name: str,
    *,
    num_discrete_tokens: int,
) -> torch.LongTensor:
    """Normalize discrete tokens to shape [batch, length, num_discrete_tokens]."""
    if discrete_tokens.dim() != 3:
        raise ValueError(
            f"{name} must have shape [batch, num_discrete_tokens, length] "
            f"or [batch, length, num_discrete_tokens]."
        )
    if discrete_tokens.shape[1] == num_discrete_tokens:
        return discrete_tokens.transpose(1, 2).contiguous()
    if discrete_tokens.shape[-1] == num_discrete_tokens:
        return discrete_tokens
    raise ValueError(
        f"{name} must contain one axis with size num_discrete_tokens="
        f"{num_discrete_tokens}, got shape={tuple(discrete_tokens.shape)}."
    )


def normalize_continuous_latents(
    continuous_latents: torch.FloatTensor,
    expected_len: int,
    name: str,
    *,
    continuous_latent_size: int,
) -> torch.FloatTensor:
    """Validate/reshape continuous latents to [batch, length, continuous_latent_size]."""
    if continuous_latents.dim() == 2:
        if continuous_latent_size != 1:
            raise ValueError(
                f"{name} with shape [batch, length] is only valid when "
                "continuous_latent_size == 1."
            )
        continuous_latents = continuous_latents.unsqueeze(-1)
    elif continuous_latents.dim() != 3:
        raise ValueError(
            f"{name} must have shape [batch, length] or "
            f"[batch, length, {continuous_latent_size}]."
        )

    if continuous_latents.shape[1] != expected_len:
        raise ValueError(
            f"{name} length mismatch: expected {expected_len}, got "
            f"{continuous_latents.shape[1]}."
        )
    if continuous_latents.shape[-1] != continuous_latent_size:
        raise ValueError(
            f"{name} channel mismatch: expected {continuous_latent_size}, "
            f"got {continuous_latents.shape[-1]}."
        )
    return continuous_latents


def normalize_lengths(
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


def assemble_flat_batch(
    *,
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
    pad_token_id: int,
    eos_token_id: int,
    eot_token_id: int,
    continuous_latent_size: int,
    num_discrete_tokens: int,
) -> FlatBatch:
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
                    continuous_latent_size,
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
                    continuous_latent_size,
                    dtype=cont_dtype,
                    device=device,
                )
            )

        def append_speech_continuous(
            value: torch.Tensor,
            target_block_id: int,
        ) -> None:
            sample_token_ids.append(pad_token_id)
            sample_types.append(SPEECH_CONTINUOUS_TOKEN_TYPE)
            sample_stream_ids.append(num_discrete_tokens)
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
        append_text_token(eot_token_id)

        for block_idx in range(l2):
            for stream_idx in range(num_discrete_tokens):
                append_speech_discrete(
                    token_id=int(prompt_discrete[block_idx, stream_idx].item()),
                    stream_id=stream_idx,
                    target_block_id=-1,
                )
            append_speech_continuous(
                value=prompt_continuous[block_idx],
                target_block_id=-1,
            )
        append_text_token(eos_token_id)

        for token in target_text.tolist():
            append_text_token(token)
        append_text_token(eot_token_id)

        for block_idx in range(l4):
            for stream_idx in range(num_discrete_tokens):
                append_speech_discrete(
                    token_id=int(target_discrete[block_idx, stream_idx].item()),
                    stream_id=stream_idx,
                    target_block_id=block_idx,
                )
            append_speech_continuous(
                value=target_continuous[block_idx],
                target_block_id=block_idx,
            )
        append_text_token(eos_token_id)

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
        pad_token_id,
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
        continuous_latent_size,
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

    return FlatBatch(
        token_ids=token_ids,
        continuous_values=continuous_values,
        token_type_ids=token_type_ids,
        speech_stream_ids=speech_stream_ids,
        target_block_ids=target_block_ids,
        attention_mask=derived_attention_mask,
        target_block_counts=speech_target_lengths.to(device=device, dtype=torch.long),
    )


def sample_masked_target_blocks(
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


def build_flat_batch_from_collate(
    *,
    flat_token_ids: torch.LongTensor,
    flat_continuous_values: torch.FloatTensor,
    flat_token_type_ids: torch.LongTensor,
    flat_speech_stream_ids: torch.LongTensor,
    flat_target_block_ids: torch.LongTensor,
    flat_target_block_counts: Optional[torch.LongTensor],
    attention_mask: Optional[torch.Tensor],
    continuous_latent_size: int,
) -> FlatBatch:
    """Validate collate-produced flat tensors and pack them into `FlatBatch`."""
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
        expected_last=continuous_latent_size,
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

    return FlatBatch(
        token_ids=flat_token_ids.to(dtype=torch.long, device=device),
        continuous_values=flat_continuous_values.to(device=device),
        token_type_ids=flat_token_type_ids.to(dtype=torch.long, device=device),
        speech_stream_ids=flat_speech_stream_ids.to(dtype=torch.long, device=device),
        target_block_ids=flat_target_block_ids.to(dtype=torch.long, device=device),
        attention_mask=resolved_attention,
        target_block_counts=flat_target_block_counts,
    )


def build_two_level_rope_position_embeddings(
    *,
    inputs_embeds: torch.FloatTensor,
    speech_stream_ids: torch.LongTensor,
    rotary_emb: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose global 1D RoPE with within-block stream-index RoPE."""
    batch_size, seq_len, _ = inputs_embeds.shape
    device = inputs_embeds.device

    global_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    secondary_position_ids = speech_stream_ids.clamp(min=0)

    global_cos, global_sin = rotary_emb(inputs_embeds, position_ids=global_position_ids)
    secondary_cos, secondary_sin = rotary_emb(
        inputs_embeds,
        position_ids=secondary_position_ids,
    )

    cos = global_cos * secondary_cos - global_sin * secondary_sin
    sin = global_sin * secondary_cos + global_cos * secondary_sin
    return cos, sin


def top_k_top_p_filter(
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


def resolve_generation_discrete_eos_token_id(
    discrete_eos_token_id: Optional[int],
    *,
    backbone_eos_token_id: Optional[int],
    discrete_vocab_size: int,
) -> int:
    """Resolve the discrete EOS id used during generation."""
    if discrete_eos_token_id is not None:
        return int(discrete_eos_token_id)
    candidate = backbone_eos_token_id
    if candidate is None or not (0 <= int(candidate) < discrete_vocab_size):
        return 0
    return int(candidate)


def infer_special_discrete_token_ids(
    discrete_eos_token_id: int,
    *,
    backbone_eos_token_id: Optional[int],
    backbone_pad_token_id: Optional[int],
    discrete_vocab_size: int,
) -> tuple[int, ...]:
    """Collect discrete token ids treated as special for loss/silence heuristics."""
    ids: set[int] = set()
    if 0 <= discrete_eos_token_id < discrete_vocab_size:
        ids.add(int(discrete_eos_token_id))

    cfg_eos = backbone_eos_token_id
    cfg_pad = backbone_pad_token_id
    if cfg_eos is not None and 0 <= int(cfg_eos) < discrete_vocab_size:
        ids.add(int(cfg_eos))
        if 0 <= int(cfg_eos) - 1 < discrete_vocab_size:
            ids.add(int(cfg_eos) - 1)
    if cfg_pad is not None and 0 <= int(cfg_pad) < discrete_vocab_size:
        ids.add(int(cfg_pad))
    return tuple(sorted(ids))


def build_special_block_mask(
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


def estimate_parallel_speech_target_lengths(
    *,
    text_prompt_lengths: torch.LongTensor,
    speech_prompt_lengths: torch.LongTensor,
    text_target_lengths: torch.LongTensor,
    max_new_blocks: Optional[int],
) -> torch.LongTensor:
    """
    Estimate target speech blocks for parallel generation from prompt length ratios.

    The estimate mirrors the usual text-to-speech duration heuristic:
    target_blocks ~= target_text_len * (prompt_speech_len / prompt_text_len),
    with bounded ratio and per-sample safety floors.
    """
    batch_size = int(text_prompt_lengths.shape[0])
    device = text_prompt_lengths.device
    estimated = torch.zeros(batch_size, dtype=torch.long, device=device)
    cap = None if max_new_blocks is None else max(0, int(max_new_blocks))

    for sample_idx in range(batch_size):
        prompt_text_len = int(text_prompt_lengths[sample_idx].item())
        prompt_speech_len = int(speech_prompt_lengths[sample_idx].item())
        target_text_len = int(text_target_lengths[sample_idx].item())

        if target_text_len > 0:
            ratio = 1.0
            if prompt_text_len > 0:
                ratio = float(prompt_speech_len) / float(prompt_text_len)
            ratio = max(0.5, min(8.0, ratio))
            est_blocks = int(math.ceil(float(target_text_len) * ratio))
            est_blocks = max(target_text_len, est_blocks)
        elif cap is not None:
            est_blocks = cap
        else:
            est_blocks = max(1, prompt_speech_len)

        if cap is not None:
            est_blocks = min(est_blocks, cap)
        estimated[sample_idx] = max(0, int(est_blocks))

    return estimated
