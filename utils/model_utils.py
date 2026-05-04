from __future__ import annotations

import math
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers.utils import ModelOutput


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
    prior_latents: Optional[torch.FloatTensor] = None
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


def normalize_raw_text_batch(
    value: str | Sequence[str],
    name: str,
) -> list[str]:
    """Normalize a raw text input to a non-empty list of strings."""
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


def read_wav_mono(path_like: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV file and return mono float32 audio in [-1, 1] and sample rate."""
    resolved = Path(path_like).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Audio file not found: {resolved}")

    with wave.open(str(resolved), "rb") as handle:
        channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        sample_rate = int(handle.getframerate())
        num_frames = int(handle.getnframes())
        raw = handle.readframes(num_frames)

    if num_frames <= 0:
        raise ValueError(f"Audio file is empty: {resolved}")

    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        sign_bit = 1 << 23
        values = (values ^ sign_bit) - sign_bit
        audio = values.astype(np.float32) / float(1 << 23)
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes.")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32, copy=False), sample_rate


def to_mono_audio_array(
    audio_like: Any,
    *,
    name: str,
) -> np.ndarray:
    """Convert arbitrary waveform-like input to finite mono float32 numpy array."""
    if torch.is_tensor(audio_like):
        array = audio_like.detach().to(dtype=torch.float32, device="cpu").numpy()
    else:
        array = np.asarray(audio_like, dtype=np.float32)
    if array.size <= 0:
        raise ValueError(f"{name} waveform must not be empty.")
    if array.ndim == 1:
        mono = array
    elif array.ndim == 2:
        axis = 0 if array.shape[0] <= array.shape[1] else 1
        mono = array.mean(axis=axis)
    else:
        mono = array.reshape(-1)
    mono = np.nan_to_num(mono.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    if mono.size <= 0:
        raise ValueError(f"{name} waveform must not be empty.")
    return mono


def extract_audio_with_sample_rate(
    raw_prompt: Any,
    *,
    name: str,
) -> tuple[np.ndarray, int | None]:
    """Parse raw prompt input into (mono_waveform, optional_sample_rate)."""
    if isinstance(raw_prompt, (str, Path)):
        return read_wav_mono(raw_prompt)
    if (
        isinstance(raw_prompt, tuple)
        and len(raw_prompt) == 2
        and isinstance(raw_prompt[1], (int, np.integer))
    ):
        waveform = to_mono_audio_array(raw_prompt[0], name=name)
        sample_rate = int(raw_prompt[1])
        if sample_rate <= 0:
            raise ValueError(f"{name} sample rate must be > 0.")
        return waveform, sample_rate
    return to_mono_audio_array(raw_prompt, name=name), None


def resample_audio_if_needed(
    waveform: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Resample mono waveform with linear interpolation when rates differ."""
    if source_rate == target_rate:
        return waveform
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Resampling rates must be > 0.")
    src = torch.from_numpy(waveform).view(1, 1, -1)
    target_length = max(1, int(round(float(waveform.shape[0]) * target_rate / source_rate)))
    dst = F.interpolate(
        src,
        size=target_length,
        mode="linear",
        align_corners=False,
    )
    return dst[0, 0].cpu().numpy().astype(np.float32, copy=False)


def build_default_mimi_speech_encoder(
    *,
    num_discrete_tokens: int,
    device: torch.device,
    continuous_dtype: torch.dtype,
    mimi_model_name_or_path: str,
    mimi_revision: str,
    mimi_token: str | bool | None,
    mimi_local_files_only: bool,
    raw_prompt_name: str = "raw_speech_prompt",
) -> Callable[[Any], tuple[torch.Tensor, torch.Tensor]]:
    """Build a Mimi-backed speech encoder returning (discrete_codes, pre-upsample_latents)."""
    try:
        from transformers import AutoFeatureExtractor, MimiModel
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Default generate_e2e Mimi encoder requires transformers with Mimi support."
        ) from exc

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        mimi_model_name_or_path,
        revision=mimi_revision,
        token=mimi_token,
        local_files_only=bool(mimi_local_files_only),
    )
    mimi_sample_rate = int(getattr(feature_extractor, "sampling_rate", 24_000))
    mimi_model = MimiModel.from_pretrained(
        mimi_model_name_or_path,
        revision=mimi_revision,
        token=mimi_token,
        local_files_only=bool(mimi_local_files_only),
    )
    mimi_model.to(device=device)
    mimi_model.eval()

    def _default_speech_encoder(raw_prompt: Any) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_audio, prompt_sample_rate = extract_audio_with_sample_rate(
            raw_prompt,
            name=raw_prompt_name,
        )
        if prompt_sample_rate is None:
            prompt_sample_rate = mimi_sample_rate
        if prompt_sample_rate != mimi_sample_rate:
            prompt_audio = resample_audio_if_needed(
                prompt_audio,
                source_rate=prompt_sample_rate,
                target_rate=mimi_sample_rate,
            )

        features = feature_extractor(
            raw_audio=prompt_audio,
            sampling_rate=mimi_sample_rate,
            return_tensors="pt",
        )
        input_values = features["input_values"]
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)
        if input_values.dim() != 3:
            raise ValueError(f"Unexpected Mimi input shape: {tuple(input_values.shape)}")
        input_values = input_values.to(device=device)

        padding_mask = features.get("padding_mask")
        if padding_mask is not None:
            if padding_mask.dim() == 2:
                padding_mask = padding_mask.unsqueeze(1)
            if padding_mask.dim() != 3:
                raise ValueError(f"Unexpected Mimi padding_mask shape: {tuple(padding_mask.shape)}")
            padding_mask = padding_mask.to(device=device)

        encoded = mimi_model.encode(
            input_values=input_values,
            padding_mask=padding_mask,
            num_quantizers=int(num_discrete_tokens),
            return_dict=True,
        )
        prompt_codes = encoded.audio_codes
        if prompt_codes is None:
            raise RuntimeError("Mimi encode did not return audio_codes.")
        prompt_latents = mimi_model.quantizer.decode(prompt_codes)

        discrete = prompt_codes[0].transpose(0, 1).to(dtype=torch.long)
        continuous = prompt_latents[0].transpose(0, 1).to(dtype=continuous_dtype)
        return discrete, continuous

    return _default_speech_encoder


def build_lazy_mimi_speech_decoder(
    *,
    device: torch.device,
    continuous_dtype: torch.dtype,
    mimi_model_name_or_path: str,
    mimi_revision: str,
    mimi_token: str | bool | None,
    mimi_local_files_only: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a lazily initialized Mimi decoder from pre-upsample latents to waveform."""
    decoder_state: dict[str, Any] = {}

    def _default_speech_decoder(latents: torch.Tensor) -> torch.Tensor:
        decoder = decoder_state.get("decoder")
        if decoder is None:
            from models.mimi_latent_decoder import MimiPreUpsampleLatentDecoder

            decoder = MimiPreUpsampleLatentDecoder(
                pretrained_model_name_or_path=mimi_model_name_or_path,
                device=str(device),
                dtype=continuous_dtype,
                local_files_only=bool(mimi_local_files_only),
                revision=mimi_revision,
                token=mimi_token,
            )
            decoder_state["decoder"] = decoder
        return decoder(latents)

    return _default_speech_decoder


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
    include_continuous_stream: bool = True,
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
            if include_continuous_stream:
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
            if include_continuous_stream:
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


def sample_discrete_ids(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> torch.LongTensor:
    """Sample token ids from logits with optional temperature/top-k/top-p filtering."""
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


def inject_continuous_backbone_noise(
    clean_latents: torch.FloatTensor,
) -> torch.FloatTensor:
    """Inject random noise into continuous latents before backbone conditioning."""
    k = torch.rand(
        (*clean_latents.shape[:-1], 1),
        device=clean_latents.device,
        dtype=clean_latents.dtype,
    )
    e = torch.randn_like(clean_latents)
    return torch.sqrt(k) * e + torch.sqrt(1.0 - k) * clean_latents


def sample_flow_training_inputs(
    continuous_targets: torch.FloatTensor,
    flow_timesteps: Optional[torch.FloatTensor] = None,
    noise: Optional[torch.FloatTensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample flow-matching mixtures and denoising targets for continuous latents."""
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


def get_time_shifted_steps(
    *,
    num_steps: int,
    t_shift: float,
    device: torch.device,
) -> torch.FloatTensor:
    """Build [0, 1] timesteps with a time-shift transform."""
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1.")
    if t_shift <= 0.0:
        raise ValueError("t_shift must be > 0.")
    base = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=torch.float32)
    return t_shift * base / (1.0 + (t_shift - 1.0) * base)


def build_parallel_stable_unmask_schedule(
    *,
    maskable_lengths: torch.LongTensor,
    num_steps: int,
    t_shift: float,
) -> torch.LongTensor:
    """
    Build per-sample unmask counts for stable parallel decoding.

    The schedule mirrors iterative unmasking:
    k_n = r_n - r_{n-1} with time-shifted cumulative ratio r_n.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1.")

    batch_size = int(maskable_lengths.shape[0])
    device = maskable_lengths.device
    schedule = torch.zeros(batch_size, num_steps, dtype=torch.long, device=device)
    if batch_size <= 0:
        return schedule

    time_steps = get_time_shifted_steps(
        num_steps=num_steps,
        t_shift=t_shift,
        device=device,
    )
    time_deltas = time_steps[1:] - time_steps[:-1]

    for sample_idx in range(batch_size):
        total_maskable = max(0, int(maskable_lengths[sample_idx].item()))
        if total_maskable <= 0:
            continue

        remaining = total_maskable
        for step_idx in range(num_steps):
            if remaining <= 0:
                break
            if step_idx >= num_steps - 1:
                step_unmask = remaining
            else:
                delta = float(time_deltas[step_idx].item())
                step_unmask = int(math.ceil(float(total_maskable) * delta))
                step_unmask = min(remaining, max(0, step_unmask))
            schedule[sample_idx, step_idx] = int(step_unmask)
            remaining -= int(step_unmask)

    return schedule


def gumbel_sample_scores(
    scores: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Apply temperature-scaled Gumbel perturbation for position selection."""
    if temperature <= 0.0:
        return scores
    scaled_scores = scores / temperature
    uniform = torch.rand_like(scaled_scores)
    uniform = uniform.clamp(min=1e-10, max=1.0 - 1e-10)
    gumbel_noise = -torch.log(-torch.log(uniform))
    return scaled_scores + gumbel_noise


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
