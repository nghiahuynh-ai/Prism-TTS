from __future__ import annotations

import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers.utils import ModelOutput


# Token-type ids for the flattened autoregressive sequence.
# Speech is now modeled as a single continuous latent per frame (no vector
# quantization), so there is no separate discrete stream type.
TEXT_TOKEN_TYPE = 0
SPEECH_TOKEN_TYPE = 1


@dataclass
class FlatBatch:
    """Packed, causal, continuous-only sequence consumed by the AR backbone.

    Layout per sample:
    text_prompt -> EOT -> [prompt speech frames] -> EOS -> text_target -> EOT
      -> [target speech frames]

    Each speech frame occupies a single position carrying a continuous latent in
    `continuous_values` (raw, un-normalized). `target_block_ids` holds the target
    frame index (0-based) at target speech positions and -1 elsewhere.
    """

    token_ids: torch.LongTensor
    continuous_values: torch.FloatTensor
    token_type_ids: torch.LongTensor
    target_block_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
    target_block_counts: torch.LongTensor


@dataclass
class PrismTTSOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    consistency_loss: Optional[torch.Tensor] = None
    eos_loss: Optional[torch.Tensor] = None


@dataclass
class PrismTTSGenerationOutput(ModelOutput):
    text_ids: Optional[torch.LongTensor] = None
    continuous_latents: Optional[torch.FloatTensor] = None
    eos_scores: Optional[torch.FloatTensor] = None


def normalize_text_tokens(
    text_tokens: torch.LongTensor,
    name: str,
) -> torch.LongTensor:
    """Validate text token tensor shape as [batch, length]."""
    if text_tokens.dim() != 2:
        raise ValueError(f"{name} must have shape [batch, length].")
    return text_tokens


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


def inject_continuous_backbone_noise(
    clean_latents: torch.FloatTensor,
) -> torch.FloatTensor:
    """Inject random VP-schedule noise into continuous latents fed to the AR backbone.

    Per frame, interpolate between the clean latent and Gaussian noise at a random
    SNR (k ~ U(0, 1)): sqrt(k)*eps + sqrt(1-k)*clean. Applied only to the backbone
    *input context* during training; the per-frame head target stays clean. This
    regularizes the causal backbone against exposure bias, since at inference it
    consumes imperfect generated frames rather than ground-truth latents.
    """
    k = torch.rand(
        (*clean_latents.shape[:-1], 1),
        device=clean_latents.device,
        dtype=clean_latents.dtype,
    )
    e = torch.randn_like(clean_latents)
    return torch.sqrt(k) * e + torch.sqrt(1.0 - k) * clean_latents


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
    num_quantizers: int,
    device: torch.device,
    continuous_dtype: torch.dtype,
    mimi_model_name_or_path: str,
    mimi_revision: str,
    mimi_token: str | bool | None,
    mimi_local_files_only: bool,
    raw_prompt_name: str = "raw_speech_prompt",
) -> Callable[[Any], torch.Tensor]:
    """Build a Mimi-backed encoder returning continuous pre-upsample latents [L, D].

    Vector quantization is bypassed for modeling: the encoder still runs Mimi's
    RVQ to obtain codes, then reconstructs the continuous pre-upsample latent via
    `quantizer.decode`. `num_quantizers` controls the fidelity of that continuous
    reconstruction and should match how the training features were prepared.
    """
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

    def _default_speech_encoder(raw_prompt: Any) -> torch.Tensor:
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
            num_quantizers=int(num_quantizers),
            return_dict=True,
        )
        prompt_codes = encoded.audio_codes
        if prompt_codes is None:
            raise RuntimeError("Mimi encode did not return audio_codes.")
        prompt_latents = mimi_model.quantizer.decode(prompt_codes)

        continuous = prompt_latents[0].transpose(0, 1).to(dtype=continuous_dtype)
        return continuous

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
    continuous_prompt: torch.FloatTensor,
    text_target: torch.LongTensor,
    continuous_target: torch.FloatTensor,
    text_prompt_lengths: torch.LongTensor,
    speech_prompt_lengths: torch.LongTensor,
    text_target_lengths: torch.LongTensor,
    speech_target_lengths: torch.LongTensor,
    pad_token_id: int,
    eos_token_id: int,
    eot_token_id: int,
    continuous_latent_size: int,
) -> FlatBatch:
    """Assemble split prompt/target tensors into one causal continuous-only sequence.

    Layout: text_prompt -> EOT -> [prompt frames] -> EOS -> text_target -> EOT
      -> [target frames]. Target frames carry frame ids 0..L-1 in
    `target_block_ids`; everything else is -1. There is no trailing EOS after the
    target frames (end-of-sequence is predicted by the model's EOS head).
    """
    batch_size = int(text_prompt.shape[0])
    device = text_prompt.device
    cont_dtype = continuous_prompt.dtype

    token_ids_per_sample: list[torch.LongTensor] = []
    continuous_per_sample: list[torch.FloatTensor] = []
    token_types_per_sample: list[torch.LongTensor] = []
    target_block_ids_per_sample: list[torch.LongTensor] = []

    for sample_idx in range(batch_size):
        l1 = int(text_prompt_lengths[sample_idx].item())
        l2 = int(speech_prompt_lengths[sample_idx].item())
        l3 = int(text_target_lengths[sample_idx].item())
        l4 = int(speech_target_lengths[sample_idx].item())

        sample_token_ids: list[int] = []
        sample_types: list[int] = []
        sample_target_block_ids: list[int] = []
        sample_continuous: list[torch.Tensor] = []

        zero_cont = torch.zeros(continuous_latent_size, dtype=cont_dtype, device=device)

        def append_text(token_id: int) -> None:
            sample_token_ids.append(int(token_id))
            sample_types.append(TEXT_TOKEN_TYPE)
            sample_target_block_ids.append(-1)
            sample_continuous.append(zero_cont)

        def append_speech(value: torch.Tensor, target_block_id: int) -> None:
            sample_token_ids.append(pad_token_id)
            sample_types.append(SPEECH_TOKEN_TYPE)
            sample_target_block_ids.append(int(target_block_id))
            sample_continuous.append(value)

        for token in text_prompt[sample_idx, :l1].tolist():
            append_text(token)
        append_text(eot_token_id)

        for block_idx in range(l2):
            append_speech(continuous_prompt[sample_idx, block_idx], -1)
        append_text(eos_token_id)

        for token in text_target[sample_idx, :l3].tolist():
            append_text(token)
        append_text(eot_token_id)

        for block_idx in range(l4):
            append_speech(continuous_target[sample_idx, block_idx], block_idx)

        token_ids_per_sample.append(
            torch.tensor(sample_token_ids, dtype=torch.long, device=device)
        )
        token_types_per_sample.append(
            torch.tensor(sample_types, dtype=torch.long, device=device)
        )
        target_block_ids_per_sample.append(
            torch.tensor(sample_target_block_ids, dtype=torch.long, device=device)
        )
        continuous_per_sample.append(torch.stack(sample_continuous, dim=0))

    max_seq_len = max(int(x.shape[0]) for x in token_ids_per_sample)
    token_ids = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long, device=device)
    token_type_ids = torch.full(
        (batch_size, max_seq_len), TEXT_TOKEN_TYPE, dtype=torch.long, device=device
    )
    target_block_ids = torch.full((batch_size, max_seq_len), -1, dtype=torch.long, device=device)
    continuous_values = torch.zeros(
        batch_size, max_seq_len, continuous_latent_size, dtype=cont_dtype, device=device
    )
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=device)

    for sample_idx in range(batch_size):
        seq_len = int(token_ids_per_sample[sample_idx].shape[0])
        token_ids[sample_idx, :seq_len] = token_ids_per_sample[sample_idx]
        token_type_ids[sample_idx, :seq_len] = token_types_per_sample[sample_idx]
        target_block_ids[sample_idx, :seq_len] = target_block_ids_per_sample[sample_idx]
        continuous_values[sample_idx, :seq_len, :] = continuous_per_sample[sample_idx]
        attention_mask[sample_idx, :seq_len] = True

    return FlatBatch(
        token_ids=token_ids,
        continuous_values=continuous_values,
        token_type_ids=token_type_ids,
        target_block_ids=target_block_ids,
        attention_mask=attention_mask,
        target_block_counts=speech_target_lengths.to(device=device, dtype=torch.long),
    )


def build_flat_batch_from_collate(
    *,
    flat_token_ids: torch.LongTensor,
    flat_continuous_values: torch.FloatTensor,
    flat_token_type_ids: torch.LongTensor,
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
        target_block_ids=flat_target_block_ids.to(dtype=torch.long, device=device),
        attention_mask=resolved_attention,
        target_block_counts=flat_target_block_counts,
    )
