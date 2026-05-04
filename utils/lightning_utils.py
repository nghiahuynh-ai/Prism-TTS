from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from utils.model_utils import PrismTTSGenerationOutput


@dataclass
class PrismBatch:
    text_prompt_lengths: Optional[torch.LongTensor] = None
    speech_prompt_lengths: Optional[torch.LongTensor] = None
    text_target_lengths: Optional[torch.LongTensor] = None
    speech_target_lengths: Optional[torch.LongTensor] = None
    text_target: Optional[torch.LongTensor] = None
    discrete_target: Optional[torch.LongTensor] = None
    continuous_target: Optional[torch.FloatTensor] = None
    text_prompt: Optional[torch.LongTensor] = None
    discrete_prompt: Optional[torch.LongTensor] = None
    continuous_prompt: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None
    flat_token_ids: Optional[torch.LongTensor] = None
    flat_continuous_values: Optional[torch.FloatTensor] = None
    flat_token_type_ids: Optional[torch.LongTensor] = None
    flat_speech_stream_ids: Optional[torch.LongTensor] = None
    flat_target_block_ids: Optional[torch.LongTensor] = None
    flat_target_block_counts: Optional[torch.LongTensor] = None
    flow_timesteps: Optional[torch.FloatTensor] = None
    noise: Optional[torch.FloatTensor] = None


@dataclass
class PeriodicEvalSample:
    sample_source: str
    batch_inputs: PrismBatch
    generations: Mapping[str, PrismTTSGenerationOutput]
    text_prompt: str
    text_target: str
    synthesized_text_by_method: Mapping[str, str]


def batch_size_from_prism_batch(batch_inputs: Any) -> int:
    """Infer batch size from a PrismBatch-like object."""
    for tensor in (
        getattr(batch_inputs, "text_target", None),
        getattr(batch_inputs, "text_prompt", None),
        getattr(batch_inputs, "flat_token_ids", None),
    ):
        if tensor is not None:
            return int(tensor.shape[0])
    raise ValueError("Unable to infer batch size from PrismBatch.")


def max_target_length_from_prism_batch(batch_inputs: Any) -> int:
    """Infer max target speech length from a PrismBatch-like object."""
    speech_target_lengths = getattr(batch_inputs, "speech_target_lengths", None)
    if speech_target_lengths is not None:
        return int(torch.as_tensor(speech_target_lengths).max().item())

    flat_target_block_counts = getattr(batch_inputs, "flat_target_block_counts", None)
    if flat_target_block_counts is not None:
        return int(torch.as_tensor(flat_target_block_counts).max().item())

    discrete_target = getattr(batch_inputs, "discrete_target", None)
    if discrete_target is not None:
        normalized = torch.as_tensor(discrete_target)
        if normalized.dim() >= 2:
            return int(normalized.shape[-2])

    text_target = getattr(batch_inputs, "text_target", None)
    if text_target is not None:
        return int(text_target.shape[1])
    return 0


def slice_batch_to_sample(value: Any, sample_idx: int = 0) -> Any:
    """Slice nested batch structures down to a single sample while preserving batch dim."""
    if torch.is_tensor(value):
        if value.dim() == 0 or int(value.shape[0]) < 1:
            return value
        sample_idx = max(0, min(int(sample_idx), int(value.shape[0]) - 1))
        return value[sample_idx : sample_idx + 1]
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or int(value.shape[0]) < 1:
            return value
        sample_idx = max(0, min(int(sample_idx), int(value.shape[0]) - 1))
        return value[sample_idx : sample_idx + 1]
    if isinstance(value, Mapping):
        return {
            key: slice_batch_to_sample(item, sample_idx=sample_idx)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(slice_batch_to_sample(item, sample_idx=sample_idx) for item in value)
    if isinstance(value, list):
        return [slice_batch_to_sample(item, sample_idx=sample_idx) for item in value]
    return value


def move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move nested tensor containers to a torch device."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    return value


def extract_tokenizer_from_loader(value: Any) -> Optional[Any]:
    """Find a tokenizer field by recursively traversing nested loader structures."""
    if value is None:
        return None

    dataset = getattr(value, "dataset", None)
    if dataset is not None:
        tokenizer = getattr(dataset, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer

    for attr_name in ("iterables", "loaders"):
        nested = getattr(value, attr_name, None)
        tokenizer = extract_tokenizer_from_loader(nested)
        if tokenizer is not None:
            return tokenizer

    if isinstance(value, Mapping):
        for nested in value.values():
            tokenizer = extract_tokenizer_from_loader(nested)
            if tokenizer is not None:
                return tokenizer
        return None

    if isinstance(value, (list, tuple)):
        for nested in value:
            tokenizer = extract_tokenizer_from_loader(nested)
            if tokenizer is not None:
                return tokenizer
        return None

    return None


def decode_audio(
    representation: torch.Tensor,
    *,
    audio_decoder: Optional[Any],
) -> Optional[np.ndarray]:
    """Decode speech representation into normalized mono waveform."""
    if audio_decoder is None:
        return None

    try:
        decoded = audio_decoder(representation.detach())
    except Exception:
        return None

    if isinstance(decoded, torch.Tensor):
        decoded = decoded.detach().cpu().float().numpy()
    else:
        decoded = np.asarray(decoded, dtype=np.float32)

    decoded = np.squeeze(decoded)
    if decoded.ndim > 1:
        decoded = decoded.reshape(-1)
    if decoded.size == 0:
        return None

    decoded = np.nan_to_num(decoded, copy=False)
    peak = float(np.max(np.abs(decoded)))
    if peak > 0.0:
        decoded = decoded / peak
    return decoded.astype(np.float32)


def compute_log_mel_spectrogram(
    audio: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int = 1024,
    win_length: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
) -> Optional[np.ndarray]:
    """Compute a log-mel spectrogram image array from waveform audio."""
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size < 2:
        return None

    waveform_tensor = torch.from_numpy(waveform)
    window = torch.hann_window(win_length, dtype=torch.float32, device=waveform_tensor.device)
    stft = torch.stft(
        waveform_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True,
    )
    power_spec = stft.abs().pow(2.0)
    if power_spec.numel() == 0:
        return None

    mel_filter = build_mel_filter_bank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=0.0,
        f_max=float(sample_rate) * 0.5,
        device=power_spec.device,
    )
    mel_power = mel_filter @ power_spec
    log_mel = torch.log10(mel_power.clamp_min(1e-10))
    return log_mel.detach().cpu().numpy()


def build_mel_filter_bank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
    device: torch.device,
) -> torch.Tensor:
    """Build a triangular mel filter bank matrix."""
    if sample_rate < 1:
        raise ValueError("sample_rate must be >= 1 for mel filter bank construction.")
    if n_fft < 2:
        raise ValueError("n_fft must be >= 2 for mel filter bank construction.")
    if n_mels < 1:
        raise ValueError("n_mels must be >= 1 for mel filter bank construction.")

    nyquist = float(sample_rate) * 0.5
    f_min = max(0.0, float(f_min))
    f_max = min(max(f_min + 1e-6, float(f_max)), nyquist)
    n_freqs = n_fft // 2 + 1

    def hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + (freq_hz / 700.0))

    def mel_to_hz(freq_mel: np.ndarray) -> np.ndarray:
        return 700.0 * (np.power(10.0, freq_mel / 2595.0) - 1.0)

    mel_points = np.linspace(
        hz_to_mel(np.array([f_min], dtype=np.float32))[0],
        hz_to_mel(np.array([f_max], dtype=np.float32))[0],
        num=n_mels + 2,
        dtype=np.float32,
    )
    hz_points = mel_to_hz(mel_points)
    fft_bins = np.floor(((n_fft + 1) * hz_points) / float(sample_rate)).astype(np.int64)
    fft_bins = np.clip(fft_bins, 0, n_freqs - 1)

    filters = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for mel_idx in range(1, n_mels + 1):
        left = int(fft_bins[mel_idx - 1])
        center = int(fft_bins[mel_idx])
        right = int(fft_bins[mel_idx + 1])

        if center <= left:
            center = min(left + 1, n_freqs - 1)
        if right <= center:
            right = min(center + 1, n_freqs)

        if center > left:
            filters[mel_idx - 1, left:center] = np.linspace(
                0.0,
                1.0,
                num=center - left,
                endpoint=False,
                dtype=np.float32,
            )
        if right > center:
            filters[mel_idx - 1, center:right] = np.linspace(
                1.0,
                0.0,
                num=right - center,
                endpoint=False,
                dtype=np.float32,
            )

    return torch.tensor(filters, dtype=torch.float32, device=device)
