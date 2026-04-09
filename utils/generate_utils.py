from __future__ import annotations

import math
import warnings
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LlamaConfig

from dataset.dataset import SharedVocabTokenizer
from models.prism_tts import PrismTTS

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ImportError("generate.py requires PyYAML (`pip install pyyaml`).") from exc


__all__ = [
    "align_prompt_streams",
    "build_model",
    "build_teacher_forcing_target_text",
    "discrete_quality_score",
    "estimate_max_new_blocks",
    "is_collapsed_discrete_stats",
    "load_checkpoint",
    "read_wav",
    "read_yaml",
    "require_mapping",
    "resample_if_needed",
    "resolve_device",
    "resolve_shared_delay_tokens",
    "resolve_torch_dtype",
    "safe_tokenize",
    "save_mel_spectrogram_plot",
    "summarize_discrete_generation",
    "trim_latent_tail_from_special_blocks",
    "write_wav",
]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    normalized = dtype.strip().lower()
    mapping: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    resolved = mapping.get(normalized)
    if resolved is None:
        raise ValueError(f"Unsupported dtype {dtype!r}.")
    return resolved


def read_yaml(path: Path) -> dict[str, Any]:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML at {resolved}, got {type(payload).__name__}.")
    return payload


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required mapping '{key}'.")
    return value


def build_model(model_config: dict[str, Any]) -> PrismTTS:
    model_cfg = require_mapping(model_config, "model")
    prism_cfg = require_mapping(model_cfg, "prism_tts")
    llama_cfg = dict(require_mapping(model_cfg, "llama_config"))

    llama_config = LlamaConfig(**llama_cfg)
    return PrismTTS(
        llama_config=llama_config,
        num_discrete_tokens=int(prism_cfg["num_discrete_tokens"]),
        discrete_vocab_size=int(prism_cfg["discrete_vocab_size"]),
        continuous_latent_size=int(prism_cfg["continuous_latent_size"]),
        flow_num_res_blocks=int(prism_cfg.get("flow_num_res_blocks", 4)),
        flow_model_channels=prism_cfg.get("flow_model_channels"),
        flow_loss_weight=float(prism_cfg.get("flow_loss_weight", 1.0)),
        text_loss_weight=float(prism_cfg.get("text_loss_weight", 0.1)),
        flow_sample_steps=int(prism_cfg.get("flow_sample_steps", 16)),
    )


def _extract_model_state_dict(
    checkpoint_payload: dict[str, Any],
    *,
    use_ema: bool,
) -> dict[str, torch.Tensor]:
    if use_ema:
        ema_state = checkpoint_payload.get("ema_state")
        if isinstance(ema_state, dict) and ema_state:
            return dict(ema_state)

    state_dict = checkpoint_payload.get("state_dict")
    if isinstance(state_dict, dict) and state_dict:
        stripped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                stripped[key[len("model.") :]] = value
            else:
                stripped[key] = value
        return stripped

    if checkpoint_payload and all(isinstance(key, str) for key in checkpoint_payload):
        if all(torch.is_tensor(value) for value in checkpoint_payload.values()):
            return dict(checkpoint_payload)

    raise ValueError("Unable to find a model state dict in checkpoint payload.")


def load_checkpoint(
    model: PrismTTS,
    checkpoint_path: Path,
    *,
    use_ema: bool,
) -> None:
    resolved = checkpoint_path.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")

    payload = torch.load(resolved, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload).__name__}.")

    state_dict = _extract_model_state_dict(payload, use_ema=use_ema)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        missing_preview = ", ".join(missing[:10])
        raise RuntimeError(f"Missing model keys ({len(missing)}): {missing_preview}")
    if unexpected:
        unexpected_preview = ", ".join(unexpected[:10])
        raise RuntimeError(f"Unexpected model keys ({len(unexpected)}): {unexpected_preview}")


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    resolved = path.expanduser()
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


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample_rate: {sample_rate}")

    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("Cannot write empty waveform.")
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = np.round(audio * 32767.0).astype(np.int16)

    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(resolved), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _build_mel_filter_bank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
    device: torch.device,
) -> torch.Tensor:
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


def _compute_log_mel_spectrogram(
    audio: np.ndarray,
    *,
    sample_rate: int,
) -> np.ndarray | None:
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size < 2:
        return None

    waveform_tensor = torch.from_numpy(waveform)
    n_fft = 1024
    win_length = 1024
    hop_length = 256
    n_mels = 80

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

    mel_filter = _build_mel_filter_bank(
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


def save_mel_spectrogram_plot(
    *,
    waveform: np.ndarray,
    sample_rate: int,
    output_audio_path: Path,
) -> Path | None:
    log_mel = _compute_log_mel_spectrogram(waveform, sample_rate=sample_rate)
    if log_mel is None:
        return None

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        warnings.warn(
            "matplotlib is not installed; skipping mel spectrogram plot save.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    mel_path = output_audio_path.parent / f"{output_audio_path.stem}_mel.png"
    fig, ax = plt.subplots(figsize=(8.0, 3.2), dpi=150, constrained_layout=True)
    image = ax.imshow(
        log_mel,
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    ax.set_title("Synthesized Mel Spectrogram", fontsize=11, weight="bold")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel Bin")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(mel_path)
    plt.close(fig)
    return mel_path


def resample_if_needed(
    waveform: np.ndarray,
    src_sample_rate: int,
    dst_sample_rate: int,
) -> np.ndarray:
    if src_sample_rate == dst_sample_rate:
        return waveform
    if src_sample_rate <= 0 or dst_sample_rate <= 0:
        raise ValueError("Sample rates must be positive for resampling.")

    source = torch.as_tensor(waveform, dtype=torch.float32).view(1, 1, -1)
    target_length = max(1, int(round(source.shape[-1] * float(dst_sample_rate) / src_sample_rate)))
    resampled = F.interpolate(
        source,
        size=target_length,
        mode="linear",
        align_corners=False,
    )
    return resampled[0, 0].cpu().numpy().astype(np.float32, copy=False)


def resolve_shared_delay_tokens(collate_cfg: dict[str, Any]) -> int:
    stream_delay = collate_cfg.get("stream_delay")
    delay_ms = collate_cfg.get("discrete_stream_delay_ms")
    frame_rate_hz = float(collate_cfg.get("codec_frame_rate_hz", 12.5))

    if stream_delay is not None and delay_ms is not None:
        raise ValueError(
            "data.collate.stream_delay and data.collate.discrete_stream_delay_ms cannot both be set."
        )
    if frame_rate_hz <= 0:
        raise ValueError("data.collate.codec_frame_rate_hz must be > 0.")
    if stream_delay is not None:
        value = int(stream_delay)
        if value < 0:
            raise ValueError("data.collate.stream_delay must be non-negative.")
        return value
    if delay_ms is not None:
        delay_value = float(delay_ms)
        if delay_value < 0:
            raise ValueError("data.collate.discrete_stream_delay_ms must be non-negative.")
        return int(math.ceil((delay_value * frame_rate_hz) / 1000.0))
    return 0


def _pad_1d(tensor: torch.Tensor, length: int, pad_value: int) -> torch.Tensor:
    if tensor.dim() != 1:
        raise ValueError(f"Expected 1D tensor, got {tuple(tensor.shape)}.")
    if length < tensor.shape[0]:
        raise ValueError("Cannot pad to shorter length.")
    out = torch.full((length,), pad_value, dtype=tensor.dtype)
    out[: tensor.shape[0]] = tensor
    return out


def _pad_2d(tensor: torch.Tensor, length: int, pad_value: int | float) -> torch.Tensor:
    if tensor.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {tuple(tensor.shape)}.")
    if length < tensor.shape[0]:
        raise ValueError("Cannot pad to shorter length.")
    out = torch.full((length, tensor.shape[1]), pad_value, dtype=tensor.dtype)
    out[: tensor.shape[0], :] = tensor
    return out


def align_prompt_streams(
    *,
    text_tokens: torch.LongTensor,
    discrete_tokens: torch.LongTensor,
    continuous_latents: torch.FloatTensor,
    delay_token_id: int,
    eos_token_id: int,
    text_pad_value: int,
    discrete_pad_value: int,
    continuous_pad_value: float,
    shared_delay_tokens: int,
) -> tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
    if text_tokens.dim() != 1:
        raise ValueError(f"text_tokens must be 1D, got {tuple(text_tokens.shape)}.")
    if discrete_tokens.dim() != 2:
        raise ValueError(f"discrete_tokens must be 2D [L, N], got {tuple(discrete_tokens.shape)}.")
    if continuous_latents.dim() != 2:
        raise ValueError(
            f"continuous_latents must be 2D [L, C], got {tuple(continuous_latents.shape)}."
        )
    if discrete_tokens.shape[0] != continuous_latents.shape[0]:
        raise ValueError(
            "discrete_tokens and continuous_latents length mismatch: "
            f"{discrete_tokens.shape[0]} vs {continuous_latents.shape[0]}"
        )

    text_length = int(text_tokens.shape[0])
    discrete_length = int(discrete_tokens.shape[0])
    extra_delay = max(0, text_length - discrete_length - shared_delay_tokens + 1)
    effective_delay = shared_delay_tokens + extra_delay

    text_eos = torch.tensor([eos_token_id], dtype=torch.long)
    text_stream = torch.cat([text_tokens.to(dtype=torch.long), text_eos], dim=0)

    discrete_streams: list[torch.Tensor] = []
    for stream_idx in range(int(discrete_tokens.shape[1])):
        prefix = torch.full((effective_delay,), delay_token_id, dtype=torch.long)
        eos = torch.tensor([eos_token_id], dtype=torch.long)
        stream = torch.cat([prefix, discrete_tokens[:, stream_idx].to(dtype=torch.long), eos], dim=0)
        discrete_streams.append(stream)

    continuous_prefix = torch.zeros(
        (effective_delay, int(continuous_latents.shape[1])),
        dtype=torch.float32,
    )
    continuous_eos = torch.zeros((1, int(continuous_latents.shape[1])), dtype=torch.float32)
    continuous_stream = torch.cat(
        [continuous_prefix, continuous_latents.to(dtype=torch.float32), continuous_eos],
        dim=0,
    )

    max_length = max(
        int(text_stream.shape[0]),
        int(continuous_stream.shape[0]),
        max(int(stream.shape[0]) for stream in discrete_streams) if discrete_streams else 0,
    )

    text_aligned = _pad_1d(text_stream, max_length, text_pad_value)
    discrete_aligned = torch.full(
        (max_length, len(discrete_streams)),
        discrete_pad_value,
        dtype=torch.long,
    )
    for stream_idx, stream in enumerate(discrete_streams):
        discrete_aligned[: stream.shape[0], stream_idx] = stream
    continuous_aligned = _pad_2d(continuous_stream, max_length, continuous_pad_value).to(
        dtype=torch.float32
    )

    return text_aligned, discrete_aligned, continuous_aligned


def estimate_max_new_blocks(
    *,
    target_text_token_count: int,
    prompt_text_token_count: int,
    prompt_frame_count: int,
    shared_delay_tokens: int,
    duration_scale: float,
    trailing_pad_blocks: int,
) -> int:
    if target_text_token_count < 1:
        raise ValueError("target_text_token_count must be >= 1.")
    if duration_scale <= 0.0:
        raise ValueError("duration_scale must be > 0.")

    ratio = 1.0
    if prompt_text_token_count > 0:
        ratio = float(prompt_frame_count) / float(prompt_text_token_count)
    ratio = max(0.5, min(8.0, ratio * duration_scale))

    estimated_frames = int(math.ceil(target_text_token_count * ratio))
    min_blocks = target_text_token_count + 1
    inferred = max(min_blocks, estimated_frames + shared_delay_tokens + 1)
    return inferred + max(0, int(trailing_pad_blocks))


def build_teacher_forcing_target_text(
    *,
    target_text_tokens: torch.LongTensor,
    eos_token_id: int,
    pad_token_id: int,
    total_blocks: int,
) -> torch.LongTensor:
    if target_text_tokens.dim() != 1:
        raise ValueError(f"target_text_tokens must be 1D, got {tuple(target_text_tokens.shape)}.")
    min_required = int(target_text_tokens.shape[0]) + 1
    if total_blocks < min_required:
        raise ValueError(
            f"total_blocks={total_blocks} is too small, need at least {min_required} "
            "(text tokens + EOS)."
        )

    out = torch.full((total_blocks,), pad_token_id, dtype=torch.long)
    if target_text_tokens.numel() > 0:
        out[: target_text_tokens.shape[0]] = target_text_tokens
    out[target_text_tokens.shape[0]] = eos_token_id
    return out


def _normalize_generated_discrete(
    discrete_ids: torch.LongTensor,
    expected_stream_count: int,
) -> torch.LongTensor:
    if discrete_ids.dim() != 2:
        raise ValueError(
            f"Expected discrete ids with shape [N, L] or [L, N], got {tuple(discrete_ids.shape)}."
        )
    if discrete_ids.shape[0] == expected_stream_count:
        return discrete_ids.transpose(0, 1).contiguous()
    if discrete_ids.shape[1] == expected_stream_count:
        return discrete_ids.contiguous()
    raise ValueError(
        f"Unable to infer discrete stream axis for shape {tuple(discrete_ids.shape)} "
        f"with expected_stream_count={expected_stream_count}."
    )


def summarize_discrete_generation(
    *,
    discrete_ids: torch.LongTensor,
    num_discrete_tokens: int,
    special_token_ids: tuple[int, ...],
) -> dict[str, float]:
    discrete_by_time = _normalize_generated_discrete(
        discrete_ids=discrete_ids,
        expected_stream_count=num_discrete_tokens,
    )
    length = int(discrete_by_time.shape[0])
    if length < 1:
        return {
            "length": 0.0,
            "unique_tokens": 0.0,
            "special_ratio": 0.0,
            "longest_run_ratio": 0.0,
        }

    unique_tokens = int(torch.unique(discrete_by_time).numel())
    special_ratio = 0.0
    if len(special_token_ids) > 0:
        special = torch.tensor(
            list(special_token_ids),
            dtype=discrete_by_time.dtype,
            device=discrete_by_time.device,
        )
        is_special = torch.isin(discrete_by_time, special).all(dim=-1)
        special_ratio = float(is_special.float().mean().item())

    # With N streams, treat a full N-token block as one step for run-length stats.
    longest_run = 1
    current_run = 1
    for idx in range(1, length):
        if torch.equal(discrete_by_time[idx], discrete_by_time[idx - 1]):
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1

    return {
        "length": float(length),
        "unique_tokens": float(unique_tokens),
        "special_ratio": float(special_ratio),
        "longest_run_ratio": float(longest_run / max(1, length)),
    }


def is_collapsed_discrete_stats(stats: dict[str, float]) -> bool:
    return (
        stats["unique_tokens"] <= 3.0
        or stats["longest_run_ratio"] >= 0.65
        or stats["special_ratio"] >= 0.85
    )


def discrete_quality_score(stats: dict[str, float]) -> float:
    length = max(1.0, float(stats["length"]))
    diversity = float(stats["unique_tokens"]) / length
    return diversity + (1.0 - float(stats["special_ratio"])) + (1.0 - float(stats["longest_run_ratio"]))


def trim_latent_tail_from_special_blocks(
    *,
    latents: torch.FloatTensor,
    discrete_ids: torch.LongTensor,
    num_discrete_tokens: int,
    special_token_ids: tuple[int, ...],
) -> torch.FloatTensor:
    discrete_by_time = _normalize_generated_discrete(
        discrete_ids=discrete_ids,
        expected_stream_count=num_discrete_tokens,
    )
    if discrete_by_time.shape[0] != latents.shape[0]:
        # Keep original latents if shapes are inconsistent.
        return latents
    if len(special_token_ids) == 0:
        return latents

    special = torch.tensor(
        list(special_token_ids),
        dtype=discrete_by_time.dtype,
        device=discrete_by_time.device,
    )
    is_special = torch.isin(discrete_by_time, special).all(dim=-1)
    non_special_indices = torch.nonzero(~is_special, as_tuple=False)
    if non_special_indices.numel() == 0:
        return latents

    keep_length = int(non_special_indices[-1].item()) + 1
    if keep_length <= 0:
        return latents
    return latents[:keep_length]


def safe_tokenize(tokenizer: SharedVocabTokenizer, text: str, field_name: str) -> torch.LongTensor:
    token_ids = tokenizer(text)
    if len(token_ids) == 0 and text.strip() != "":
        warnings.warn(
            f"{field_name} became empty after OOV filtering. Check dataset/vocab.txt coverage.",
            RuntimeWarning,
            stacklevel=2,
        )
    return torch.tensor(token_ids, dtype=torch.long)
