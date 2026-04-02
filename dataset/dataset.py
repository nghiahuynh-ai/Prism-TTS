from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.dataset_utils import (
    _normalize_split_sample,
    _pad_1d,
    _pad_2d,
    _to_float_2d,
    _to_long_1d,
    _to_long_2d,
)

DEFAULT_DISCRETE_TOKEN_COUNT = 2048


def build_shared_token_layout(discrete_token_count: int) -> tuple[int, int, int, int]:
    """
    Return (delay_token_id, eos_token_id, pad_token_id, text_token_offset)
    for a given discrete token count.
    """
    if discrete_token_count < 1:
        raise ValueError("discrete_token_count must be >= 1.")
    delay_token_id = discrete_token_count
    eos_token_id = discrete_token_count + 1
    pad_token_id = discrete_token_count + 2
    text_token_offset = discrete_token_count + 3
    return delay_token_id, eos_token_id, pad_token_id, text_token_offset


(
    DEFAULT_DELAY_TOKEN_ID,
    DEFAULT_EOS_TOKEN_ID,
    DEFAULT_PAD_TOKEN_ID,
    DEFAULT_TEXT_TOKEN_OFFSET,
) = build_shared_token_layout(DEFAULT_DISCRETE_TOKEN_COUNT)


class SharedVocabTokenizer:
    """
    Character-level tokenizer with shared text/discrete id space:
    - [0, discrete_token_count - 1] -> discrete ids
    - discrete_token_count -> DELAY
    - discrete_token_count + 1 -> EOS
    - discrete_token_count + 2 -> PAD
    - the rest -> text characters from vocab file
    """

    def __init__(
        self,
        vocab_path: str | Path,
        text_token_offset: int,
        eos_token_id: int,
        append_eos: bool = False,
    ) -> None:
        self.vocab_path = Path(vocab_path).expanduser().resolve()
        if not self.vocab_path.is_file():
            raise FileNotFoundError(f"Vocab file not found: {self.vocab_path}")

        self.text_token_offset = int(text_token_offset)
        self.eos_token_id = int(eos_token_id)
        self.append_eos = append_eos
        self.char_to_id = self._load_char_vocab(self.vocab_path)

    @staticmethod
    def _load_char_vocab(vocab_path: Path) -> dict[str, int]:
        char_to_local_id: dict[str, int] = {}
        with vocab_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                token = raw_line.rstrip("\r\n")
                if token == "":
                    continue
                if len(token) != 1:
                    raise ValueError(
                        f"Invalid vocab token at line {line_number}: {token!r}. "
                        "Expected exactly one character per line."
                    )
                if token in char_to_local_id:
                    raise ValueError(f"Duplicate vocab token at line {line_number}: {token!r}.")
                char_to_local_id[token] = len(char_to_local_id)
        if not char_to_local_id:
            raise ValueError(f"Vocab file is empty: {vocab_path}")
        return char_to_local_id

    def __call__(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for index, char in enumerate(text):
            local_id = self.char_to_id.get(char)
            if local_id is None:
                warnings.warn(
                    f"Skipping out-of-vocab character {char!r} at index {index} "
                    f"for vocab {self.vocab_path}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            token_ids.append(self.text_token_offset + local_id)
        if self.append_eos:
            token_ids.append(self.eos_token_id)
        return token_ids


@dataclass(frozen=True)
class ManifestEntry:
    file_name: str
    duration: float
    transcript: str
    target_npy_path: Path
    prompt_file_name: str
    prompt_duration: float
    prompt_transcript: str
    prompt_npy_path: Path


class PrismDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Prism-TTS dataset.

    Expected manifest line format:
    file_name|duration|transcript|target_npy|prompt_file_name|prompt_duration|prompt_transcript|prompt_npy

    Shared token-id layout:
    - [0, discrete_token_count - 1]: discrete tokens
    - discrete_token_count: DELAY
    - discrete_token_count + 1: EOS
    - discrete_token_count + 2: PAD
    - [discrete_token_count + 3, ...]: text tokens from vocab.txt

    Each NPY file must contain:
    - discrete: int array, shape [L, N]
    - continuous: float array, shape [L, D]
    where N and D can be enforced via constructor configs.
    """

    def __init__(
        self,
        source: str | Path | Sequence[Mapping[str, Any]],
        prompt_length: int | None = None,
        min_prompt_length: int = 1,
        min_target_length: int = 1,
        vocab_path: str | Path | None = None,
        manifest_root: str | Path | None = None,
        discrete_token_count: int = DEFAULT_DISCRETE_TOKEN_COUNT,
        discrete_stream_count: int | None = None,
        continuous_feature_dim: int | None = None,
        append_eos_to_text: bool = False,
        cache_npy: bool = False,
        cache_npz: bool | None = None,
    ) -> None:
        self.discrete_token_count = int(discrete_token_count)
        (
            self.delay_token_id,
            self.eos_token_id,
            self.pad_token_id,
            self.text_token_offset,
        ) = build_shared_token_layout(self.discrete_token_count)

        resolved_vocab_path = (
            Path(vocab_path).expanduser().resolve()
            if vocab_path is not None
            else Path(__file__).resolve().parent / "vocab.txt"
        )
        self.tokenizer = SharedVocabTokenizer(
            vocab_path=resolved_vocab_path,
            text_token_offset=self.text_token_offset,
            eos_token_id=self.eos_token_id,
            append_eos=append_eos_to_text,
        )
        self.vocab_path = resolved_vocab_path

        if cache_npz is not None:
            cache_npy = bool(cache_npz)
        self.cache_npy = bool(cache_npy)
        self._npy_cache: dict[str, tuple[torch.LongTensor, torch.FloatTensor]] = {}
        self.discrete_stream_count = (
            None if discrete_stream_count is None else int(discrete_stream_count)
        )
        self.continuous_feature_dim = (
            None if continuous_feature_dim is None else int(continuous_feature_dim)
        )
        if self.discrete_stream_count is not None and self.discrete_stream_count < 1:
            raise ValueError("discrete_stream_count must be >= 1 when provided.")
        if self.continuous_feature_dim is not None and self.continuous_feature_dim < 1:
            raise ValueError("continuous_feature_dim must be >= 1 when provided.")
        # Kept for compatibility with older call-sites.
        self.prompt_length = prompt_length
        self.min_prompt_length = min_prompt_length
        self.min_target_length = min_target_length

        self._entries: list[ManifestEntry] = []
        self._samples: list[Mapping[str, Any]] = []

        if isinstance(source, (str, Path)):
            manifest_path = Path(source).expanduser().resolve()
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
            self.manifest_path = manifest_path
            self.manifest_root = (
                Path(manifest_root).expanduser().resolve()
                if manifest_root is not None
                else manifest_path.parent
            )
            self._entries = self._load_manifest(manifest_path)
        else:
            self.manifest_path = None
            self.manifest_root = (
                Path(manifest_root).expanduser().resolve() if manifest_root is not None else None
            )
            self._samples = list(source)

    def __len__(self) -> int:
        return len(self._entries) if self._entries else len(self._samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self._entries:
            return self._build_manifest_sample(self._entries[index])
        return _normalize_split_sample(self._samples[index])

    def _load_manifest(self, manifest_path: Path) -> list[ManifestEntry]:
        entries: list[ManifestEntry] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                entries.append(self._parse_manifest_line(line, line_number))
        if not entries:
            raise ValueError(f"Manifest has no valid entries: {manifest_path}")
        return entries

    def _parse_manifest_line(self, line: str, line_number: int) -> ManifestEntry:
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 8:
            raise ValueError(
                "Each manifest line must have exactly 8 fields separated by '|'. "
                f"line={line_number}, fields={len(parts)}"
            )

        (
            file_name,
            duration_str,
            transcript,
            target_npy_str,
            prompt_file_name,
            prompt_duration_str,
            prompt_transcript,
            prompt_npy_str,
        ) = parts

        return ManifestEntry(
            file_name=file_name,
            duration=self._parse_float(duration_str, line_number, "duration"),
            transcript=transcript,
            target_npy_path=self._resolve_npy_path(target_npy_str, line_number, "target_npy"),
            prompt_file_name=prompt_file_name,
            prompt_duration=self._parse_float(prompt_duration_str, line_number, "prompt_duration"),
            prompt_transcript=prompt_transcript,
            prompt_npy_path=self._resolve_npy_path(prompt_npy_str, line_number, "prompt_npy"),
        )

    def _parse_float(self, value: str, line_number: int, field_name: str) -> float:
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {field_name} at line {line_number}: {value!r}."
            ) from exc

    def _resolve_npy_path(self, raw_path: str, line_number: int, field_name: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            if self.manifest_root is None:
                raise ValueError(
                    f"Relative {field_name} requires manifest_root (line {line_number})."
                )
            path = self.manifest_root / path
        path = path.resolve()

        if path.suffix.lower() != ".npy":
            raise ValueError(f"{field_name} at line {line_number} must be a .npy path: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"{field_name} file not found at line {line_number}: {path}")
        return path

    @staticmethod
    def _extract_modal_arrays_from_npy(
        payload: Any,
        npy_path: Path,
    ) -> tuple[Any, Any]:
        if isinstance(payload, np.ndarray) and payload.dtype.names is not None:
            field_names = set(payload.dtype.names)
            if "discrete" in field_names and "continuous" in field_names:
                discrete = payload["discrete"]
                continuous = payload["continuous"]
                if (
                    isinstance(discrete, np.ndarray)
                    and discrete.dtype == object
                    and discrete.shape == ()
                ):
                    discrete = discrete.item()
                if (
                    isinstance(continuous, np.ndarray)
                    and continuous.dtype == object
                    and continuous.shape == ()
                ):
                    continuous = continuous.item()
                return discrete, continuous

        if isinstance(payload, np.ndarray) and payload.dtype == object and payload.shape == ():
            value = payload.item()
            if isinstance(value, Mapping):
                if "discrete" in value and "continuous" in value:
                    return value["discrete"], value["continuous"]
            if isinstance(value, tuple) and len(value) == 2:
                return value[0], value[1]

        raise ValueError(
            f"{npy_path} must contain 'discrete' and 'continuous' arrays. "
            "Supported .npy formats: structured arrays with named fields or "
            "pickled dict/tuple payloads."
        )

    def _load_npy_features(self, npy_path: Path) -> tuple[torch.LongTensor, torch.FloatTensor]:
        cache_key = str(npy_path)
        if self.cache_npy and cache_key in self._npy_cache:
            discrete, continuous = self._npy_cache[cache_key]
            return discrete.clone(), continuous.clone()

        try:
            try:
                payload = np.load(npy_path, allow_pickle=False)
            except ValueError as exc:
                if "allow_pickle=False" not in str(exc):
                    raise
                payload = np.load(npy_path, allow_pickle=True)

            discrete_raw, continuous_raw = self._extract_modal_arrays_from_npy(
                payload,
                npy_path,
            )
            discrete = _to_long_2d(discrete_raw, f"{npy_path}:discrete")
            continuous = _to_float_2d(continuous_raw, f"{npy_path}:continuous")
        except Exception as exc:
            raise RuntimeError(f"Failed to load npy file {npy_path}: {exc}") from exc

        self._validate_feature_shapes(discrete, continuous, npy_path)

        if discrete.numel() > 0:
            discrete_min = int(discrete.min().item())
            discrete_max = int(discrete.max().item())
            if discrete_min < 0 or discrete_max >= self.text_token_offset:
                raise ValueError(
                    "Discrete ids must be within shared discrete/special range "
                    f"[0, {self.text_token_offset - 1}], got min={discrete_min}, "
                    f"max={discrete_max} in {npy_path}. "
                    "Text ids start from text_token_offset."
                )

        if self.cache_npy:
            self._npy_cache[cache_key] = (discrete, continuous)
            return discrete.clone(), continuous.clone()
        return discrete, continuous

    def _validate_feature_shapes(
        self,
        discrete: torch.LongTensor,
        continuous: torch.FloatTensor,
        npy_path: Path,
    ) -> None:
        if discrete.shape[0] != continuous.shape[0]:
            raise ValueError(
                f"{npy_path} length mismatch between modalities: "
                f"discrete L={discrete.shape[0]}, continuous L={continuous.shape[0]}."
            )
        if self.discrete_stream_count is not None and discrete.shape[1] != self.discrete_stream_count:
            raise ValueError(
                f"{npy_path} discrete shape mismatch: expected [L, {self.discrete_stream_count}], "
                f"got {tuple(discrete.shape)}."
            )
        if self.continuous_feature_dim is not None and continuous.shape[1] != self.continuous_feature_dim:
            raise ValueError(
                f"{npy_path} continuous shape mismatch: expected [L, {self.continuous_feature_dim}], "
                f"got {tuple(continuous.shape)}."
            )

    def _encode_text(self, text: str, field_name: str) -> torch.LongTensor:
        try:
            encoded = self.tokenizer(text)
        except Exception as exc:
            raise RuntimeError(f"tokenization failed for {field_name}: {exc}") from exc
        return _to_long_1d(encoded, field_name)

    def _build_manifest_sample(self, entry: ManifestEntry) -> dict[str, torch.Tensor]:
        discrete_target, continuous_target = self._load_npy_features(entry.target_npy_path)
        discrete_prompt, continuous_prompt = self._load_npy_features(entry.prompt_npy_path)

        text_target = self._encode_text(entry.transcript, "transcript")
        text_prompt = self._encode_text(entry.prompt_transcript, "prompt_transcript")

        return _normalize_split_sample(
            {
                "text_target": text_target,
                "discrete_target": discrete_target,
                "continuous_target": continuous_target,
                "text_prompt": text_prompt,
                "discrete_prompt": discrete_prompt,
                "continuous_prompt": continuous_prompt,
            }
        )


class BatchCollate:
    """Pad Prism-TTS samples using shared special-token ids across modalities."""

    def __init__(
        self,
        text_pad_value: int | None = None,
        discrete_pad_value: int | None = None,
        continuous_pad_value: float = 0.0,
        include_attention_mask: bool = True,
        discrete_token_count: int = DEFAULT_DISCRETE_TOKEN_COUNT,
        prompt_length: int | None = None,
        min_prompt_length: int = 1,
        min_target_length: int = 1,
        stream_delay: int | None = None,
        discrete_stream_delay_ms: float | None = None,
        codec_frame_rate_hz: float = 12.5,
    ) -> None:
        (
            self.delay_token_id,
            self.eos_token_id,
            self.pad_token_id,
            _,
        ) = build_shared_token_layout(int(discrete_token_count))
        self.text_pad_value = self.pad_token_id if text_pad_value is None else int(text_pad_value)
        self.discrete_pad_value = (
            self.pad_token_id if discrete_pad_value is None else int(discrete_pad_value)
        )
        self.continuous_pad_value = continuous_pad_value
        self.include_attention_mask = include_attention_mask
        self.discrete_token_count = int(discrete_token_count)
        # Kept for compatibility with older call-sites.
        self.prompt_length = prompt_length
        self.min_prompt_length = min_prompt_length
        self.min_target_length = min_target_length
        if stream_delay is not None and discrete_stream_delay_ms is not None:
            raise ValueError(
                "Provide either stream_delay (tokens) or "
                "discrete_stream_delay_ms (milliseconds), not both."
            )
        if codec_frame_rate_hz <= 0:
            raise ValueError("codec_frame_rate_hz must be > 0.")
        self.codec_frame_rate_hz = float(codec_frame_rate_hz)
        self.stream_delay = None if stream_delay is None else int(stream_delay)
        if self.stream_delay is not None and self.stream_delay < 0:
            raise ValueError("stream_delay must be non-negative.")
        self.discrete_stream_delay_ms = (
            None if discrete_stream_delay_ms is None else float(discrete_stream_delay_ms)
        )
        if self.discrete_stream_delay_ms is not None and self.discrete_stream_delay_ms < 0:
            raise ValueError("discrete_stream_delay_ms must be non-negative.")

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("BatchCollate received an empty batch.")

        samples = [self._validate_collate_sample(item) for item in batch]
        prepared = [self._prepare_sample_streams(sample) for sample in samples]

        collated: dict[str, torch.Tensor] = {
            "text": _pad_1d([x["text"] for x in prepared], self.text_pad_value),
            "discrete": _pad_2d([x["discrete"] for x in prepared], self.discrete_pad_value),
            "continuous": _pad_2d([x["continuous"] for x in prepared], self.continuous_pad_value),
            "prompt_lengths": torch.tensor(
                [int(x["prompt_length"]) for x in prepared],
                dtype=torch.long,
            ),
            "target_lengths": torch.tensor(
                [int(x["target_length"]) for x in prepared],
                dtype=torch.long,
            ),
        }
        if self.include_attention_mask:
            collated["attention_mask"] = _pad_1d(
                [x["attention_mask"] for x in prepared],
                False,
            ).to(dtype=torch.bool)

        # Legacy split keys are kept for compatibility with call-sites that still
        # require explicit prompt/target tensors (e.g., generation utilities).
        collated["text_prompt"] = _pad_1d([x["text_prompt"] for x in prepared], self.text_pad_value)
        collated["discrete_prompt"] = _pad_2d(
            [x["discrete_prompt"] for x in prepared],
            self.discrete_pad_value,
        )
        collated["continuous_prompt"] = _pad_2d(
            [x["continuous_prompt"] for x in prepared],
            self.continuous_pad_value,
        )
        collated["text_target"] = _pad_1d([x["text_target"] for x in prepared], self.text_pad_value)
        collated["discrete_target"] = _pad_2d(
            [x["discrete_target"] for x in prepared],
            self.discrete_pad_value,
        )
        collated["continuous_target"] = _pad_2d(
            [x["continuous_target"] for x in prepared],
            self.continuous_pad_value,
        )

        self._collate_optional_1d(prepared, collated, key="flow_timesteps", pad_value=0.0)
        self._collate_optional_2d(prepared, collated, key="noise", pad_value=0.0)
        return collated

    def _validate_collate_sample(self, sample: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        required_keys = (
            "text_target",
            "discrete_target",
            "continuous_target",
            "text_prompt",
            "discrete_prompt",
            "continuous_prompt",
        )
        missing = [key for key in required_keys if key not in sample]
        if missing:
            raise KeyError(f"Missing required keys in collate sample: {missing}")

        def require_tensor(name: str, expected_dim: int) -> torch.Tensor:
            value = sample[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"{name} must be a torch.Tensor in BatchCollate. "
                    "Use PrismDataset (or _normalize_split_sample) before collate."
                )
            if value.dim() != expected_dim:
                raise ValueError(f"{name} must be {expected_dim}D, got shape {tuple(value.shape)}.")
            return value

        text_target = require_tensor("text_target", 1)
        discrete_target = require_tensor("discrete_target", 2)
        continuous_target = require_tensor("continuous_target", 2)
        text_prompt = require_tensor("text_prompt", 1)
        discrete_prompt = require_tensor("discrete_prompt", 2)
        continuous_prompt = require_tensor("continuous_prompt", 2)

        if discrete_target.shape[1] != discrete_prompt.shape[1]:
            raise ValueError(
                "discrete_target and discrete_prompt must have the same number of discrete streams."
            )
        if continuous_target.shape[1] != continuous_prompt.shape[1]:
            raise ValueError(
                "continuous_target and continuous_prompt must have the same channel size."
            )
        if discrete_target.shape[0] != continuous_target.shape[0]:
            raise ValueError("discrete_target and continuous_target must share the same length.")
        if discrete_prompt.shape[0] != continuous_prompt.shape[0]:
            raise ValueError("discrete_prompt and continuous_prompt must share the same length.")

        normalized: dict[str, torch.Tensor] = {
            "text_target": text_target,
            "discrete_target": discrete_target,
            "continuous_target": continuous_target,
            "text_prompt": text_prompt,
            "discrete_prompt": discrete_prompt,
            "continuous_prompt": continuous_prompt,
        }

        if "attention_mask" in sample and sample["attention_mask"] is not None:
            raise ValueError(
                "Per-sample attention_mask is not supported in BatchCollate. "
                "BatchCollate now builds pad-only attention masks from aligned streams."
            )

        if "flow_timesteps" in sample and sample["flow_timesteps"] is not None:
            flow_timesteps = sample["flow_timesteps"]
            if not isinstance(flow_timesteps, torch.Tensor) or flow_timesteps.dim() != 1:
                raise ValueError("flow_timesteps must be a 1D torch.Tensor when provided.")
            normalized["flow_timesteps"] = flow_timesteps

        if "noise" in sample and sample["noise"] is not None:
            noise = sample["noise"]
            if not isinstance(noise, torch.Tensor) or noise.dim() != 2:
                raise ValueError("noise must be a 2D torch.Tensor when provided.")
            if noise.shape[1] != continuous_target.shape[1]:
                raise ValueError("noise channel size must match continuous_target channel size.")
            normalized["noise"] = noise

        return normalized

    def _resolve_shared_delay(self) -> int:
        if self.stream_delay is not None:
            return int(self.stream_delay)
        if self.discrete_stream_delay_ms is not None:
            return int(math.ceil((self.discrete_stream_delay_ms * self.codec_frame_rate_hz) / 1000.0))
        return 0

    @staticmethod
    def _pad_to_length_1d(
        tensor: torch.Tensor,
        length: int,
        pad_value: int | float,
    ) -> torch.Tensor:
        if tensor.dim() != 1:
            raise ValueError(f"Expected 1D tensor, got shape {tuple(tensor.shape)}.")
        if length < tensor.shape[0]:
            raise ValueError("Cannot pad to a length smaller than the source tensor.")
        if length == tensor.shape[0]:
            return tensor
        out = torch.full((length,), pad_value, dtype=tensor.dtype, device=tensor.device)
        out[: tensor.shape[0]] = tensor
        return out

    @staticmethod
    def _pad_to_length_2d(
        tensor: torch.Tensor,
        length: int,
        pad_value: int | float,
    ) -> torch.Tensor:
        if tensor.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got shape {tuple(tensor.shape)}.")
        if length < tensor.shape[0]:
            raise ValueError("Cannot pad to a length smaller than the source tensor.")
        if length == tensor.shape[0]:
            return tensor
        out = torch.full(
            (length, tensor.shape[1]),
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        out[: tensor.shape[0], :] = tensor
        return out

    def _global_extra_delay(self, text_length: int, discrete_length: int, shared_delay: int) -> int:
        # Ensure the last text token index is strictly smaller than the last real
        # discrete/continuous token index in this part.
        # last_text_idx = text_length - 1
        # last_discrete_idx = shared_delay + extra + discrete_length - 1
        # need: last_discrete_idx > last_text_idx
        required = text_length - discrete_length - shared_delay + 1
        return max(0, int(required))

    def _build_aligned_part(
        self,
        text_tokens: torch.LongTensor,
        discrete_tokens: torch.LongTensor,
        continuous_latents: torch.FloatTensor,
        shared_delay: int,
    ) -> dict[str, torch.Tensor]:
        if text_tokens.dim() != 1:
            raise ValueError("text_tokens must be 1D.")
        if discrete_tokens.dim() != 2:
            raise ValueError("discrete_tokens must be 2D [L, N].")
        if continuous_latents.dim() != 2:
            raise ValueError("continuous_latents must be 2D [L, C].")
        if discrete_tokens.shape[0] != continuous_latents.shape[0]:
            raise ValueError("discrete_tokens and continuous_latents must share length.")
        text_length = int(text_tokens.shape[0])
        discrete_length = int(discrete_tokens.shape[0])
        extra_delay = self._global_extra_delay(
            text_length=text_length,
            discrete_length=discrete_length,
            shared_delay=shared_delay,
        )
        effective_delay = int(shared_delay) + extra_delay
        continuous_delay = effective_delay

        text_eos = text_tokens.new_tensor([self.eos_token_id], dtype=text_tokens.dtype)
        text_stream = torch.cat([text_tokens, text_eos], dim=0)

        discrete_streams: list[torch.Tensor] = []
        for stream_idx in range(int(discrete_tokens.shape[1])):
            prefix = discrete_tokens.new_full((effective_delay,), self.delay_token_id)
            eos = discrete_tokens.new_full((1,), self.eos_token_id)
            stream = torch.cat([prefix, discrete_tokens[:, stream_idx], eos], dim=0)
            discrete_streams.append(stream)

        continuous_channel_count = int(continuous_latents.shape[1])
        continuous_prefix = continuous_latents.new_zeros(
            (continuous_delay, continuous_channel_count),
        )
        continuous_eos = continuous_latents.new_zeros((1, continuous_channel_count))
        continuous_stream = torch.cat(
            [continuous_prefix, continuous_latents, continuous_eos],
            dim=0,
        )

        max_stream_len = max(
            int(text_stream.shape[0]),
            max(int(stream.shape[0]) for stream in discrete_streams) if discrete_streams else 0,
            int(continuous_stream.shape[0]),
        )
        text_aligned = self._pad_to_length_1d(text_stream, max_stream_len, self.text_pad_value)

        discrete_aligned = discrete_tokens.new_full(
            (max_stream_len, len(discrete_streams)),
            self.discrete_pad_value,
        )
        for stream_idx, stream in enumerate(discrete_streams):
            discrete_aligned[: stream.shape[0], stream_idx] = stream

        continuous_aligned = self._pad_to_length_2d(
            continuous_stream,
            max_stream_len,
            self.continuous_pad_value,
        )

        return {
            "text": text_aligned,
            "discrete": discrete_aligned,
            "continuous": continuous_aligned,
        }

    def _maybe_align_flow_timesteps(
        self,
        flow_timesteps: torch.Tensor | None,
        target_continuous_len: int,
        target_aligned_len: int,
        target_continuous_delay: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if flow_timesteps is None:
            return None
        if flow_timesteps.shape[0] != target_continuous_len:
            raise ValueError(
                "flow_timesteps length must match raw target continuous length "
                f"({target_continuous_len})."
            )
        aligned = torch.zeros(target_aligned_len, dtype=flow_timesteps.dtype, device=device)
        start = target_continuous_delay
        end = start + target_continuous_len
        aligned[start:end] = flow_timesteps.to(device=device)
        return aligned

    def _maybe_align_noise(
        self,
        noise: torch.Tensor | None,
        target_continuous_shape: tuple[int, int],
        target_aligned_len: int,
        target_continuous_delay: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if noise is None:
            return None
        target_len, channels = target_continuous_shape
        if noise.shape != (target_len, channels):
            raise ValueError(
                "noise must have shape [raw_target_len, continuous_channels]. "
                f"Expected {(target_len, channels)}, got {tuple(noise.shape)}."
            )
        aligned = torch.zeros(
            target_aligned_len,
            channels,
            dtype=noise.dtype,
            device=device,
        )
        start = target_continuous_delay
        end = start + target_len
        aligned[start:end, :] = noise.to(device=device)
        return aligned

    def _prepare_sample_streams(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        shared_delay = self._resolve_shared_delay()

        prompt_part = self._build_aligned_part(
            text_tokens=sample["text_prompt"],
            discrete_tokens=sample["discrete_prompt"],
            continuous_latents=sample["continuous_prompt"],
            shared_delay=shared_delay,
        )
        target_part = self._build_aligned_part(
            text_tokens=sample["text_target"],
            discrete_tokens=sample["discrete_target"],
            continuous_latents=sample["continuous_target"],
            shared_delay=shared_delay,
        )

        text = torch.cat([prompt_part["text"], target_part["text"]], dim=0)
        discrete = torch.cat([prompt_part["discrete"], target_part["discrete"]], dim=0)
        continuous = torch.cat([prompt_part["continuous"], target_part["continuous"]], dim=0)

        prompt_length = int(prompt_part["text"].shape[0])
        target_length = int(target_part["text"].shape[0])
        attention_mask = torch.ones(
            prompt_length + target_length,
            dtype=torch.bool,
            device=text.device,
        )

        prepared: dict[str, torch.Tensor] = {
            "text": text,
            "discrete": discrete,
            "continuous": continuous,
            "attention_mask": attention_mask,
            "text_prompt": prompt_part["text"],
            "discrete_prompt": prompt_part["discrete"],
            "continuous_prompt": prompt_part["continuous"],
            "text_target": target_part["text"],
            "discrete_target": target_part["discrete"],
            "continuous_target": target_part["continuous"],
            "prompt_length": torch.tensor(prompt_length, dtype=torch.long),
            "target_length": torch.tensor(target_length, dtype=torch.long),
        }

        if "flow_timesteps" in sample:
            extra_delay = self._global_extra_delay(
                text_length=int(sample["text_target"].shape[0]),
                discrete_length=int(sample["discrete_target"].shape[0]),
                shared_delay=shared_delay,
            )
            target_delay = shared_delay + extra_delay
            aligned_flow_timesteps = self._maybe_align_flow_timesteps(
                flow_timesteps=sample["flow_timesteps"],
                target_continuous_len=int(sample["continuous_target"].shape[0]),
                target_aligned_len=target_length,
                target_continuous_delay=target_delay,
                device=text.device,
            )
            if aligned_flow_timesteps is not None:
                prepared["flow_timesteps"] = aligned_flow_timesteps

        if "noise" in sample:
            extra_delay = self._global_extra_delay(
                text_length=int(sample["text_target"].shape[0]),
                discrete_length=int(sample["discrete_target"].shape[0]),
                shared_delay=shared_delay,
            )
            target_delay = shared_delay + extra_delay
            aligned_noise = self._maybe_align_noise(
                noise=sample["noise"],
                target_continuous_shape=(
                    int(sample["continuous_target"].shape[0]),
                    int(sample["continuous_target"].shape[1]),
                ),
                target_aligned_len=target_length,
                target_continuous_delay=target_delay,
                device=text.device,
            )
            if aligned_noise is not None:
                prepared["noise"] = aligned_noise

        return prepared

    @staticmethod
    def _collate_optional_1d(
        samples: Sequence[dict[str, torch.Tensor]],
        output: dict[str, torch.Tensor],
        key: str,
        pad_value: float,
    ) -> None:
        presence = [key in sample for sample in samples]
        if not any(presence):
            return
        if not all(presence):
            raise ValueError(f"{key} must be provided for all samples or for none.")
        output[key] = _pad_1d([sample[key] for sample in samples], pad_value)

    @staticmethod
    def _collate_optional_2d(
        samples: Sequence[dict[str, torch.Tensor]],
        output: dict[str, torch.Tensor],
        key: str,
        pad_value: float,
    ) -> None:
        presence = [key in sample for sample in samples]
        if not any(presence):
            return
        if not all(presence):
            raise ValueError(f"{key} must be provided for all samples or for none.")
        output[key] = _pad_2d([sample[key] for sample in samples], pad_value)
