from __future__ import annotations

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
TEXT_TOKEN_TYPE = 0
SPEECH_DISCRETE_TOKEN_TYPE = 1
SPEECH_CONTINUOUS_TOKEN_TYPE = 2


def build_shared_token_layout(discrete_token_count: int) -> tuple[int, int, int, int]:
    """
    Return (eot_token_id, eos_token_id, pad_token_id, text_token_offset)
    for a given discrete token count.
    """
    if discrete_token_count < 1:
        raise ValueError("discrete_token_count must be >= 1.")
    eot_token_id = discrete_token_count
    eos_token_id = discrete_token_count + 1
    pad_token_id = discrete_token_count + 2
    text_token_offset = discrete_token_count + 3
    return eot_token_id, eos_token_id, pad_token_id, text_token_offset


(
    DEFAULT_EOT_TOKEN_ID,
    DEFAULT_EOS_TOKEN_ID,
    DEFAULT_PAD_TOKEN_ID,
    DEFAULT_TEXT_TOKEN_OFFSET,
) = build_shared_token_layout(DEFAULT_DISCRETE_TOKEN_COUNT)


class SharedVocabTokenizer:
    """
    Character-level tokenizer with shared text/discrete id space:
    - [0, discrete_token_count - 1] -> discrete ids
    - discrete_token_count -> EOT (text boundary marker)
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
    - discrete_token_count: EOT
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
            self.eot_token_id,
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
        # Some manifests include a trailing delimiter, producing an empty final field.
        while parts and parts[-1] == "":
            parts.pop()
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
            # Drop the final codec frame (~80ms at 12.5 Hz) from both modalities.
            discrete = discrete[:-1]
            continuous = continuous[:-1]
        except Exception as exc:
            raise RuntimeError(f"Failed to load npy file {npy_path}: {exc}") from exc

        if self.discrete_stream_count is not None and discrete.shape[1] > self.discrete_stream_count:
            discrete = discrete[:, : self.discrete_stream_count]

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
    """Pad Prism-TTS samples without delay alignment and append target EOS speech blocks."""

    def __init__(
        self,
        text_pad_value: int | None = None,
        discrete_pad_value: int | None = None,
        continuous_pad_value: float = 0.0,
        include_attention_mask: bool = True,
        discrete_token_count: int = DEFAULT_DISCRETE_TOKEN_COUNT,
        random_active_discrete_stream_count: bool = True,
        min_active_discrete_stream_count: int = 1,
        max_active_discrete_stream_count: int | None = None,
        fixed_continuous_stream_idx: int | None = None,
    ) -> None:
        (
            self.eot_token_id,
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
        self.random_active_discrete_stream_count = bool(random_active_discrete_stream_count)
        self.min_active_discrete_stream_count = int(min_active_discrete_stream_count)
        self.max_active_discrete_stream_count = (
            None
            if max_active_discrete_stream_count is None
            else int(max_active_discrete_stream_count)
        )
        self.fixed_continuous_stream_idx = (
            None if fixed_continuous_stream_idx is None else int(fixed_continuous_stream_idx)
        )
        if self.min_active_discrete_stream_count < 1:
            raise ValueError("min_active_discrete_stream_count must be >= 1.")
        if (
            self.max_active_discrete_stream_count is not None
            and self.max_active_discrete_stream_count < 1
        ):
            raise ValueError("max_active_discrete_stream_count must be >= 1 when provided.")
        if (
            self.max_active_discrete_stream_count is not None
            and self.min_active_discrete_stream_count > self.max_active_discrete_stream_count
        ):
            raise ValueError(
                "min_active_discrete_stream_count must be <= max_active_discrete_stream_count."
            )
        if self.fixed_continuous_stream_idx is not None and self.fixed_continuous_stream_idx < 0:
            raise ValueError("fixed_continuous_stream_idx must be >= 0 when provided.")

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("BatchCollate received an empty batch.")

        normalized_samples = [self._validate_collate_sample(item) for item in batch]
        active_discrete_stream_count = self._resolve_active_discrete_stream_count(normalized_samples)
        reduced_samples = [
            self._select_first_discrete_streams(sample, active_discrete_stream_count)
            for sample in normalized_samples
        ]
        samples = [self._append_target_eos_block(sample) for sample in reduced_samples]
        text_prompt_lengths = torch.tensor(
            [int(sample["text_prompt"].shape[0]) for sample in samples],
            dtype=torch.long,
        )
        speech_prompt_lengths = torch.tensor(
            [int(sample["discrete_prompt"].shape[0]) for sample in samples],
            dtype=torch.long,
        )
        text_target_lengths = torch.tensor(
            [int(sample["text_target"].shape[0]) for sample in samples],
            dtype=torch.long,
        )
        speech_target_lengths = torch.tensor(
            [int(sample["discrete_target"].shape[0]) for sample in samples],
            dtype=torch.long,
        )

        collated: dict[str, torch.Tensor] = {
            "text_prompt": _pad_1d([sample["text_prompt"] for sample in samples], self.text_pad_value),
            "discrete_prompt": _pad_2d(
                [sample["discrete_prompt"] for sample in samples],
                self.discrete_pad_value,
            ),
            "continuous_prompt": _pad_2d(
                [sample["continuous_prompt"] for sample in samples],
                self.continuous_pad_value,
            ),
            "text_target": _pad_1d([sample["text_target"] for sample in samples], self.text_pad_value),
            "discrete_target": _pad_2d(
                [sample["discrete_target"] for sample in samples],
                self.discrete_pad_value,
            ),
            "continuous_target": _pad_2d(
                [sample["continuous_target"] for sample in samples],
                self.continuous_pad_value,
            ),
            "text_prompt_lengths": text_prompt_lengths,
            "speech_prompt_lengths": speech_prompt_lengths,
            "text_target_lengths": text_target_lengths,
            "speech_target_lengths": speech_target_lengths,
            "active_discrete_stream_count": torch.tensor(
                active_discrete_stream_count,
                dtype=torch.long,
            ),
        }

        flat_per_sample = [self._build_flat_sample(sample) for sample in samples]
        collated["flat_token_ids"] = _pad_1d(
            [item["token_ids"] for item in flat_per_sample],
            self.pad_token_id,
        )
        collated["flat_token_type_ids"] = _pad_1d(
            [item["token_type_ids"] for item in flat_per_sample],
            TEXT_TOKEN_TYPE,
        )
        collated["flat_speech_stream_ids"] = _pad_1d(
            [item["speech_stream_ids"] for item in flat_per_sample],
            -1,
        )
        collated["flat_target_block_ids"] = _pad_1d(
            [item["target_block_ids"] for item in flat_per_sample],
            -1,
        )
        collated["flat_continuous_values"] = _pad_2d(
            [item["continuous_values"] for item in flat_per_sample],
            0.0,
        )
        collated["flat_target_block_counts"] = speech_target_lengths
        collated["flat_summary"] = torch.stack(
            [item["summary"] for item in flat_per_sample],
            dim=0,
        )

        if self.include_attention_mask:
            collated["attention_mask"] = _pad_1d(
                [item["attention_mask"] for item in flat_per_sample],
                False,
            ).to(dtype=torch.bool)

        self._collate_optional_1d(samples, collated, key="flow_timesteps", pad_value=0.0)
        self._collate_optional_2d(samples, collated, key="noise", pad_value=0.0)
        return collated

    def _resolve_active_discrete_stream_count(
        self,
        samples: Sequence[dict[str, torch.Tensor]],
    ) -> int:
        available_streams = min(
            int(sample["discrete_prompt"].shape[1])
            for sample in samples
        )
        available_streams = min(
            available_streams,
            min(int(sample["discrete_target"].shape[1]) for sample in samples),
        )
        if available_streams < 1:
            raise ValueError("At least one discrete stream is required for collation.")

        upper = available_streams
        if self.max_active_discrete_stream_count is not None:
            upper = min(upper, self.max_active_discrete_stream_count)
        lower = min(self.min_active_discrete_stream_count, upper)
        if lower < 1:
            raise ValueError("Resolved lower bound for active discrete streams must be >= 1.")

        if not self.random_active_discrete_stream_count:
            return int(upper)
        return int(torch.randint(low=lower, high=upper + 1, size=(1,)).item())

    @staticmethod
    def _select_first_discrete_streams(
        sample: dict[str, torch.Tensor],
        active_discrete_stream_count: int,
    ) -> dict[str, torch.Tensor]:
        selected = dict(sample)
        selected["discrete_prompt"] = sample["discrete_prompt"][:, :active_discrete_stream_count]
        selected["discrete_target"] = sample["discrete_target"][:, :active_discrete_stream_count]
        return selected

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
                "BatchCollate now builds pad-only attention masks from concatenated parts."
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

    def _has_terminal_target_eos_block(self, sample: Mapping[str, torch.Tensor]) -> bool:
        discrete_target = sample["discrete_target"]
        continuous_target = sample["continuous_target"]
        if int(discrete_target.shape[0]) < 1:
            return False
        if not bool(torch.eq(discrete_target[-1], self.eos_token_id).all().item()):
            return False
        last_continuous = continuous_target[-1]
        return bool(torch.le(last_continuous.abs().max(), 1e-8).item())

    def _append_target_eos_block(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Ensure each target speech sequence ends with one explicit EOS block:
        - discrete: all streams = eos_token_id
        - continuous: all-zero latent vector
        """
        if self._has_terminal_target_eos_block(sample):
            return sample

        discrete_target = sample["discrete_target"]
        continuous_target = sample["continuous_target"]

        eos_discrete = torch.full(
            (1, int(discrete_target.shape[1])),
            fill_value=self.eos_token_id,
            dtype=discrete_target.dtype,
            device=discrete_target.device,
        )
        eos_continuous = torch.zeros(
            (1, int(continuous_target.shape[1])),
            dtype=continuous_target.dtype,
            device=continuous_target.device,
        )

        extended = dict(sample)
        extended["discrete_target"] = torch.cat([discrete_target, eos_discrete], dim=0)
        extended["continuous_target"] = torch.cat([continuous_target, eos_continuous], dim=0)

        flow_timesteps = sample.get("flow_timesteps")
        if flow_timesteps is not None:
            extended["flow_timesteps"] = torch.cat(
                [flow_timesteps, flow_timesteps.new_zeros((1,))],
                dim=0,
            )

        noise = sample.get("noise")
        if noise is not None:
            eos_noise = noise.new_zeros((1, int(noise.shape[1])))
            extended["noise"] = torch.cat([noise, eos_noise], dim=0)

        return extended

    def _build_flat_attention_mask(self, sample: dict[str, torch.Tensor]) -> torch.BoolTensor:
        prompt_text = int(sample["text_prompt"].shape[0])
        prompt_speech = int(sample["discrete_prompt"].shape[0])
        target_text = int(sample["text_target"].shape[0])
        target_speech = int(sample["discrete_target"].shape[0])
        num_streams = int(sample["discrete_target"].shape[1]) + 1
        total = (
            prompt_text
            + 1
            + prompt_speech * num_streams
            + 1
            + target_text
            + 1
            + target_speech * num_streams
            + 1
        )
        return torch.ones(total, dtype=torch.bool)

    def _build_flat_sample(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Flatten one split Prism sample into:
        text_prompt -> EOT -> speech_prompt -> EOS -> text_target -> EOT -> speech_target -> EOS.
        `speech_target` is expected to already include the terminal EOS speech block.

        Returns core flat tensors plus a compact summary tensor:
        [text_prompt_start, text_prompt_end, speech_prompt_start, speech_prompt_end,
         text_target_start, text_target_end, speech_target_start, speech_target_end,
         sequence_length, num_discrete_streams,
         text_stream_idx, speech_discrete_stream_start_idx, speech_discrete_stream_end_idx,
         speech_continuous_stream_idx]
        where each *_start/*_end boundary is [start, end) in flattened token indices.
        """
        prompt_text = sample["text_prompt"].to(dtype=torch.long)
        prompt_discrete = sample["discrete_prompt"].to(dtype=torch.long)
        prompt_continuous = sample["continuous_prompt"].to(dtype=torch.float32)
        target_text = sample["text_target"].to(dtype=torch.long)
        target_discrete = sample["discrete_target"].to(dtype=torch.long)
        target_continuous = sample["continuous_target"].to(dtype=torch.float32)

        num_discrete_streams = int(prompt_discrete.shape[1])
        continuous_dim = int(prompt_continuous.shape[1])

        token_ids: list[int] = []
        token_type_ids: list[int] = []
        speech_stream_ids: list[int] = []
        target_block_ids: list[int] = []
        continuous_values: list[torch.Tensor] = []

        text_stream_idx = 0
        speech_discrete_stream_start_idx = 0
        speech_discrete_stream_end_idx = num_discrete_streams - 1
        speech_continuous_stream_idx = (
            num_discrete_streams
            if self.fixed_continuous_stream_idx is None
            else self.fixed_continuous_stream_idx
        )

        zero_cont = torch.zeros(continuous_dim, dtype=torch.float32)

        def append_text(token_id: int) -> None:
            token_ids.append(int(token_id))
            token_type_ids.append(TEXT_TOKEN_TYPE)
            speech_stream_ids.append(text_stream_idx)
            target_block_ids.append(-1)
            continuous_values.append(zero_cont)

        def append_discrete(token_id: int, stream_id: int, block_id: int) -> None:
            token_ids.append(int(token_id))
            token_type_ids.append(SPEECH_DISCRETE_TOKEN_TYPE)
            speech_stream_ids.append(int(stream_id))
            target_block_ids.append(int(block_id))
            continuous_values.append(zero_cont)

        def append_continuous(value: torch.Tensor, block_id: int) -> None:
            token_ids.append(self.pad_token_id)
            token_type_ids.append(SPEECH_CONTINUOUS_TOKEN_TYPE)
            speech_stream_ids.append(speech_continuous_stream_idx)
            target_block_ids.append(int(block_id))
            continuous_values.append(value)

        text_prompt_start = len(token_ids)
        for token in prompt_text.tolist():
            append_text(token)
        text_prompt_end = len(token_ids)
        append_text(self.eot_token_id)

        speech_prompt_start = len(token_ids)
        for block_idx in range(int(prompt_discrete.shape[0])):
            for stream_idx in range(num_discrete_streams):
                append_discrete(
                    int(prompt_discrete[block_idx, stream_idx].item()),
                    stream_idx,
                    -1,
                )
            append_continuous(prompt_continuous[block_idx], -1)
        speech_prompt_end = len(token_ids)
        append_text(self.eos_token_id)

        text_target_start = len(token_ids)
        for token in target_text.tolist():
            append_text(token)
        text_target_end = len(token_ids)
        append_text(self.eot_token_id)

        speech_target_start = len(token_ids)
        for block_idx in range(int(target_discrete.shape[0])):
            for stream_idx in range(num_discrete_streams):
                append_discrete(
                    int(target_discrete[block_idx, stream_idx].item()),
                    stream_idx,
                    block_idx,
                )
            append_continuous(target_continuous[block_idx], block_idx)
        speech_target_end = len(token_ids)
        append_text(self.eos_token_id)

        seq_len = len(token_ids)
        summary = torch.tensor(
            [
                text_prompt_start,
                text_prompt_end,
                speech_prompt_start,
                speech_prompt_end,
                text_target_start,
                text_target_end,
                speech_target_start,
                speech_target_end,
                seq_len,
                num_discrete_streams,
                text_stream_idx,
                speech_discrete_stream_start_idx,
                speech_discrete_stream_end_idx,
                speech_continuous_stream_idx,
            ],
            dtype=torch.long,
        )
        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "continuous_values": torch.stack(continuous_values, dim=0).to(dtype=torch.float32),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "speech_stream_ids": torch.tensor(speech_stream_ids, dtype=torch.long),
            "target_block_ids": torch.tensor(target_block_ids, dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.bool),
            "summary": summary,
        }

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
