from __future__ import annotations

import os
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
)

DEFAULT_DISCRETE_TOKEN_COUNT = 2048
TEXT_TOKEN_TYPE = 0
# Speech is modeled as a single continuous latent per frame (no vector quantization).
SPEECH_TOKEN_TYPE = 1


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
        continuous_feature_dim: int | None = None,
        append_eos_to_text: bool = False,
        cache_npy: bool = False,
        cache_npz: bool | None = None,
        load_prompt: bool = False,
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
        # The single-utterance layout does not consume the prompt npy during
        # training/eval, so it is skipped by default (halves per-sample disk I/O).
        # Enable only if a cross-utterance prompt is genuinely needed.
        self.load_prompt = bool(load_prompt)
        self._npy_cache: dict[str, torch.FloatTensor] = {}
        self.continuous_feature_dim = (
            None if continuous_feature_dim is None else int(continuous_feature_dim)
        )
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

        if path.suffix.lower() != ".npy":
            raise ValueError(f"{field_name} at line {line_number} must be a .npy path: {path}")
        # NOTE: do NOT `.resolve()` or `.is_file()` here. On networked filesystems these
        # each cost a round-trip, and manifests can have tens of thousands of entries
        # (target + prompt per line), making dataset construction dominate startup time.
        # Path normalization is cheap and stat-free; existence is validated lazily when
        # the file is actually loaded (see `_load_continuous_features`).
        return Path(os.path.normpath(path))

    @staticmethod
    def _extract_continuous_from_npy(payload: Any, npy_path: Path) -> Any:
        """Return the continuous latent array from a feature .npy (discrete ignored).

        Supported formats: a plain float array [L, D]; a structured array with a
        'continuous' field; or a pickled dict/tuple payload (legacy tuples are
        (discrete, continuous)). Any discrete stream present is not used.
        """
        if isinstance(payload, np.ndarray) and payload.dtype.names is not None:
            if "continuous" in set(payload.dtype.names):
                continuous = payload["continuous"]
                if (
                    isinstance(continuous, np.ndarray)
                    and continuous.dtype == object
                    and continuous.shape == ()
                ):
                    continuous = continuous.item()
                return continuous

        if isinstance(payload, np.ndarray) and payload.dtype == object and payload.shape == ():
            value = payload.item()
            if isinstance(value, Mapping) and "continuous" in value:
                return value["continuous"]
            if isinstance(value, tuple) and len(value) == 2:
                return value[1]  # legacy (discrete, continuous)
            return value

        if isinstance(payload, np.ndarray) and payload.dtype != object:
            return payload

        raise ValueError(
            f"{npy_path} must contain a 'continuous' array. Supported .npy formats: "
            "a plain float array [L, D], a structured array with a 'continuous' field, "
            "or a pickled dict/tuple payload."
        )

    def _load_continuous_features(self, npy_path: Path) -> torch.FloatTensor:
        cache_key = str(npy_path)
        if self.cache_npy and cache_key in self._npy_cache:
            return self._npy_cache[cache_key].clone()

        try:
            try:
                payload = np.load(npy_path, allow_pickle=False)
            except ValueError as exc:
                if "allow_pickle=False" not in str(exc):
                    raise
                payload = np.load(npy_path, allow_pickle=True)

            continuous_raw = self._extract_continuous_from_npy(payload, npy_path)
            continuous = _to_float_2d(continuous_raw, f"{npy_path}:continuous")
        except Exception as exc:
            raise RuntimeError(f"Failed to load npy file {npy_path}: {exc}") from exc

        if (
            self.continuous_feature_dim is not None
            and continuous.shape[1] != self.continuous_feature_dim
        ):
            raise ValueError(
                f"{npy_path} continuous shape mismatch: expected [L, {self.continuous_feature_dim}], "
                f"got {tuple(continuous.shape)}."
            )

        if self.cache_npy:
            self._npy_cache[cache_key] = continuous
            return continuous.clone()
        return continuous

    def _encode_text(self, text: str, field_name: str) -> torch.LongTensor:
        try:
            encoded = self.tokenizer(text)
        except Exception as exc:
            raise RuntimeError(f"tokenization failed for {field_name}: {exc}") from exc
        return _to_long_1d(encoded, field_name)

    def _build_manifest_sample(self, entry: ManifestEntry) -> dict[str, torch.Tensor]:
        continuous_target = self._load_continuous_features(entry.target_npy_path)
        text_target = self._encode_text(entry.transcript, "transcript")

        if self.load_prompt:
            continuous_prompt = self._load_continuous_features(entry.prompt_npy_path)
            text_prompt = self._encode_text(entry.prompt_transcript, "prompt_transcript")
        else:
            # Prompt is unused by the single-utterance layout: skip its npy read and
            # emit empty placeholders (kept for collate/schema compatibility).
            continuous_prompt = continuous_target.new_zeros((0, continuous_target.shape[1]))
            text_prompt = text_target.new_zeros((0,))

        return _normalize_split_sample(
            {
                "text_target": text_target,
                "continuous_target": continuous_target,
                "text_prompt": text_prompt,
                "continuous_prompt": continuous_prompt,
            }
        )


class BatchCollate:
    """Pad Prism-TTS samples and flatten each into a single-utterance continuous sequence."""

    def __init__(
        self,
        text_pad_value: int | None = None,
        continuous_pad_value: float = 0.0,
        include_attention_mask: bool = True,
        discrete_token_count: int = DEFAULT_DISCRETE_TOKEN_COUNT,
    ) -> None:
        # `discrete_token_count` only fixes the shared token-id layout (EOT/EOS/PAD/
        # text offset); no discrete speech stream is emitted in this branch.
        (
            self.eot_token_id,
            self.eos_token_id,
            self.pad_token_id,
            _,
        ) = build_shared_token_layout(int(discrete_token_count))
        self.text_pad_value = self.pad_token_id if text_pad_value is None else int(text_pad_value)
        self.continuous_pad_value = continuous_pad_value
        self.include_attention_mask = include_attention_mask
        self.discrete_token_count = int(discrete_token_count)

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("BatchCollate received an empty batch.")

        samples = [self._validate_collate_sample(item) for item in batch]
        text_prompt_lengths = torch.tensor(
            [int(sample["text_prompt"].shape[0]) for sample in samples],
            dtype=torch.long,
        )
        speech_prompt_lengths = torch.tensor(
            [int(sample["continuous_prompt"].shape[0]) for sample in samples],
            dtype=torch.long,
        )
        text_target_lengths = torch.tensor(
            [int(sample["text_target"].shape[0]) for sample in samples],
            dtype=torch.long,
        )
        speech_target_lengths = torch.tensor(
            [int(sample["continuous_target"].shape[0]) for sample in samples],
            dtype=torch.long,
        )

        collated: dict[str, torch.Tensor] = {
            "text_prompt": _pad_1d([sample["text_prompt"] for sample in samples], self.text_pad_value),
            "continuous_prompt": _pad_2d(
                [sample["continuous_prompt"] for sample in samples],
                self.continuous_pad_value,
            ),
            "text_target": _pad_1d([sample["text_target"] for sample in samples], self.text_pad_value),
            "continuous_target": _pad_2d(
                [sample["continuous_target"] for sample in samples],
                self.continuous_pad_value,
            ),
            "text_prompt_lengths": text_prompt_lengths,
            "speech_prompt_lengths": speech_prompt_lengths,
            "text_target_lengths": text_target_lengths,
            "speech_target_lengths": speech_target_lengths,
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
        collated["flat_target_block_ids"] = _pad_1d(
            [item["target_block_ids"] for item in flat_per_sample],
            -1,
        )
        collated["flat_continuous_values"] = _pad_2d(
            [item["continuous_values"] for item in flat_per_sample],
            0.0,
        )
        collated["flat_target_block_counts"] = speech_target_lengths

        if self.include_attention_mask:
            collated["attention_mask"] = _pad_1d(
                [item["attention_mask"] for item in flat_per_sample],
                False,
            ).to(dtype=torch.bool)

        return collated

    def _validate_collate_sample(self, sample: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        required_keys = (
            "text_target",
            "continuous_target",
            "text_prompt",
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
        continuous_target = require_tensor("continuous_target", 2)
        text_prompt = require_tensor("text_prompt", 1)
        continuous_prompt = require_tensor("continuous_prompt", 2)

        if continuous_target.shape[1] != continuous_prompt.shape[1]:
            raise ValueError(
                "continuous_target and continuous_prompt must have the same channel size."
            )

        normalized: dict[str, torch.Tensor] = {
            "text_target": text_target,
            "continuous_target": continuous_target,
            "text_prompt": text_prompt,
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

    def _build_flat_sample(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Flatten one split Prism sample into a single-utterance continuous sequence:
        text_target -> EOT -> [target frames].

        Each speech frame is a single position carrying a continuous latent. Every
        frame is a target (`target_block_ids` = 0..L-1 at speech positions, -1 at
        text/EOT); the first fraction of frames is used as an in-context condition
        via masking in the model. There is no prompt prefix and no literal EOS
        token (end-of-sequence is learned by the model's EOS head).
        """
        target_text = sample["text_target"].to(dtype=torch.long)
        target_continuous = sample["continuous_target"].to(dtype=torch.float32)

        continuous_dim = int(target_continuous.shape[1])

        token_ids: list[int] = []
        token_type_ids: list[int] = []
        target_block_ids: list[int] = []
        continuous_values: list[torch.Tensor] = []

        zero_cont = torch.zeros(continuous_dim, dtype=torch.float32)

        def append_text(token_id: int) -> None:
            token_ids.append(int(token_id))
            token_type_ids.append(TEXT_TOKEN_TYPE)
            target_block_ids.append(-1)
            continuous_values.append(zero_cont)

        def append_speech(value: torch.Tensor, block_id: int) -> None:
            token_ids.append(self.pad_token_id)
            token_type_ids.append(SPEECH_TOKEN_TYPE)
            target_block_ids.append(int(block_id))
            continuous_values.append(value)

        for token in target_text.tolist():
            append_text(token)
        append_text(self.eot_token_id)

        for block_idx in range(int(target_continuous.shape[0])):
            append_speech(target_continuous[block_idx], block_idx)

        seq_len = len(token_ids)
        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "continuous_values": torch.stack(continuous_values, dim=0).to(dtype=torch.float32),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_block_ids": torch.tensor(target_block_ids, dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.bool),
        }
