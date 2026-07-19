from __future__ import annotations

import os
import random
import warnings
from array import array
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

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
        index_manifest: bool = False,
        manifest_progress_every: int = 0,
        manifest_read_buffer_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        self.discrete_token_count = int(discrete_token_count)
        (
            self.eot_token_id,
            self.eos_token_id,
            self.pad_token_id,
            self.text_token_offset,
        ) = build_shared_token_layout(self.discrete_token_count)

        resolved_vocab_path = (
            self._absolute_path(vocab_path)
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
        self.manifest_progress_every = int(manifest_progress_every)
        if self.manifest_progress_every < 0:
            raise ValueError("manifest_progress_every must be >= 0.")
        self.manifest_read_buffer_bytes = int(manifest_read_buffer_bytes)
        if self.manifest_read_buffer_bytes < 1:
            raise ValueError("manifest_read_buffer_bytes must be >= 1.")
        self._npy_cache: dict[str, torch.FloatTensor] = {}
        # Remember whether npy files need allow_pickle so we stop paying a failed
        # load attempt per file (each retry is an extra open() on networked FS).
        self._npy_allow_pickle: bool | None = None
        self.continuous_feature_dim = (
            None if continuous_feature_dim is None else int(continuous_feature_dim)
        )
        if self.continuous_feature_dim is not None and self.continuous_feature_dim < 1:
            raise ValueError("continuous_feature_dim must be >= 1 when provided.")

        self._entries: list[ManifestEntry] = []
        self._samples: list[Mapping[str, Any]] = []
        self._manifest_line_offsets = array("Q")
        self._manifest_line_numbers = array("I")
        self._manifest_handle = None
        self._manifest_handle_pid: int | None = None
        self.index_manifest = bool(index_manifest)

        if isinstance(source, (str, Path)):
            # Keep manifest setup stat-free until it is opened. ``Path.resolve``
            # and ``is_file`` can each trigger a network metadata round trip on
            # NFS/Lustre, which makes an unavailable mount look like a hang.
            manifest_path = self._absolute_path(source)
            self.manifest_path = manifest_path
            self.manifest_root = (
                self._absolute_path(manifest_root)
                if manifest_root is not None
                else manifest_path.parent
            )
            if self.index_manifest:
                self._build_manifest_index(manifest_path)
            else:
                self._entries = self._load_manifest(manifest_path)
        else:
            self.manifest_path = None
            self.manifest_root = (
                self._absolute_path(manifest_root) if manifest_root is not None else None
            )
            self._samples = list(source)

    @staticmethod
    def _absolute_path(value: str | Path) -> Path:
        """Make an absolute path without resolving symlinks or touching the filesystem."""
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return Path(os.path.abspath(path))

    def __len__(self) -> int:
        if self.manifest_path is not None and self.index_manifest:
            return len(self._manifest_line_offsets)
        return len(self._entries) if self._entries else len(self._samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.manifest_path is not None and self.index_manifest:
            manifest_index = self._normalize_manifest_index(index)
            return self._build_manifest_sample(self._manifest_entry_at(manifest_index))
        if self._entries:
            return self._build_manifest_sample(self._entries[index])
        return _normalize_split_sample(self._samples[index])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_manifest_handle"] = None
        state["_manifest_handle_pid"] = None
        return state

    def __del__(self) -> None:
        if hasattr(self, "_manifest_handle"):
            self._close_manifest_handle()

    def _close_manifest_handle(self) -> None:
        if self._manifest_handle is not None:
            self._manifest_handle.close()
            self._manifest_handle = None
            self._manifest_handle_pid = None

    def _build_manifest_index(self, manifest_path: Path) -> None:
        self._manifest_line_offsets = array("Q")
        self._manifest_line_numbers = array("I")
        line_number = 0
        try:
            with manifest_path.open(
                "rb",
                buffering=self.manifest_read_buffer_bytes,
            ) as handle:
                while True:
                    offset = handle.tell()
                    raw_line = handle.readline()
                    if not raw_line:
                        break
                    line_number += 1
                    line = raw_line.strip()
                    if not line or line.startswith(b"#"):
                        continue
                    self._manifest_line_offsets.append(offset)
                    self._manifest_line_numbers.append(line_number)
                    if (
                        self.manifest_progress_every > 0
                        and line_number % self.manifest_progress_every == 0
                    ):
                        print(
                            f"[PrismDataset] Indexed {line_number:,} lines from "
                            f"{manifest_path} "
                            f"({len(self._manifest_line_offsets):,} samples).",
                            flush=True,
                        )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}") from exc
        except IsADirectoryError as exc:
            raise ValueError(f"Manifest path is a directory, not a file: {manifest_path}") from exc
        if not self._manifest_line_offsets:
            raise ValueError(f"Manifest has no valid entries: {manifest_path}")

    def _normalize_manifest_index(self, index: int) -> int:
        index = int(index)
        manifest_length = len(self._manifest_line_offsets)
        if index < 0:
            index += manifest_length
        if index < 0 or index >= manifest_length:
            raise IndexError(f"Manifest index out of range: {index}")
        return index

    def _ensure_manifest_handle(self):
        if self.manifest_path is None:
            raise RuntimeError("Manifest handle requested for an in-memory dataset.")
        current_pid = os.getpid()
        if self._manifest_handle is None or self._manifest_handle_pid != current_pid:
            self._close_manifest_handle()
            self._manifest_handle = self.manifest_path.open(
                "rb",
                buffering=self.manifest_read_buffer_bytes,
            )
            self._manifest_handle_pid = current_pid
        return self._manifest_handle

    def _manifest_entry_at(self, manifest_index: int) -> ManifestEntry:
        handle = self._ensure_manifest_handle()
        offset = int(self._manifest_line_offsets[manifest_index])
        line_number = int(self._manifest_line_numbers[manifest_index])
        handle.seek(offset)
        raw_line = handle.readline()
        if not raw_line:
            raise RuntimeError(
                f"Unexpected EOF at manifest sample {manifest_index} "
                f"(line {line_number}) in {self.manifest_path}."
            )
        try:
            line = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Manifest line {line_number} is not valid UTF-8 in {self.manifest_path}."
            ) from exc
        if not line or line.startswith("#"):
            raise RuntimeError(
                f"Manifest index drift at line {line_number} in {self.manifest_path}."
            )
        return self._parse_manifest_line(line, line_number)

    def iter_manifest_entries(self) -> Iterator[ManifestEntry]:
        if self.manifest_path is None:
            return
        if self.index_manifest:
            for manifest_index in range(len(self._manifest_line_offsets)):
                yield self._manifest_entry_at(manifest_index)
            return
        yield from self._entries

    def _load_manifest(self, manifest_path: Path) -> list[ManifestEntry]:
        entries: list[ManifestEntry] = []
        try:
            # The default 8 KiB TextIO buffer can turn a 45k-row manifest into
            # thousands of high-latency NFS reads. A multi-megabyte buffer keeps
            # parsing streaming and bounded in memory while reducing network
            # round trips by orders of magnitude.
            with manifest_path.open(
                "r",
                encoding="utf-8",
                buffering=self.manifest_read_buffer_bytes,
            ) as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    entries.append(self._parse_manifest_line(line, line_number))
                    if (
                        self.manifest_progress_every > 0
                        and line_number % self.manifest_progress_every == 0
                    ):
                        print(
                            f"[PrismDataset] Parsed {line_number:,} lines from "
                            f"{manifest_path} ({len(entries):,} samples).",
                            flush=True,
                        )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}") from exc
        except IsADirectoryError as exc:
            raise ValueError(f"Manifest path is a directory, not a file: {manifest_path}") from exc
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
            if self._npy_allow_pickle is None:
                try:
                    payload = np.load(npy_path, allow_pickle=False)
                    self._npy_allow_pickle = False
                except ValueError as exc:
                    if "allow_pickle=False" not in str(exc):
                        raise
                    payload = np.load(npy_path, allow_pickle=True)
                    self._npy_allow_pickle = True
            else:
                payload = np.load(npy_path, allow_pickle=self._npy_allow_pickle)

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


class LazyPrismDataset(PrismDataset, IterableDataset[dict[str, torch.Tensor]]):
    """Stream a manifest lazily without parsing all metadata at startup.

    Every DataLoader worker reads a disjoint byte range of the manifest. This
    avoids opening, reading, and parsing the full metadata file in the main
    process before model construction. Train shuffling is provided by a bounded
    reservoir-style buffer, so it never needs the full manifest in memory.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        shuffle_manifest: bool = False,
        shuffle_buffer_size: int = 256,
        shuffle_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(source, (str, Path)):
            raise TypeError("LazyPrismDataset source must be a manifest path.")
        if shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be >= 1.")

        manifest_root = kwargs.get("manifest_root")
        # Reuse PrismDataset's tokenizer and sample construction without asking
        # it to parse the supplied manifest eagerly.
        super().__init__(source=[], **kwargs)
        self.manifest_path = self._absolute_path(source)
        self.manifest_root = (
            self._absolute_path(manifest_root)
            if manifest_root is not None
            else self.manifest_path.parent
        )
        self.shuffle_manifest = bool(shuffle_manifest)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.shuffle_seed = None if shuffle_seed is None else int(shuffle_seed)
        self._stream_epoch = 0

    def __len__(self) -> int:
        # Returning an invented length would make Lightning stop an iterable
        # stream early. Trainer.max_steps controls training length instead.
        raise TypeError("LazyPrismDataset has no eager manifest length.")

    @staticmethod
    def _distributed_shard() -> tuple[int, int]:
        """Return the process rank/world size without requiring torch.distributed."""
        try:
            rank = int(os.environ.get("RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError:
            return 0, 1
        if rank < 0 or world_size < 1 or rank >= world_size:
            return 0, 1
        return rank, world_size

    def _iter_manifest_entries(
        self,
        *,
        shard_index: int,
        shard_count: int,
    ) -> Iterator[ManifestEntry]:
        assert self.manifest_path is not None
        try:
            file_size = self.manifest_path.stat().st_size
            start = (file_size * shard_index) // shard_count
            end = (file_size * (shard_index + 1)) // shard_count

            with self.manifest_path.open(
                "rb",
                buffering=self.manifest_read_buffer_bytes,
            ) as handle:
                if start > 0:
                    # Discard the partial line containing the shard boundary;
                    # the following line is owned wholly by this shard.
                    handle.seek(start - 1)
                    handle.readline()

                while True:
                    line_offset = handle.tell()
                    if line_offset >= end:
                        break
                    raw_line = handle.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8").strip()
                    if not line or line.startswith("#"):
                        continue
                    # The original line number is intentionally not computed:
                    # doing so would require scanning all preceding metadata.
                    yield self._parse_manifest_line(line, line_offset)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}") from exc
        except IsADirectoryError as exc:
            raise ValueError(
                f"Manifest path is a directory, not a file: {self.manifest_path}"
            ) from exc

    def _shuffle_entries(
        self,
        entries: Iterator[ManifestEntry],
        *,
        seed: int,
    ) -> Iterator[ManifestEntry]:
        if not self.shuffle_manifest or self.shuffle_buffer_size == 1:
            yield from entries
            return

        rng = random.Random(seed)
        buffer: list[ManifestEntry] = []
        for entry in entries:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(entry)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = entry

        while buffer:
            index = rng.randrange(len(buffer))
            yield buffer.pop(index)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        rank, world_size = self._distributed_shard()
        shard_count = world_size * worker_count
        shard_index = rank * worker_count + worker_id
        epoch = self._stream_epoch
        self._stream_epoch += 1

        if rank == 0 and worker_id == 0:
            print(
                "[LazyPrismDataset] Streaming metadata lazily "
                f"across {shard_count} shard(s), shuffle_buffer_size="
                f"{self.shuffle_buffer_size if self.shuffle_manifest else 0}.",
                flush=True,
            )

        seed_base = self.shuffle_seed
        if seed_base is None:
            seed_base = 0 if worker is None else int(worker.seed)
        entries = self._iter_manifest_entries(
            shard_index=shard_index,
            shard_count=shard_count,
        )
        shuffled_entries = self._shuffle_entries(
            entries,
            seed=seed_base + shard_index + epoch * shard_count,
        )
        for entry in shuffled_entries:
            yield self._build_manifest_sample(entry)


class BatchCollate:
    """Pad Prism-TTS samples and flatten each into a single-utterance continuous sequence."""

    def __init__(
        self,
        text_pad_value: int | None = None,
        continuous_pad_value: float = 0.0,
        include_attention_mask: bool = True,
        discrete_token_count: int = DEFAULT_DISCRETE_TOKEN_COUNT,
        flat_sequence_length_multiple: int = 1,
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
        self.flat_sequence_length_multiple = int(flat_sequence_length_multiple)
        if self.flat_sequence_length_multiple < 1:
            raise ValueError("flat_sequence_length_multiple must be >= 1.")

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
            pad_to_multiple=self.flat_sequence_length_multiple,
        )
        collated["flat_token_type_ids"] = _pad_1d(
            [item["token_type_ids"] for item in flat_per_sample],
            TEXT_TOKEN_TYPE,
            pad_to_multiple=self.flat_sequence_length_multiple,
        )
        collated["flat_target_block_ids"] = _pad_1d(
            [item["target_block_ids"] for item in flat_per_sample],
            -1,
            pad_to_multiple=self.flat_sequence_length_multiple,
        )
        collated["flat_continuous_values"] = _pad_2d(
            [item["continuous_values"] for item in flat_per_sample],
            0.0,
            pad_to_multiple=self.flat_sequence_length_multiple,
        )
        collated["flat_target_block_counts"] = speech_target_lengths

        if self.include_attention_mask:
            collated["attention_mask"] = _pad_1d(
                [item["attention_mask"] for item in flat_per_sample],
                False,
                pad_to_multiple=self.flat_sequence_length_multiple,
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
        num_text = int(target_text.shape[0])
        num_frames = int(target_continuous.shape[0])
        prefix_len = num_text + 1  # text tokens + EOT
        seq_len = prefix_len + num_frames

        # Vectorized layout (avoids a per-frame Python loop for long utterances):
        # [text_target ... EOT] then [target frames]. Text/EOT positions carry the
        # pad token id and zero latents; frame positions carry pad token id + latent.
        token_ids = torch.full((seq_len,), self.pad_token_id, dtype=torch.long)
        token_ids[:num_text] = target_text
        token_ids[num_text] = self.eot_token_id

        token_type_ids = torch.full((seq_len,), SPEECH_TOKEN_TYPE, dtype=torch.long)
        token_type_ids[:prefix_len] = TEXT_TOKEN_TYPE

        target_block_ids = torch.full((seq_len,), -1, dtype=torch.long)
        if num_frames > 0:
            target_block_ids[prefix_len:] = torch.arange(num_frames, dtype=torch.long)

        continuous_values = torch.zeros(seq_len, continuous_dim, dtype=torch.float32)
        if num_frames > 0:
            continuous_values[prefix_len:] = target_continuous

        return {
            "token_ids": token_ids,
            "continuous_values": continuous_values,
            "token_type_ids": token_type_ids,
            "target_block_ids": target_block_ids,
            "attention_mask": torch.ones(seq_len, dtype=torch.bool),
        }
