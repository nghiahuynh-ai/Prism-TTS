from __future__ import annotations

import math
import os
import random
import re
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
from torch.utils.data import Sampler


def _estimate_text_token_count(text: str, char_to_id: Mapping[str, int], append_eos: bool) -> int:
    count = 0
    for char in text:
        if char in char_to_id:
            count += 1
    if append_eos:
        count += 1
    return max(1, count)


def _estimate_discrete_length(duration_seconds: float, codec_frame_rate_hz: float) -> int:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        return 1
    return max(1, int(round(duration_seconds * codec_frame_rate_hz)))


def _estimate_concat_sequence_length(
    text_target_length: int,
    speech_target_length: int,
) -> int:
    # Single-utterance continuous-only layout: text_target -> EOT -> [target frames].
    # One position per speech frame; no prompt prefix, no discrete streams.
    text_target_length = max(1, int(text_target_length))
    speech_target_length = max(1, int(speech_target_length))
    return (
        text_target_length
        + 1  # EOT after text target
        + speech_target_length
    )


def _safe_1d_length(value: Any, *, field_name: str) -> int:
    if value is None:
        raise ValueError(f"Missing required field {field_name!r} in sample.")
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) == 0:
            raise ValueError(f"{field_name} must be at least 1D.")
        return max(1, int(shape[0]))
    if isinstance(value, Sequence):
        return max(1, len(value))
    raise ValueError(f"Unsupported {field_name} value type: {type(value).__name__}.")


def _safe_2d_length(value: Any, *, field_name: str) -> int:
    if value is None:
        raise ValueError(f"Missing required field {field_name!r} in sample.")
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) < 2:
            raise ValueError(f"{field_name} must be at least 2D.")
        return max(1, int(shape[0]))
    if isinstance(value, Sequence):
        return max(1, len(value))
    raise ValueError(f"Unsupported {field_name} value type: {type(value).__name__}.")


def _compile_vocab_char_pattern(char_to_id: Mapping[str, int]) -> Optional[re.Pattern[str]]:
    """Build a character-class regex matching vocab chars, for C-speed counting."""
    chars = [c for c in char_to_id.keys() if isinstance(c, str) and len(c) == 1]
    if not chars or len(chars) != len(char_to_id):
        return None
    return re.compile("[" + re.escape("".join(chars)) + "]")


_LENGTH_CACHE_VERSION = 1


def _length_cache_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(manifest_path.name + ".prism-lengths.npz")


def _length_cache_disabled() -> bool:
    return os.environ.get("PRISM_TTS_DISABLE_LENGTH_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _load_cached_lengths(
    manifest_path: Path,
    *,
    manifest_stat: os.stat_result,
    codec_frame_rate_hz: float,
    append_eos: bool,
    vocab_size: int,
) -> Optional[list[int]]:
    cache_path = _length_cache_path(manifest_path)
    try:
        with np.load(cache_path) as data:
            if (
                int(data["version"][()]) != _LENGTH_CACHE_VERSION
                or int(data["manifest_size"][()]) != int(manifest_stat.st_size)
                or int(data["manifest_mtime_ns"][()]) != int(manifest_stat.st_mtime_ns)
                or float(data["codec_frame_rate_hz"][()]) != float(codec_frame_rate_hz)
                or bool(data["append_eos"][()]) != bool(append_eos)
                or int(data["vocab_size"][()]) != int(vocab_size)
            ):
                return None
            lengths = data["lengths"].astype(np.int64).tolist()
    except Exception:
        return None
    print(
        f"[adaptive_batching] Loaded {len(lengths):,} cached sample lengths "
        f"from {cache_path}.",
        flush=True,
    )
    return lengths


def _save_cached_lengths(
    manifest_path: Path,
    lengths: Sequence[int],
    *,
    manifest_stat: os.stat_result,
    codec_frame_rate_hz: float,
    append_eos: bool,
    vocab_size: int,
) -> None:
    cache_path = _length_cache_path(manifest_path)
    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez(
            tmp_path,
            version=_LENGTH_CACHE_VERSION,
            manifest_size=int(manifest_stat.st_size),
            manifest_mtime_ns=int(manifest_stat.st_mtime_ns),
            codec_frame_rate_hz=float(codec_frame_rate_hz),
            append_eos=bool(append_eos),
            vocab_size=int(vocab_size),
            lengths=np.asarray(lengths, dtype=np.int32),
        )
        os.replace(tmp_path, cache_path)
    except OSError as exc:
        print(
            f"[adaptive_batching] Could not cache sample lengths to {cache_path}: {exc}",
            flush=True,
        )
        return
    print(
        f"[adaptive_batching] Cached {len(lengths):,} sample lengths to {cache_path}.",
        flush=True,
    )


def _stream_estimate_lengths_from_manifest(
    manifest_path: Path,
    *,
    char_to_id: Mapping[str, int],
    append_eos: bool,
    codec_frame_rate_hz: float,
    read_buffer_bytes: int,
    progress_every: int,
) -> list[int]:
    """Estimate lengths in one sequential manifest read with a light line parse.

    Only `duration` and `transcript` are needed, so this avoids the per-line
    seek + full ManifestEntry parse of `iter_manifest_entries`, which takes
    many silent minutes on multi-million-line network manifests.
    """
    pattern = _compile_vocab_char_pattern(char_to_id)
    lengths: list[int] = []
    with manifest_path.open(
        "r",
        encoding="utf-8",
        buffering=max(8192, int(read_buffer_bytes)),
    ) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 3:
                raise ValueError(
                    f"Manifest line {line_number} in {manifest_path} has only "
                    f"{len(parts)} '|'-separated fields; length estimation needs "
                    "at least file_name|duration|transcript."
                )
            try:
                duration = float(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"Manifest line {line_number} in {manifest_path} has a "
                    f"non-numeric duration field: {parts[1]!r}."
                ) from exc
            # _parse_manifest_line strips every field; mirror it so estimated
            # text lengths match what the dataset will actually tokenize.
            transcript = parts[2].strip()
            if pattern is not None:
                text_len = len(pattern.findall(transcript))
            else:
                text_len = sum(1 for ch in transcript if ch in char_to_id)
            if append_eos:
                text_len += 1
            lengths.append(
                _estimate_concat_sequence_length(
                    text_target_length=max(1, text_len),
                    speech_target_length=_estimate_discrete_length(
                        duration, codec_frame_rate_hz
                    ),
                )
            )
            if progress_every > 0 and line_number % progress_every == 0:
                print(
                    f"[adaptive_batching] Estimated lengths for {line_number:,} "
                    f"manifest lines ({len(lengths):,} samples) from {manifest_path}.",
                    flush=True,
                )
    return lengths


def estimate_prism_sample_lengths(
    dataset: Any,
    *,
    codec_frame_rate_hz: float,
) -> list[int]:
    """Estimate concatenated sequence length for each Prism sample."""
    if codec_frame_rate_hz <= 0:
        raise ValueError("codec_frame_rate_hz must be > 0.")

    entries = getattr(dataset, "_entries", None)
    tokenizer = getattr(dataset, "tokenizer", None)
    if (
        isinstance(entries, Sequence)
        and len(entries) > 0
        and tokenizer is not None
        and hasattr(tokenizer, "char_to_id")
    ):
        char_to_id = getattr(tokenizer, "char_to_id")
        append_eos = bool(getattr(tokenizer, "append_eos", False))
        if not isinstance(char_to_id, Mapping):
            raise ValueError("dataset.tokenizer.char_to_id must be a mapping.")

        lengths: list[int] = []
        for entry in entries:
            text_target_len = _estimate_text_token_count(
                str(getattr(entry, "transcript")),
                char_to_id=char_to_id,
                append_eos=append_eos,
            )
            target_frame_len = _estimate_discrete_length(
                float(getattr(entry, "duration")),
                codec_frame_rate_hz,
            )
            lengths.append(
                _estimate_concat_sequence_length(
                    text_target_length=text_target_len,
                    speech_target_length=target_frame_len,
                )
            )
        return lengths

    manifest_path = getattr(dataset, "manifest_path", None)
    if (
        manifest_path is not None
        and tokenizer is not None
        and isinstance(getattr(tokenizer, "char_to_id", None), Mapping)
    ):
        manifest_path = Path(manifest_path)
        char_to_id = tokenizer.char_to_id
        append_eos = bool(getattr(tokenizer, "append_eos", False))
        expected_count: Optional[int] = None
        try:
            expected_count = len(dataset)
        except TypeError:
            pass

        manifest_stat = manifest_path.stat()
        cache_enabled = not _length_cache_disabled()
        if cache_enabled:
            cached = _load_cached_lengths(
                manifest_path,
                manifest_stat=manifest_stat,
                codec_frame_rate_hz=codec_frame_rate_hz,
                append_eos=append_eos,
                vocab_size=len(char_to_id),
            )
            if cached is not None and (
                expected_count is None or len(cached) == expected_count
            ):
                return cached

        lengths = _stream_estimate_lengths_from_manifest(
            manifest_path,
            char_to_id=char_to_id,
            append_eos=append_eos,
            codec_frame_rate_hz=codec_frame_rate_hz,
            read_buffer_bytes=int(
                getattr(dataset, "manifest_read_buffer_bytes", 4 * 1024 * 1024)
            ),
            progress_every=int(getattr(dataset, "manifest_progress_every", 0)),
        )
        if expected_count is not None and len(lengths) != expected_count:
            raise ValueError(
                f"Estimated {len(lengths)} sample lengths from {manifest_path} but the "
                f"dataset reports {expected_count} samples; the manifest appears to "
                "have changed since it was indexed."
            )
        if cache_enabled:
            _save_cached_lengths(
                manifest_path,
                lengths,
                manifest_stat=manifest_stat,
                codec_frame_rate_hz=codec_frame_rate_hz,
                append_eos=append_eos,
                vocab_size=len(char_to_id),
            )
        return lengths

    iter_manifest_entries = getattr(dataset, "iter_manifest_entries", None)
    if callable(iter_manifest_entries) and tokenizer is not None and hasattr(
        tokenizer, "char_to_id"
    ):
        char_to_id = getattr(tokenizer, "char_to_id")
        append_eos = bool(getattr(tokenizer, "append_eos", False))
        if not isinstance(char_to_id, Mapping):
            raise ValueError("dataset.tokenizer.char_to_id must be a mapping.")
        return [
            _estimate_concat_sequence_length(
                text_target_length=_estimate_text_token_count(
                    str(entry.transcript),
                    char_to_id=char_to_id,
                    append_eos=append_eos,
                ),
                speech_target_length=_estimate_discrete_length(
                    float(entry.duration),
                    codec_frame_rate_hz,
                ),
            )
            for entry in iter_manifest_entries()
        ]

    samples = getattr(dataset, "_samples", None)
    if isinstance(samples, Sequence) and len(samples) > 0:
        lengths = []
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise ValueError(
                    "Expected each in-memory dataset sample to be a mapping for adaptive batching."
                )

            text_target_len = _safe_1d_length(sample.get("text_target"), field_name="text_target")
            target_frame_len = _safe_2d_length(
                sample.get("continuous_target"),
                field_name="continuous_target",
            )
            lengths.append(
                _estimate_concat_sequence_length(
                    text_target_length=text_target_len,
                    speech_target_length=target_frame_len,
                )
            )
        return lengths

    raise ValueError(
        "Adaptive batching requires PrismDataset-backed samples "
        "(manifest entries or in-memory mapped samples)."
    )


class AdaptiveMemoryBatchSampler(Sampler[list[int]]):
    """
    Variable-size batch sampler targeting a memory budget proxy.

    Memory estimate per candidate batch:
        estimated_cost = batch_size * (max_sequence_length_in_batch ** 2)
    """

    def __init__(
        self,
        sample_lengths: Sequence[int],
        *,
        target_batch_cost: int,
        max_batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if not sample_lengths:
            raise ValueError("sample_lengths must not be empty.")
        if target_batch_cost < 1:
            raise ValueError("target_batch_cost must be >= 1.")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1.")
        if drop_last:
            raise ValueError(
                "drop_last=True is not allowed for AdaptiveMemoryBatchSampler because "
                "all training samples must be used."
            )

        # Compact typed storage: a Python list of ints costs ~36 bytes/sample
        # (and is copied into every forked DataLoader worker); at multi-million
        # sample manifests that is hundreds of MB.
        self.sample_lengths = array("I", (max(1, int(length)) for length in sample_lengths))
        self.target_batch_cost = int(target_batch_cost)
        self.max_batch_size = int(max_batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = False
        self.seed = int(seed)
        self._epoch = 0
        self._cached_len: Optional[int] = None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _ordered_indices(self) -> list[int]:
        indices = list(range(len(self.sample_lengths)))
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(indices)
        self._epoch += 1
        return indices

    def _build_batches(self, ordered_indices: Sequence[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        cursor = 0
        total = len(ordered_indices)
        while cursor < total:
            batch: list[int] = []
            batch_max_len = 0

            while cursor < total and len(batch) < self.max_batch_size:
                sample_idx = int(ordered_indices[cursor])
                sample_len = self.sample_lengths[sample_idx]

                next_batch_size = len(batch) + 1
                next_max_len = max(batch_max_len, sample_len)
                next_cost = next_batch_size * (next_max_len**2)
                if batch and next_cost > self.target_batch_cost:
                    break

                batch.append(sample_idx)
                batch_max_len = next_max_len
                cursor += 1

            if not batch:
                batch = [int(ordered_indices[cursor])]
                cursor += 1

            batches.append(batch)

        return batches

    def __iter__(self):
        ordered = self._ordered_indices()
        for batch in self._build_batches(ordered):
            yield batch

    def __len__(self) -> int:
        # Deterministic (unshuffled order), and Lightning queries it repeatedly;
        # rebuilding every batch per call is an O(dataset) pass each time.
        if self._cached_len is None:
            self._cached_len = len(self._build_batches(range(len(self.sample_lengths))))
        return self._cached_len
