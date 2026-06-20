from __future__ import annotations

import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Union

from torch.utils.data import Sampler

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None

_PARALLEL_CHAR_VOCAB: frozenset[str] | None = None
_PARALLEL_APPEND_EOS = False
_PARALLEL_CODEC_FRAME_RATE_HZ = 0.0
_PARALLEL_NUM_DISCRETE_STREAMS = 1


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
    text_prompt_length: int,
    speech_prompt_length: int,
    text_target_length: int,
    speech_target_length: int,
    num_discrete_streams: int,
) -> int:
    text_prompt_length = max(1, int(text_prompt_length))
    speech_prompt_length = max(1, int(speech_prompt_length))
    text_target_length = max(1, int(text_target_length))
    # Training collate appends one explicit terminal EOS speech block.
    speech_target_length = max(1, int(speech_target_length)) + 1
    num_discrete_streams = max(1, int(num_discrete_streams))
    speech_block_size = num_discrete_streams + 1
    return (
        text_prompt_length
        + 1  # EOT after text prompt
        + speech_prompt_length * speech_block_size
        + 1  # EOS after speech prompt
        + text_target_length
        + 1  # EOT after text target
        + speech_target_length * speech_block_size
        + 1  # EOS after speech target
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


def _iter_manifest_entries(dataset: Any):
    iterator_fn = getattr(dataset, "iter_manifest_entries", None)
    if callable(iterator_fn):
        entries_iter = iterator_fn()
        if entries_iter is not None:
            return entries_iter

    entries = getattr(dataset, "_entries", None)
    if isinstance(entries, Sequence) and len(entries) > 0:
        return iter(entries)

    return None


def _progress(
    iterable: Iterable[Any],
    *,
    total: int | None,
    desc: str,
    unit: str,
):
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=False,
    )


def _parse_env_int_silent(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(minimum, int(default))
    try:
        value = int(raw)
    except ValueError:
        return max(minimum, int(default))
    return max(minimum, value)


def _manifest_length_fields_from_line(raw_line: str, *, line_number: int) -> tuple[float, str, float, str]:
    line = raw_line.strip()
    parts = [part.strip() for part in line.split("|")]
    # Some manifests include a trailing delimiter, producing an empty final field.
    while parts and parts[-1] == "":
        parts.pop()
    if len(parts) != 8:
        raise ValueError(
            "Each manifest line must have exactly 8 fields separated by '|'. "
            f"line={line_number}, fields={len(parts)}"
        )

    try:
        duration = float(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid duration at line {line_number}: {parts[1]!r}.") from exc
    try:
        prompt_duration = float(parts[5])
    except ValueError as exc:
        raise ValueError(f"Invalid prompt_duration at line {line_number}: {parts[5]!r}.") from exc

    transcript = parts[2]
    prompt_transcript = parts[6]
    return duration, transcript, prompt_duration, prompt_transcript


def _iter_manifest_line_chunks(
    manifest_path: Path,
    *,
    chunk_size: int,
):
    chunk: list[tuple[int, str]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            chunk.append((line_number, line))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def _parallel_manifest_length_worker_init(
    char_vocab: frozenset[str],
    append_eos: bool,
    codec_frame_rate_hz: float,
    num_discrete_streams: int,
) -> None:
    global _PARALLEL_CHAR_VOCAB
    global _PARALLEL_APPEND_EOS
    global _PARALLEL_CODEC_FRAME_RATE_HZ
    global _PARALLEL_NUM_DISCRETE_STREAMS
    _PARALLEL_CHAR_VOCAB = char_vocab
    _PARALLEL_APPEND_EOS = bool(append_eos)
    _PARALLEL_CODEC_FRAME_RATE_HZ = float(codec_frame_rate_hz)
    _PARALLEL_NUM_DISCRETE_STREAMS = max(1, int(num_discrete_streams))


def _estimate_lengths_for_manifest_chunk(
    chunk: Sequence[tuple[int, str]],
) -> list[int]:
    if _PARALLEL_CHAR_VOCAB is None:
        raise RuntimeError("Parallel manifest length worker is uninitialized.")

    char_vocab = _PARALLEL_CHAR_VOCAB
    append_eos = _PARALLEL_APPEND_EOS
    codec_frame_rate_hz = _PARALLEL_CODEC_FRAME_RATE_HZ
    num_discrete_streams = _PARALLEL_NUM_DISCRETE_STREAMS

    lengths: list[int] = []
    for line_number, raw_line in chunk:
        duration, transcript, prompt_duration, prompt_transcript = _manifest_length_fields_from_line(
            raw_line,
            line_number=line_number,
        )

        text_prompt_len = 0
        for char in prompt_transcript:
            if char in char_vocab:
                text_prompt_len += 1
        if append_eos:
            text_prompt_len += 1
        text_prompt_len = max(1, text_prompt_len)

        text_target_len = 0
        for char in transcript:
            if char in char_vocab:
                text_target_len += 1
        if append_eos:
            text_target_len += 1
        text_target_len = max(1, text_target_len)

        prompt_discrete_len = _estimate_discrete_length(prompt_duration, codec_frame_rate_hz)
        target_discrete_len = _estimate_discrete_length(duration, codec_frame_rate_hz)
        lengths.append(
            _estimate_concat_sequence_length(
                text_prompt_length=text_prompt_len,
                speech_prompt_length=prompt_discrete_len,
                text_target_length=text_target_len,
                speech_target_length=target_discrete_len,
                num_discrete_streams=num_discrete_streams,
            )
        )
    return lengths


def _estimate_manifest_lengths_from_file(
    *,
    manifest_path: Path,
    char_to_id: Mapping[str, int],
    append_eos: bool,
    codec_frame_rate_hz: float,
    num_discrete_streams: int,
    total_entries: int | None,
) -> list[int]:
    chunk_size = _parse_env_int_silent("PRISM_TTS_ADAPTIVE_LENGTH_CHUNK_SIZE", 4096, minimum=64)
    default_workers = min(16, max(1, os.cpu_count() or 1))
    requested_workers = _parse_env_int_silent(
        "PRISM_TTS_ADAPTIVE_LENGTH_WORKERS",
        default_workers,
        minimum=1,
    )
    min_parallel_samples = _parse_env_int_silent(
        "PRISM_TTS_ADAPTIVE_LENGTH_MIN_PARALLEL_SAMPLES",
        50_000,
        minimum=1,
    )
    should_parallelize = (
        requested_workers > 1
        and total_entries is not None
        and total_entries >= min_parallel_samples
    )

    char_vocab = frozenset(char_to_id.keys())
    total_chunks = None
    if total_entries is not None:
        total_chunks = max(1, int(math.ceil(total_entries / float(chunk_size))))

    def _run_sequential() -> list[int]:
        _parallel_manifest_length_worker_init(
            char_vocab,
            append_eos,
            codec_frame_rate_hz,
            num_discrete_streams,
        )
        seq_lengths: list[int] = []
        for chunk_lengths in _progress(
            (
                _estimate_lengths_for_manifest_chunk(chunk)
                for chunk in _iter_manifest_line_chunks(manifest_path, chunk_size=chunk_size)
            ),
            total=total_chunks,
            desc="adaptive_batching: estimating lengths",
            unit="chunk",
        ):
            seq_lengths.extend(chunk_lengths)
        return seq_lengths

    if not should_parallelize:
        lengths = _run_sequential()
    else:
        try:
            lengths = []
            with ProcessPoolExecutor(
                max_workers=requested_workers,
                initializer=_parallel_manifest_length_worker_init,
                initargs=(
                    char_vocab,
                    append_eos,
                    codec_frame_rate_hz,
                    num_discrete_streams,
                ),
            ) as pool:
                for chunk_lengths in _progress(
                    pool.map(
                        _estimate_lengths_for_manifest_chunk,
                        _iter_manifest_line_chunks(manifest_path, chunk_size=chunk_size),
                        chunksize=1,
                    ),
                    total=total_chunks,
                    desc="adaptive_batching: estimating lengths",
                    unit="chunk",
                ):
                    lengths.extend(chunk_lengths)
        except (OSError, PermissionError):
            lengths = _run_sequential()

    if total_entries is not None and len(lengths) != total_entries:
        raise ValueError(
            "Adaptive length estimation produced a sample count mismatch: "
            f"expected {total_entries}, got {len(lengths)}."
        )
    return lengths


def estimate_prism_sample_lengths(
    dataset: Any,
    *,
    codec_frame_rate_hz: float,
    num_discrete_streams_override: int | None = None,
) -> list[int]:
    """Estimate concatenated sequence length for each Prism sample."""
    if codec_frame_rate_hz <= 0:
        raise ValueError("codec_frame_rate_hz must be > 0.")
    if num_discrete_streams_override is not None and int(num_discrete_streams_override) < 1:
        raise ValueError("num_discrete_streams_override must be >= 1 when provided.")

    def _resolved_stream_count(default_stream_count: int) -> int:
        if num_discrete_streams_override is not None:
            return max(1, int(num_discrete_streams_override))
        return max(1, int(default_stream_count))

    tokenizer = getattr(dataset, "tokenizer", None)
    manifest_path = getattr(dataset, "manifest_path", None)
    if manifest_path is not None and tokenizer is not None and hasattr(tokenizer, "char_to_id"):
        char_to_id = getattr(tokenizer, "char_to_id")
        append_eos = bool(getattr(tokenizer, "append_eos", False))
        if not isinstance(char_to_id, Mapping):
            raise ValueError("dataset.tokenizer.char_to_id must be a mapping.")
        num_discrete_streams = _resolved_stream_count(
            int(getattr(dataset, "discrete_stream_count", 1) or 1)
        )
        total_entries = len(dataset) if hasattr(dataset, "__len__") else None
        resolved_manifest_path = Path(manifest_path).expanduser().resolve()
        lengths = _estimate_manifest_lengths_from_file(
            manifest_path=resolved_manifest_path,
            char_to_id=char_to_id,
            append_eos=append_eos,
            codec_frame_rate_hz=codec_frame_rate_hz,
            num_discrete_streams=num_discrete_streams,
            total_entries=total_entries,
        )
        if lengths:
            return lengths

    manifest_entries = _iter_manifest_entries(dataset)
    if manifest_entries is not None and tokenizer is not None and hasattr(tokenizer, "char_to_id"):
        char_to_id = getattr(tokenizer, "char_to_id")
        append_eos = bool(getattr(tokenizer, "append_eos", False))
        if not isinstance(char_to_id, Mapping):
            raise ValueError("dataset.tokenizer.char_to_id must be a mapping.")
        num_discrete_streams = _resolved_stream_count(
            int(getattr(dataset, "discrete_stream_count", 1) or 1)
        )
        total_entries = len(dataset) if hasattr(dataset, "__len__") else None

        lengths: list[int] = []
        for entry in _progress(
            manifest_entries,
            total=total_entries,
            desc="adaptive_batching: estimating lengths",
            unit="sample",
        ):
            text_prompt_len = _estimate_text_token_count(
                str(getattr(entry, "prompt_transcript")),
                char_to_id=char_to_id,
                append_eos=append_eos,
            )
            text_target_len = _estimate_text_token_count(
                str(getattr(entry, "transcript")),
                char_to_id=char_to_id,
                append_eos=append_eos,
            )
            prompt_discrete_len = _estimate_discrete_length(
                float(getattr(entry, "prompt_duration")),
                codec_frame_rate_hz,
            )
            target_discrete_len = _estimate_discrete_length(
                float(getattr(entry, "duration")),
                codec_frame_rate_hz,
            )
            lengths.append(
                _estimate_concat_sequence_length(
                    text_prompt_length=text_prompt_len,
                    speech_prompt_length=prompt_discrete_len,
                    text_target_length=text_target_len,
                    speech_target_length=target_discrete_len,
                    num_discrete_streams=num_discrete_streams,
                )
            )
        if lengths:
            return lengths

    samples = getattr(dataset, "_samples", None)
    if isinstance(samples, Sequence) and len(samples) > 0:
        lengths = []
        for sample in _progress(
            samples,
            total=len(samples),
            desc="adaptive_batching: estimating lengths",
            unit="sample",
        ):
            if not isinstance(sample, Mapping):
                raise ValueError(
                    "Expected each in-memory dataset sample to be a mapping for adaptive batching."
                )

            text_prompt_len = _safe_1d_length(sample.get("text_prompt"), field_name="text_prompt")
            text_target_len = _safe_1d_length(sample.get("text_target"), field_name="text_target")
            prompt_discrete_len = _safe_2d_length(
                sample.get("discrete_prompt"),
                field_name="discrete_prompt",
            )
            target_discrete_len = _safe_2d_length(
                sample.get("discrete_target"),
                field_name="discrete_target",
            )
            discrete_prompt = sample.get("discrete_prompt")
            num_discrete_streams = _resolved_stream_count(
                int(getattr(discrete_prompt, "shape", [1, 1])[1])
            )
            lengths.append(
                _estimate_concat_sequence_length(
                    text_prompt_length=text_prompt_len,
                    speech_prompt_length=prompt_discrete_len,
                    text_target_length=text_target_len,
                    speech_target_length=target_discrete_len,
                    num_discrete_streams=num_discrete_streams,
                )
            )
        return lengths

    raise ValueError(
        "Adaptive batching requires PrismDataset-backed samples "
        "(manifest entries or in-memory mapped samples)."
    )


class AdaptiveMemoryBatchSampler(Sampler[Union[list[int], list[tuple[int, int]]]]):
    """
    Variable-size batch sampler targeting a memory budget proxy.

    Memory estimate per candidate batch:
        estimated_cost = batch_size * (max_sequence_length_in_batch ** 2)

    Optional length bucketing can reduce pad waste by keeping similarly sized
    samples near each other before greedy packing.
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
        bucket_by_length: bool = True,
        length_bucket_size: int | None = None,
        sample_lengths_by_stream_count: Mapping[int, Sequence[int]] | None = None,
        random_active_discrete_stream_count: bool = False,
        min_active_discrete_stream_count: int = 1,
        max_active_discrete_stream_count: int | None = None,
        yield_active_discrete_stream_count_with_indices: bool = False,
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

        normalized_lengths = [max(1, int(length)) for length in sample_lengths]
        self.sample_lengths = normalized_lengths
        self.target_batch_cost = int(target_batch_cost)
        self.max_batch_size = int(max_batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = False
        self.seed = int(seed)
        self.bucket_by_length = bool(bucket_by_length)
        self.length_bucket_size = (
            None if length_bucket_size is None else int(length_bucket_size)
        )
        self.random_active_discrete_stream_count = bool(random_active_discrete_stream_count)
        self.min_active_discrete_stream_count = int(min_active_discrete_stream_count)
        self.max_active_discrete_stream_count = (
            None
            if max_active_discrete_stream_count is None
            else int(max_active_discrete_stream_count)
        )
        self.yield_active_discrete_stream_count_with_indices = bool(
            yield_active_discrete_stream_count_with_indices
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
        if self.length_bucket_size is not None and self.length_bucket_size < 1:
            raise ValueError("length_bucket_size must be >= 1 when provided.")

        self._sample_lengths_by_stream_count: dict[int, list[int]] = {}
        if sample_lengths_by_stream_count is not None:
            if not isinstance(sample_lengths_by_stream_count, Mapping):
                raise ValueError("sample_lengths_by_stream_count must be a mapping when provided.")
            for stream_count_raw, lengths_raw in sample_lengths_by_stream_count.items():
                stream_count = int(stream_count_raw)
                if stream_count < 1:
                    raise ValueError("sample_lengths_by_stream_count keys must be >= 1.")
                normalized_stream_lengths = [max(1, int(length)) for length in lengths_raw]
                if len(normalized_stream_lengths) != len(self.sample_lengths):
                    raise ValueError(
                        "sample_lengths_by_stream_count entries must match sample_lengths size."
                    )
                self._sample_lengths_by_stream_count[stream_count] = normalized_stream_lengths

        if self.random_active_discrete_stream_count and not self._sample_lengths_by_stream_count:
            raise ValueError(
                "random_active_discrete_stream_count=True requires sample_lengths_by_stream_count."
            )
        if (
            self.yield_active_discrete_stream_count_with_indices
            and not self._sample_lengths_by_stream_count
        ):
            raise ValueError(
                "yield_active_discrete_stream_count_with_indices=True requires "
                "sample_lengths_by_stream_count."
            )

        self._active_stream_choices = self._resolve_active_stream_choices()
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _ordered_indices_and_rng(self) -> tuple[list[int], random.Random]:
        indices = list(range(len(self.sample_lengths)))
        rng = random.Random(self.seed + self._epoch)
        if self.shuffle:
            if self.bucket_by_length:
                indices = self._length_bucketed_shuffle(indices, rng=rng)
            else:
                rng.shuffle(indices)
        self._epoch += 1
        return indices, rng

    def _length_bucketed_shuffle(
        self,
        indices: Sequence[int],
        *,
        rng: random.Random,
    ) -> list[int]:
        if not indices:
            return []

        sorted_indices = sorted(indices, key=lambda idx: self.sample_lengths[idx])
        auto_bucket_size = max(512, self.max_batch_size * 8)
        bucket_size = (
            auto_bucket_size
            if self.length_bucket_size is None
            else self.length_bucket_size
        )

        buckets = [
            sorted_indices[start : start + bucket_size]
            for start in range(0, len(sorted_indices), bucket_size)
        ]
        rng.shuffle(buckets)

        ordered: list[int] = []
        for bucket in buckets:
            # Randomly reverse per bucket so we do not always walk short->long.
            if rng.random() < 0.5:
                bucket = list(reversed(bucket))
            ordered.extend(bucket)
        return ordered

    def _resolve_active_stream_choices(self) -> list[int]:
        if not self._sample_lengths_by_stream_count:
            return []

        sorted_stream_counts = sorted(self._sample_lengths_by_stream_count.keys())
        upper_bound = (
            sorted_stream_counts[-1]
            if self.max_active_discrete_stream_count is None
            else self.max_active_discrete_stream_count
        )
        return [
            stream_count
            for stream_count in sorted_stream_counts
            if self.min_active_discrete_stream_count <= stream_count <= upper_bound
        ]

    def _sample_batch_stream_count(self, rng: random.Random) -> int | None:
        if not self._sample_lengths_by_stream_count:
            return None
        if not self._active_stream_choices:
            raise ValueError(
                "No stream-count choices are available for adaptive batching. "
                "Check min/max_active_discrete_stream_count and sample_lengths_by_stream_count."
            )

        if self.random_active_discrete_stream_count:
            return int(rng.choice(self._active_stream_choices))
        return int(self._active_stream_choices[-1])

    def _sample_length_for(
        self,
        sample_idx: int,
        *,
        active_stream_count: int | None,
    ) -> int:
        if active_stream_count is None:
            return self.sample_lengths[sample_idx]
        stream_lengths = self._sample_lengths_by_stream_count.get(active_stream_count)
        if stream_lengths is None:
            raise ValueError(
                "Missing sample lengths for stream count "
                f"{active_stream_count} in sample_lengths_by_stream_count."
            )
        return stream_lengths[sample_idx]

    def _build_batches(
        self,
        ordered_indices: Sequence[int],
        *,
        rng: random.Random,
    ) -> list[list[int] | list[tuple[int, int]]]:
        batches: list[list[int] | list[tuple[int, int]]] = []
        cursor = 0
        total = len(ordered_indices)
        while cursor < total:
            active_stream_count = self._sample_batch_stream_count(rng)
            batch: list[int] = []
            batch_max_len = 0

            while cursor < total and len(batch) < self.max_batch_size:
                sample_idx = int(ordered_indices[cursor])
                sample_len = self._sample_length_for(
                    sample_idx,
                    active_stream_count=active_stream_count,
                )

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

            if (
                self.yield_active_discrete_stream_count_with_indices
                and active_stream_count is not None
            ):
                batches.append(
                    [(sample_idx, int(active_stream_count)) for sample_idx in batch]
                )
            else:
                batches.append(batch)

        return batches

    def __iter__(self):
        ordered, rng = self._ordered_indices_and_rng()
        for batch in self._build_batches(ordered, rng=rng):
            yield batch

    def __len__(self) -> int:
        ordered = list(range(len(self.sample_lengths)))
        rng = random.Random(self.seed)
        return len(self._build_batches(ordered, rng=rng))
