from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

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


def estimate_prism_sample_lengths(
    dataset: Any,
    *,
    codec_frame_rate_hz: float,
) -> list[int]:
    """Estimate concatenated sequence length for each Prism sample."""
    if codec_frame_rate_hz <= 0:
        raise ValueError("codec_frame_rate_hz must be > 0.")

    tokenizer = getattr(dataset, "tokenizer", None)
    manifest_entries = _iter_manifest_entries(dataset)
    if manifest_entries is not None and tokenizer is not None and hasattr(tokenizer, "char_to_id"):
        char_to_id = getattr(tokenizer, "char_to_id")
        append_eos = bool(getattr(tokenizer, "append_eos", False))
        if not isinstance(char_to_id, Mapping):
            raise ValueError("dataset.tokenizer.char_to_id must be a mapping.")
        num_discrete_streams = int(getattr(dataset, "discrete_stream_count", 1) or 1)

        lengths: list[int] = []
        for entry in manifest_entries:
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
        for sample in samples:
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
            num_discrete_streams = int(getattr(discrete_prompt, "shape", [1, 1])[1])
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

        normalized_lengths = [max(1, int(length)) for length in sample_lengths]
        self.sample_lengths = normalized_lengths
        self.target_batch_cost = int(target_batch_cost)
        self.max_batch_size = int(max_batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = False
        self.seed = int(seed)
        self._epoch = 0

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
        ordered = list(range(len(self.sample_lengths)))
        return len(self._build_batches(ordered))
