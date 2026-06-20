from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.adaptive_batching import AdaptiveMemoryBatchSampler, estimate_prism_sample_lengths


def test_adaptive_sampler_carries_overflow_and_packs_short_samples():
    sampler = AdaptiveMemoryBatchSampler(
        sample_lengths=[10, 10, 10, 4, 4, 4, 4],
        target_batch_cost=240,
        max_batch_size=8,
        shuffle=False,
        drop_last=False,
        seed=0,
    )
    batches = list(iter(sampler))
    assert batches == [[0, 1], [2, 3], [4, 5, 6]]


def test_estimate_prism_sample_lengths_from_manifest_metadata():
    tokenizer = SimpleNamespace(
        char_to_id={"a": 0, "b": 1, "c": 2},
        append_eos=False,
    )
    entries = [
        SimpleNamespace(
            prompt_transcript="ab",
            transcript="abc",
            prompt_duration=0.16,
            duration=0.16,
        ),
        SimpleNamespace(
            prompt_transcript="aaaa",
            transcript="bb",
            prompt_duration=0.08,
            duration=0.24,
        ),
    ]
    dataset = SimpleNamespace(_entries=entries, tokenizer=tokenizer)
    lengths = estimate_prism_sample_lengths(
        dataset,
        codec_frame_rate_hz=12.5,
    )
    assert lengths == [11, 12]


def test_estimate_prism_sample_lengths_with_stream_override():
    tokenizer = SimpleNamespace(
        char_to_id={"a": 0, "b": 1, "c": 2},
        append_eos=False,
    )
    entries = [
        SimpleNamespace(
            prompt_transcript="ab",
            transcript="abc",
            prompt_duration=0.16,
            duration=0.16,
        ),
        SimpleNamespace(
            prompt_transcript="aaaa",
            transcript="bb",
            prompt_duration=0.08,
            duration=0.24,
        ),
    ]
    dataset = SimpleNamespace(_entries=entries, tokenizer=tokenizer)
    lengths = estimate_prism_sample_lengths(
        dataset,
        codec_frame_rate_hz=12.5,
        num_discrete_streams_override=3,
    )
    assert lengths == [17, 20]


def test_adaptive_sampler_emits_stream_count_indices_when_prebuilt_stream_schedule_enabled():
    lengths_by_stream_count = {
        1: [10, 10, 10, 10, 10],
        4: [25, 25, 25, 25, 25],
    }
    sampler = AdaptiveMemoryBatchSampler(
        sample_lengths=lengths_by_stream_count[4],
        target_batch_cost=1200,
        max_batch_size=4,
        shuffle=False,
        drop_last=False,
        seed=3,
        sample_lengths_by_stream_count=lengths_by_stream_count,
        random_active_discrete_stream_count=True,
        min_active_discrete_stream_count=1,
        max_active_discrete_stream_count=4,
        yield_active_discrete_stream_count_with_indices=True,
    )

    batches = list(iter(sampler))
    assert batches

    used_indices: list[int] = []
    for batch in batches:
        assert batch
        stream_counts = {int(active_count) for _, active_count in batch}
        assert len(stream_counts) == 1
        active_stream_count = next(iter(stream_counts))
        assert active_stream_count in {1, 4}

        indices = [int(sample_idx) for sample_idx, _ in batch]
        used_indices.extend(indices)
        max_batch_length = max(lengths_by_stream_count[active_stream_count][sample_idx] for sample_idx in indices)
        estimated_cost = len(indices) * (max_batch_length**2)
        if len(indices) > 1:
            assert estimated_cost <= 1200

    assert sorted(used_indices) == [0, 1, 2, 3, 4]
