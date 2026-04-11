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
    assert lengths == [19, 20]
