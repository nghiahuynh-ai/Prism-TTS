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
    # Single-utterance continuous-only layout: text_target + EOT + target_frames.
    # sample 0: transcript "abc" (3) + 1 + round(0.16*12.5)=2 -> 6
    # sample 1: transcript "bb" (2) + 1 + round(0.24*12.5)=3 -> 6
    assert lengths == [6, 6]


def test_estimate_prism_sample_lengths_streams_indexed_manifest(tmp_path):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "# comment line\n"
        "a.wav|0.16|abc|a.npy|p.wav|0.1|ab|p.npy\n"
        "\n"
        "b.wav|0.24|bbz|b.npy|p.wav|0.1|ab|p.npy\n",
        encoding="utf-8",
    )
    tokenizer = SimpleNamespace(char_to_id={"a": 0, "b": 1, "c": 2}, append_eos=False)
    dataset = SimpleNamespace(
        _entries=[],
        tokenizer=tokenizer,
        manifest_path=manifest,
        manifest_read_buffer_bytes=8192,
        manifest_progress_every=0,
    )

    # "z" is outside the vocab and must not count toward the text length.
    # sample 0: 3 chars + 1 + round(0.16*12.5)=2 -> 6
    # sample 1: 2 chars + 1 + round(0.24*12.5)=3 -> 6
    lengths = estimate_prism_sample_lengths(dataset, codec_frame_rate_hz=12.5)
    assert lengths == [6, 6]

    # Second call must round-trip through the sidecar cache with the same result.
    cache_path = manifest.with_name(manifest.name + ".prism-lengths.npz")
    assert cache_path.exists()
    cached_lengths = estimate_prism_sample_lengths(dataset, codec_frame_rate_hz=12.5)
    assert cached_lengths == [6, 6]

    # A different codec rate must invalidate the cache, not reuse it.
    # sample 0: 3 + 1 + round(0.16*25)=4 -> 8; sample 1: 2 + 1 + round(0.24*25)=6 -> 9
    assert estimate_prism_sample_lengths(dataset, codec_frame_rate_hz=25.0) == [8, 9]
