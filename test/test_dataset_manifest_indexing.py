from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.adaptive_batching import estimate_prism_sample_lengths
from dataset.dataset import PrismDataset


def _write_payload(path: Path, *, discrete: np.ndarray, continuous: np.ndarray) -> None:
    payload = {"discrete": discrete, "continuous": continuous}
    np.save(path, payload, allow_pickle=True)


def test_prism_dataset_builds_manifest_index_without_materializing_entries(tmp_path: Path):
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("a\nb\nc\n", encoding="utf-8")

    discrete = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    continuous = np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32)

    target_a = tmp_path / "target_a.npy"
    prompt_a = tmp_path / "prompt_a.npy"
    target_b = tmp_path / "target_b.npy"
    prompt_b = tmp_path / "prompt_b.npy"
    _write_payload(target_a, discrete=discrete, continuous=continuous)
    _write_payload(prompt_a, discrete=discrete, continuous=continuous)
    _write_payload(target_b, discrete=discrete, continuous=continuous)
    _write_payload(prompt_b, discrete=discrete, continuous=continuous)

    manifest_path = tmp_path / "manifest.txt"
    manifest_path.write_text(
        (
            "# comment line\n"
            "utt_a|0.16|abc|target_a.npy|pr_a|0.08|ab|prompt_a.npy\n"
            "utt_b|0.24|bc|target_b.npy|pr_b|0.16|a|prompt_b.npy\n"
        ),
        encoding="utf-8",
    )

    dataset = PrismDataset(
        source=manifest_path,
        vocab_path=vocab_path,
        discrete_stream_count=2,
        continuous_feature_dim=1,
    )

    assert len(dataset) == 2
    assert dataset._entries == []
    assert len(dataset._manifest_line_offsets) == 2

    sample = dataset[0]
    assert sample["text_target"].shape[0] == 3
    assert sample["text_prompt"].shape[0] == 2
    assert tuple(sample["discrete_target"].shape) == (2, 2)
    assert tuple(sample["continuous_target"].shape) == (2, 1)


def test_estimate_lengths_with_streamed_manifest_entries(tmp_path: Path):
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("a\nb\nc\n", encoding="utf-8")

    discrete = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    continuous = np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32)
    for name in ("target_a", "prompt_a", "target_b", "prompt_b"):
        _write_payload(tmp_path / f"{name}.npy", discrete=discrete, continuous=continuous)

    manifest_path = tmp_path / "manifest.txt"
    manifest_path.write_text(
        (
            "utt_a|0.16|abc|target_a.npy|pr_a|0.08|ab|prompt_a.npy\n"
            "utt_b|0.24|bc|target_b.npy|pr_b|0.16|a|prompt_b.npy\n"
        ),
        encoding="utf-8",
    )

    dataset = PrismDataset(
        source=manifest_path,
        vocab_path=vocab_path,
        discrete_stream_count=2,
        continuous_feature_dim=1,
    )
    lengths = estimate_prism_sample_lengths(dataset, codec_frame_rate_hz=12.5)
    assert lengths == [21, 25]
