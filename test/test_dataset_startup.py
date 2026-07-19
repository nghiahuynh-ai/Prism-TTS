from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import LazyPrismDataset, PrismDataset


def test_manifest_loader_accepts_large_configured_read_buffer(tmp_path):
    manifest_path = tmp_path / "manifest.txt"
    manifest_path.write_text(
        "target.wav|1.0|hello|target.npy|prompt.wav|1.0|prompt|prompt.npy\n",
        encoding="utf-8",
    )

    dataset = PrismDataset(
        manifest_path,
        vocab_path=PROJECT_ROOT / "dataset" / "vocab.txt",
        manifest_read_buffer_bytes=1024 * 1024,
    )

    assert len(dataset) == 1
    assert dataset._entries[0].target_npy_path == tmp_path / "target.npy"


def test_lazy_manifest_defers_parsing_until_iteration(tmp_path):
    manifest_path = tmp_path / "manifest.txt"
    manifest_path.write_text(
        "one.wav|1.0|one|one.npy|prompt.wav|1.0|prompt|prompt.npy\n"
        "two.wav|1.0|two|two.npy|prompt.wav|1.0|prompt|prompt.npy\n",
        encoding="utf-8",
    )

    dataset = LazyPrismDataset(
        manifest_path,
        vocab_path=PROJECT_ROOT / "dataset" / "vocab.txt",
        shuffle_manifest=False,
    )

    # Dataset construction only records the manifest path; it does not parse
    # every row or invent a length for Lightning to consume eagerly.
    with pytest.raises(TypeError):
        len(dataset)

    entries = list(dataset._iter_manifest_entries(shard_index=0, shard_count=1))
    assert [entry.file_name for entry in entries] == ["one.wav", "two.wav"]
    assert entries[0].target_npy_path == tmp_path / "one.npy"
