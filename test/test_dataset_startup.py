from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import PrismDataset


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
