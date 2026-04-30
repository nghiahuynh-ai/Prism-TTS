from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import BatchCollate, PrismDataset, build_shared_token_layout


def _build_base_sample() -> dict[str, torch.Tensor]:
    return {
        "text_prompt": torch.tensor([10, 11], dtype=torch.long),
        "discrete_prompt": torch.tensor(
            [[1, 2, 3], [4, 5, 6]],
            dtype=torch.long,
        ),
        "continuous_prompt": torch.tensor(
            [[0.1, 0.2], [0.3, 0.4]],
            dtype=torch.float32,
        ),
        "text_target": torch.tensor([20], dtype=torch.long),
        "discrete_target": torch.tensor(
            [[7, 8, 9], [10, 11, 12]],
            dtype=torch.long,
        ),
        "continuous_target": torch.tensor(
            [[0.5, 0.6], [0.7, 0.8]],
            dtype=torch.float32,
        ),
    }


def test_prism_dataset_samples_first_m_discrete_streams():
    sample = _build_base_sample()
    dataset = PrismDataset(
        source=[sample],
        discrete_stream_count=3,
        sample_discrete_stream_count=True,
    )

    torch.manual_seed(0)
    expected_stream_counts = [3, 1, 3, 1, 2]
    for expected_m in expected_stream_counts:
        out = dataset[0]
        assert int(out["discrete_prompt"].shape[1]) == expected_m
        assert int(out["discrete_target"].shape[1]) == expected_m
        assert torch.equal(out["discrete_prompt"], sample["discrete_prompt"][:, :expected_m])
        assert torch.equal(out["discrete_target"], sample["discrete_target"][:, :expected_m])


def test_batch_collate_pads_variable_discrete_stream_counts_to_configured_max():
    sample_a = _build_base_sample()
    sample_b = {
        "text_prompt": torch.tensor([31], dtype=torch.long),
        "discrete_prompt": torch.tensor([[40]], dtype=torch.long),
        "continuous_prompt": torch.tensor([[0.9, 1.0]], dtype=torch.float32),
        "text_target": torch.tensor([41], dtype=torch.long),
        "discrete_target": torch.tensor([[50]], dtype=torch.long),
        "continuous_target": torch.tensor([[1.1, 1.2]], dtype=torch.float32),
    }

    collate = BatchCollate(discrete_token_count=100, discrete_stream_count=3)
    _, eos_token_id, pad_token_id, _ = build_shared_token_layout(100)
    out = collate([sample_a, sample_b])

    assert tuple(out["discrete_prompt"].shape) == (2, 2, 3)
    assert tuple(out["discrete_target"].shape) == (2, 3, 3)
    assert out["discrete_prompt"][1, 0].tolist() == [40, pad_token_id, pad_token_id]
    assert out["discrete_target"][1, 1].tolist() == [eos_token_id, pad_token_id, pad_token_id]

    # Flat sequence uses the sampled stream count per sample (sample_b has exactly one stream).
    assert int(out["flat_summary"][1, 9].item()) == 1
    valid = out["attention_mask"][1]
    sample_stream_ids = out["flat_speech_stream_ids"][1][valid]
    assert int(sample_stream_ids.max().item()) == 1
