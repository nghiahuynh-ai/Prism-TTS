from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import BatchCollate, build_shared_token_layout


def _make_sample(
    text_prompt: list[int],
    discrete_prompt: list[list[int]],
    continuous_prompt: list[list[float]],
    text_target: list[int],
    discrete_target: list[list[int]],
    continuous_target: list[list[float]],
) -> dict[str, torch.Tensor]:
    return {
        "text_prompt": torch.tensor(text_prompt, dtype=torch.long),
        "discrete_prompt": torch.tensor(discrete_prompt, dtype=torch.long),
        "continuous_prompt": torch.tensor(continuous_prompt, dtype=torch.float32),
        "text_target": torch.tensor(text_target, dtype=torch.long),
        "discrete_target": torch.tensor(discrete_target, dtype=torch.long),
        "continuous_target": torch.tensor(continuous_target, dtype=torch.float32),
    }


def test_batch_collate_builds_delayed_concat_streams_and_lengths():
    collate = BatchCollate(
        discrete_token_count=100,
        discrete_stream_delay_ms=80.0,
        codec_frame_rate_hz=12.5,
    )
    delay_token_id, eos_token_id, pad_token_id, _ = build_shared_token_layout(100)

    sample_a = _make_sample(
        text_prompt=[200, 201, 202],
        discrete_prompt=[[1, 10], [2, 20]],
        continuous_prompt=[[0.1], [0.2]],
        text_target=[210, 211],
        discrete_target=[[3, 30], [4, 40]],
        continuous_target=[[0.3], [0.4]],
    )
    sample_b = _make_sample(
        text_prompt=[300],
        discrete_prompt=[[5, 50]],
        continuous_prompt=[[0.5]],
        text_target=[310],
        discrete_target=[[6, 60]],
        continuous_target=[[0.6]],
    )

    out = collate([sample_a, sample_b])

    assert tuple(out["text"].shape) == (2, 9)
    assert tuple(out["discrete"].shape) == (2, 9, 2)
    assert tuple(out["continuous"].shape) == (2, 9, 1)
    assert out["prompt_lengths"].tolist() == [5, 3]
    assert out["target_lengths"].tolist() == [4, 3]

    expected_text_a = [
        200,
        201,
        202,
        eos_token_id,
        pad_token_id,
        210,
        211,
        eos_token_id,
        pad_token_id,
    ]
    assert out["text"][0].tolist() == expected_text_a

    expected_d1_a = [
        delay_token_id,
        delay_token_id,
        1,
        2,
        eos_token_id,
        delay_token_id,
        3,
        4,
        eos_token_id,
    ]
    expected_d2_a = [
        delay_token_id,
        delay_token_id,
        10,
        20,
        eos_token_id,
        delay_token_id,
        30,
        40,
        eos_token_id,
    ]
    assert out["discrete"][0, :, 0].tolist() == expected_d1_a
    assert out["discrete"][0, :, 1].tolist() == expected_d2_a

    expected_cont_a = [0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.3, 0.4, 0.0]
    torch.testing.assert_close(
        out["continuous"][0, :, 0],
        torch.tensor(expected_cont_a, dtype=torch.float32),
    )

    assert out["attention_mask"][0].tolist() == [True] * 9
    assert out["attention_mask"][1].tolist() == [True] * 6 + [False] * 3
