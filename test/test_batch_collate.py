from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import BatchCollate, SPEECH_FRAME_TOKEN_TYPE, build_shared_token_layout  # noqa: E402


def _sample(
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


def test_batch_collate_builds_fused_shifted_ar_frames() -> None:
    collate = BatchCollate(discrete_token_count=100)
    eot_id, eos_id, pad_id, _ = build_shared_token_layout(100)
    first = _sample(
        text_prompt=[200, 201, 202],
        discrete_prompt=[[1, 10], [2, 20]],
        continuous_prompt=[[0.1], [0.2]],
        text_target=[210, 211],
        discrete_target=[[3, 30], [4, 40]],
        continuous_target=[[0.3], [0.4]],
    )
    second = _sample(
        text_prompt=[300],
        discrete_prompt=[[5, 50]],
        continuous_prompt=[[0.5]],
        text_target=[310],
        discrete_target=[[6, 60]],
        continuous_target=[[0.6]],
    )

    out = collate([first, second])

    # Target EOS frames are appended before building the shifted sequence.
    assert out["speech_target_lengths"].tolist() == [3, 2]
    assert out["discrete_target"][0, 2].tolist() == [eos_id, eos_id]
    assert out["continuous_target"][0, 2].tolist() == [0.0]

    # One model token per speech frame: text+EOT+prompt+EOS+text+EOT+(T-1) inputs.
    first_len = 3 + 1 + 2 + 1 + 2 + 1 + (3 - 1)
    second_len = 1 + 1 + 1 + 1 + 1 + 1 + (2 - 1)
    assert tuple(out["flat_token_ids"].shape) == (2, first_len)
    assert out["attention_mask"][0].tolist() == [True] * first_len
    assert out["attention_mask"][1].tolist() == [True] * second_len + [False] * (first_len - second_len)
    assert tuple(out["flat_discrete_values"].shape) == (2, first_len, 2)
    assert tuple(out["flat_continuous_values"].shape) == (2, first_len, 1)
    assert tuple(out["flat_target_discrete_values"].shape) == (2, first_len, 2)
    assert tuple(out["flat_target_continuous_values"].shape) == (2, first_len, 1)

    # Target EOT is the anchor for frame 0. Each following input frame predicts its successor.
    target_ids = out["flat_target_block_ids"][0]
    anchors = torch.nonzero(target_ids >= 0, as_tuple=False).squeeze(1)
    assert anchors.tolist() == [9, 10, 11]
    assert target_ids[anchors].tolist() == [0, 1, 2]
    assert out["flat_token_ids"][0, 9].item() == eot_id
    assert out["flat_token_type_ids"][0, 10:12].tolist() == [SPEECH_FRAME_TOKEN_TYPE] * 2
    assert out["flat_discrete_values"][0, 10].tolist() == [3, 30]
    assert out["flat_discrete_values"][0, 11].tolist() == [4, 40]
    assert out["flat_target_discrete_values"][0, anchors].tolist() == [
        [3, 30],
        [4, 40],
        [eos_id, eos_id],
    ]
    assert torch.allclose(out["flat_target_continuous_values"][0, anchors[-1]], torch.zeros(1))

    # Text positions have no speech-frame input and use the discrete pad value only as storage.
    assert out["flat_token_ids"][0, :4].tolist() == [200, 201, 202, eot_id]
    assert out["flat_discrete_values"][0, 0].tolist() == [pad_id, pad_id]
    assert tuple(out["flat_summary"].shape) == (2, 10)
