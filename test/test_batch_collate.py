from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import (
    SPEECH_TOKEN_TYPE,
    TEXT_TOKEN_TYPE,
    BatchCollate,
    build_shared_token_layout,
)


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


def _expected_flat_len(text_target, speech_target) -> int:
    # Single-utterance layout: text_target + EOT + target frames.
    return text_target + 1 + speech_target


def test_batch_collate_builds_continuous_only_causal_layout():
    collate = BatchCollate(discrete_token_count=100)
    eot_token_id, eos_token_id, pad_token_id, _ = build_shared_token_layout(100)

    sample_a = _make_sample(
        text_prompt=[200, 201, 202],
        discrete_prompt=[[1], [2]],
        continuous_prompt=[[0.1, 0.1], [0.2, 0.2]],
        text_target=[210, 211],
        discrete_target=[[3], [4], [5]],
        continuous_target=[[0.3, 0.3], [0.4, 0.4], [0.5, 0.5]],
    )
    sample_b = _make_sample(
        text_prompt=[300],
        discrete_prompt=[[5]],
        continuous_prompt=[[0.5, 0.5]],
        text_target=[310],
        discrete_target=[[6], [7]],
        continuous_target=[[0.6, 0.6], [0.7, 0.7]],
    )

    out = collate([sample_a, sample_b])

    # Split lengths are still emitted (prompt fields are ignored by the flat layout).
    assert out["text_prompt_lengths"].tolist() == [3, 1]
    assert out["speech_prompt_lengths"].tolist() == [2, 1]
    assert out["text_target_lengths"].tolist() == [2, 1]
    assert out["speech_target_lengths"].tolist() == [3, 2]

    # Discrete streams are no longer emitted into the flattened sequence.
    assert "flat_speech_stream_ids" not in out
    assert "flat_summary" not in out
    assert "discrete_prompt" not in out
    assert "discrete_target" not in out

    # Single-utterance flat layout: text_target -> EOT -> target frames (no prompt,
    # no mid-sequence EOS, no trailing EOS token).
    len_a = _expected_flat_len(2, 3)  # = 6
    len_b = _expected_flat_len(1, 2)  # = 4
    assert tuple(out["flat_token_ids"].shape) == (2, len_a)
    assert tuple(out["flat_token_type_ids"].shape) == (2, len_a)
    assert tuple(out["flat_target_block_ids"].shape) == (2, len_a)
    assert tuple(out["flat_continuous_values"].shape) == (2, len_a, 2)
    assert out["flat_target_block_counts"].tolist() == [3, 2]

    assert out["attention_mask"][0].tolist() == [True] * len_a
    assert out["attention_mask"][1].tolist() == [True] * len_b + [False] * (len_a - len_b)

    # No literal EOS token appears in the flattened sequence.
    assert int((out["flat_token_ids"] == eos_token_id).sum().item()) == 0

    # Sample A: text target (2) then EOT.
    assert out["flat_token_ids"][0, :3].tolist() == [210, 211, eot_token_id]

    # Target frames carry frame ids 0..L-1; everything else is -1.
    tgt = out["flat_target_block_ids"][0]
    valid = out["attention_mask"][0]
    frame_positions = torch.nonzero(tgt >= 0, as_tuple=False).squeeze(1)
    assert tgt[frame_positions].tolist() == [0, 1, 2]
    # Target frame positions are the final positions of the (unpadded) sequence.
    assert frame_positions.tolist() == [len_a - 3, len_a - 2, len_a - 1]

    # Every speech position is token_type SPEECH with pad token id; text is TEXT.
    speech_mask = out["flat_token_type_ids"][0] == SPEECH_TOKEN_TYPE
    assert int(speech_mask.sum().item()) == 3  # target frames only
    assert torch.all(out["flat_token_ids"][0][speech_mask] == pad_token_id)
    assert int((out["flat_token_type_ids"][0] == TEXT_TOKEN_TYPE)[valid].sum().item()) == 2 + 1

    # The last target frame's continuous latent is preserved (not zeroed).
    assert out["flat_continuous_values"][0, len_a - 1].tolist() == [0.5, 0.5]


def test_batch_collate_rounds_flat_sequence_to_fixed_bucket():
    collate = BatchCollate(
        discrete_token_count=100,
        flat_sequence_length_multiple=8,
    )
    sample = _make_sample(
        text_prompt=[200],
        discrete_prompt=[[1]],
        continuous_prompt=[[0.1, 0.1]],
        text_target=[210, 211],
        discrete_target=[[3], [4], [5]],
        continuous_target=[[0.3, 0.3], [0.4, 0.4], [0.5, 0.5]],
    )

    out = collate([sample])

    # Natural flat length is 2 text + EOT + 3 frames = 6; only model-facing
    # tensors round to the 8-position allocation bucket.
    assert out["flat_token_ids"].shape == (1, 8)
    assert out["flat_continuous_values"].shape == (1, 8, 2)
    assert out["attention_mask"].tolist() == [[True] * 6 + [False] * 2]
    assert out["text_target"].shape == (1, 2)
    assert out["continuous_target"].shape == (1, 3, 2)
