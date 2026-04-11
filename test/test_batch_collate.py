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


def _expected_flat_len(text_len: int, speech_len: int, num_discrete_streams: int) -> int:
    num_streams = num_discrete_streams + 1
    return text_len + 1 + speech_len * num_streams + 1


def test_batch_collate_builds_split_parts_and_lengths_without_delay():
    collate = BatchCollate(discrete_token_count=100)
    _, eos_token_id, pad_token_id, _ = build_shared_token_layout(100)

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

    assert tuple(out["text_prompt"].shape) == (2, 3)
    assert tuple(out["discrete_prompt"].shape) == (2, 2, 2)
    assert tuple(out["continuous_prompt"].shape) == (2, 2, 1)
    assert tuple(out["text_target"].shape) == (2, 2)
    assert tuple(out["discrete_target"].shape) == (2, 3, 2)
    assert tuple(out["continuous_target"].shape) == (2, 3, 1)

    assert out["text_prompt_lengths"].tolist() == [3, 1]
    assert out["speech_prompt_lengths"].tolist() == [2, 1]
    assert out["text_target_lengths"].tolist() == [2, 1]
    assert out["speech_target_lengths"].tolist() == [3, 2]
    # Terminal EOS speech block was appended to each sample.
    assert out["discrete_target"][0, 2].tolist() == [eos_token_id, eos_token_id]
    assert out["discrete_target"][1, 1].tolist() == [eos_token_id, eos_token_id]
    assert out["continuous_target"][0, 2].tolist() == [0.0]
    assert out["continuous_target"][1, 1].tolist() == [0.0]

    # Padding check for split parts.
    assert out["text_prompt"][1].tolist() == [300, pad_token_id, pad_token_id]

    # Flat attention mask follows: prompt_text+EOT+prompt_speech+EOS+target_text+EOT+target_speech+EOS.
    len_a = _expected_flat_len(3, 2, 2) + _expected_flat_len(2, 3, 2)
    len_b = _expected_flat_len(1, 1, 2) + _expected_flat_len(1, 2, 2)
    assert tuple(out["attention_mask"].shape) == (2, len_a)
    assert out["attention_mask"][0].tolist() == [True] * len_a
    assert out["attention_mask"][1].tolist() == [True] * len_b + [False] * (len_a - len_b)

    # Pre-flattened tensors are produced in collate and padded to the same sequence length.
    assert tuple(out["flat_token_ids"].shape) == (2, len_a)
    assert tuple(out["flat_token_type_ids"].shape) == (2, len_a)
    assert tuple(out["flat_speech_stream_ids"].shape) == (2, len_a)
    assert tuple(out["flat_target_block_ids"].shape) == (2, len_a)
    assert tuple(out["flat_continuous_values"].shape) == (2, len_a, 1)
    assert out["flat_target_block_counts"].tolist() == [3, 2]
    assert tuple(out["flat_summary"].shape) == (2, 14)

    # Sample A starts with prompt text then EOT.
    assert out["flat_token_ids"][0, :4].tolist() == [200, 201, 202, 100]
    # [text_prompt_start, text_prompt_end, speech_prompt_start, speech_prompt_end, ...]
    assert out["flat_summary"][0, :4].tolist() == [0, 3, 4, 10]
    assert int(out["flat_summary"][0, 8].item()) == len_a
    # stream summary: [text_stream_idx, speech_discrete_start, speech_discrete_end, speech_continuous_idx]
    assert out["flat_summary"][0, 10:14].tolist() == [0, 0, 1, 2]
