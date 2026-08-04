from __future__ import annotations

import pytest
import torch
from torch import nn

from utils.pretrained_weights import load_pretrained_weights


class _ToyModel(nn.Module):
    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        self.output = nn.Linear(2, output_size)


def test_load_pretrained_weights_loads_lightning_model_state_only(tmp_path) -> None:
    source = _ToyModel(output_size=2)
    with torch.no_grad():
        source.encoder.weight.fill_(3.0)
        source.output.weight.fill_(5.0)
        source.output.bias.fill_(7.0)
    checkpoint_path = tmp_path / "source.ckpt"
    torch.save(
        {
            "state_dict": {
                f"model.{key}": value.clone()
                for key, value in source.state_dict().items()
            },
            "optimizer_states": [{"state": "not loaded"}],
            "global_step": 123,
        },
        checkpoint_path,
    )

    target = _ToyModel(output_size=2)
    result = load_pretrained_weights(target, checkpoint_path)

    assert result.loaded_keys == tuple(source.state_dict())
    assert not result.missing_keys
    assert not result.unexpected_keys
    assert not result.mismatched_keys
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)


def test_load_pretrained_weights_skips_shape_mismatches_for_adaptation(
    tmp_path,
) -> None:
    source = _ToyModel(output_size=3)
    with torch.no_grad():
        source.encoder.weight.fill_(11.0)
    checkpoint_path = tmp_path / "source.pt"
    torch.save(source.state_dict(), checkpoint_path)

    target = _ToyModel(output_size=2)
    original_output = target.output.weight.detach().clone()
    result = load_pretrained_weights(target, checkpoint_path)

    assert result.loaded_keys == ("encoder.weight",)
    assert {key for key, _, _ in result.mismatched_keys} == {
        "output.weight",
        "output.bias",
    }
    assert torch.equal(target.encoder.weight, source.encoder.weight)
    assert torch.equal(target.output.weight, original_output)

    strict_target = _ToyModel(output_size=2)
    with pytest.raises(RuntimeError, match="shape mismatched"):
        load_pretrained_weights(strict_target, checkpoint_path, strict=True)
