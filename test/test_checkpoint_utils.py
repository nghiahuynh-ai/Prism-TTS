from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import LlamaConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.prism_tts import PrismTTS
from utils.checkpoint_utils import load_model_weights


def _build_tiny_model() -> PrismTTS:
    return PrismTTS(
        llama_config=LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            pad_token_id=0,
            eos_token_id=2,
        ),
        num_discrete_tokens=2,
        discrete_vocab_size=16,
        continuous_latent_size=8,
        flow_num_res_blocks=1,
        flow_model_channels=32,
        flow_sample_steps=2,
        parallel_sample_steps=2,
    )


def _clone_state_dict(model: PrismTTS) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def test_load_model_weights_strips_lightning_model_prefix(tmp_path: Path) -> None:
    model = _build_tiny_model()
    source_state = _clone_state_dict(model)
    checkpoint_path = tmp_path / "prefixed.ckpt"
    torch.save(
        {"state_dict": {f"model.{key}": value for key, value in source_state.items()}},
        checkpoint_path,
    )

    reinitialized = _build_tiny_model()
    for parameter in reinitialized.parameters():
        parameter.data.zero_()

    _, missing, unexpected = load_model_weights(
        reinitialized,
        checkpoint_path,
        use_ema=False,
        strict=True,
    )

    assert missing == []
    assert unexpected == []
    for key, value in source_state.items():
        assert torch.equal(reinitialized.state_dict()[key], value)


def test_load_model_weights_prefers_ema_when_requested(tmp_path: Path) -> None:
    model = _build_tiny_model()
    regular_state = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
    ema_state = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
    checkpoint_path = tmp_path / "ema.ckpt"
    torch.save(
        {
            "state_dict": {f"model.{key}": value for key, value in regular_state.items()},
            "ema_state": ema_state,
        },
        checkpoint_path,
    )

    _, missing, unexpected = load_model_weights(
        model,
        checkpoint_path,
        use_ema=True,
        strict=True,
    )

    assert missing == []
    assert unexpected == []
    first_key = next(iter(model.state_dict()))
    assert torch.equal(model.state_dict()[first_key], ema_state[first_key])


def test_load_model_weights_non_strict_reports_incompatible_keys(tmp_path: Path) -> None:
    model = _build_tiny_model()
    source_state = _clone_state_dict(model)
    removed_key = next(iter(source_state))
    source_state.pop(removed_key)
    source_state["unexpected.weight"] = torch.zeros(1)
    checkpoint_path = tmp_path / "non_strict.ckpt"
    torch.save(source_state, checkpoint_path)

    _, missing, unexpected = load_model_weights(
        model,
        checkpoint_path,
        use_ema=False,
        strict=False,
    )

    assert removed_key in missing
    assert "unexpected.weight" in unexpected


def test_load_model_weights_non_strict_keeps_new_active_stream_embedding_init(
    tmp_path: Path,
) -> None:
    model = _build_tiny_model()
    source_state = _clone_state_dict(model)
    removed_key = "active_stream_count_embedding.weight"
    source_state.pop(removed_key)
    checkpoint_path = tmp_path / "missing_active_stream_count.ckpt"
    torch.save(source_state, checkpoint_path)

    reinitialized = _build_tiny_model()

    _, missing, unexpected = load_model_weights(
        reinitialized,
        checkpoint_path,
        use_ema=False,
        strict=False,
    )

    assert removed_key in missing
    assert unexpected == []
    assert torch.count_nonzero(reinitialized.active_stream_count_embedding.weight).item() == 0
