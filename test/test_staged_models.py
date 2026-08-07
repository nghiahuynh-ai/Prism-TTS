from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import BatchCollate  # noqa: E402
from models.prism_continuous_meanflow import PrismContinuousMeanFlowTTS  # noqa: E402
from models.prism_discrete_tts import PrismDiscreteTTS  # noqa: E402
from utils import model_utils as MU  # noqa: E402
from utils import generate_utils  # noqa: E402


DEFAULT_DISCRETE_MODEL_CONFIG = PROJECT_ROOT / "config" / "model_discrete.yaml"
DEFAULT_CONTINUOUS_MODEL_CONFIG = PROJECT_ROOT / "config" / "model_continuous_meanflow.yaml"


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def print_configured_stage_model_sizes() -> None:
    """Print configured Stage-1/Stage-2 parameter counts without allocating weights."""
    reports: list[tuple[str, Path, int]] = []
    for stage_name, config_path in (
        ("discrete", DEFAULT_DISCRETE_MODEL_CONFIG),
        ("continuous_meanflow", DEFAULT_CONTINUOUS_MODEL_CONFIG),
    ):
        config = generate_utils.read_yaml(config_path)
        with torch.device("meta"):
            model = generate_utils.build_model(config)
        reports.append((stage_name, config_path, _parameter_count(model)))

    print("Configured staged model sizes:")
    for stage_name, config_path, parameter_count in reports:
        fp32_mib = parameter_count * 4 / (1024**2)
        bf16_mib = parameter_count * 2 / (1024**2)
        print(
            f"  {stage_name:<20} params={parameter_count:>15,} "
            f"fp32={fp32_mib:>9.1f} MiB bf16/fp16={bf16_mib:>9.1f} MiB "
            f"config={config_path}"
        )


def _config() -> LlamaConfig:
    config = LlamaConfig(
        vocab_size=96,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        pad_token_id=34,
        eos_token_id=33,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    return config


def _sample() -> dict[str, torch.Tensor]:
    return {
        "text_prompt": torch.tensor([40, 41], dtype=torch.long),
        "discrete_prompt": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        "continuous_prompt": torch.tensor([[0.1] * 8, [0.2] * 8], dtype=torch.float32),
        "text_target": torch.tensor([42], dtype=torch.long),
        "discrete_target": torch.tensor([[5, 6], [7, 8], [9, 10]], dtype=torch.long),
        "continuous_target": torch.tensor([[0.3] * 8, [0.4] * 8, [0.5] * 8], dtype=torch.float32),
    }


def _discrete() -> PrismDiscreteTTS:
    return PrismDiscreteTTS(
        llama_config=_config(), num_discrete_tokens=2, discrete_vocab_size=35
    )


def _continuous(**kwargs: object) -> PrismContinuousMeanFlowTTS:
    return PrismContinuousMeanFlowTTS(
        llama_config=_config(),
        num_discrete_tokens=2,
        discrete_vocab_size=35,
        continuous_latent_size=8,
        **kwargs,
    )


def test_discrete_stage_is_invariant_to_continuous_payloads() -> None:
    torch.manual_seed(0)
    model = _discrete().eval()
    batch = BatchCollate(discrete_token_count=32)([_sample()])
    changed = {key: value.clone() for key, value in batch.items()}
    changed["flat_continuous_values"].normal_(mean=100.0, std=20.0)
    changed["flat_target_continuous_values"].normal_(mean=-100.0, std=20.0)

    def run(values: dict[str, torch.Tensor]) -> torch.Tensor:
        return model(
            flat_token_ids=values["flat_token_ids"],
            flat_discrete_values=values["flat_discrete_values"],
            flat_token_type_ids=values["flat_token_type_ids"],
            flat_target_discrete_values=values["flat_target_discrete_values"],
            flat_target_block_ids=values["flat_target_block_ids"],
            attention_mask=values["attention_mask"],
        ).loss

    assert torch.allclose(run(batch), run(changed), atol=1e-7, rtol=0.0)


def test_continuous_velocity_cannot_see_future_discrete_noise_or_time() -> None:
    torch.manual_seed(1)
    model = _continuous().eval()
    length = 5
    token_ids = torch.full((1, length), model.pad_token_id, dtype=torch.long)
    discrete = torch.tensor([[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]], dtype=torch.long)
    continuous = torch.randn(1, length, 8)
    timesteps = torch.tensor([[0.1, 0.25, 0.5, 0.75, 0.9]])
    token_types = torch.full((1, length), MU.SPEECH_FRAME_TOKEN_TYPE, dtype=torch.long)
    attention = torch.ones((1, length), dtype=torch.bool)

    changed_discrete = discrete.clone()
    changed_continuous = continuous.clone()
    changed_timesteps = timesteps.clone()
    changed_discrete[:, 3:] = 20
    changed_continuous[:, 3:] = -12.0
    changed_timesteps[:, 3:] = 0.02
    with torch.no_grad():
        first = model._velocity_for_sequence(
            token_ids=token_ids,
            discrete_values=discrete,
            continuous_values=continuous,
            timesteps=timesteps,
            token_type_ids=token_types,
            attention_mask=attention,
        )
        second = model._velocity_for_sequence(
            token_ids=token_ids,
            discrete_values=changed_discrete,
            continuous_values=changed_continuous,
            timesteps=changed_timesteps,
            token_type_ids=token_types,
            attention_mask=attention,
        )
    assert torch.allclose(first[:, :3], second[:, :3], atol=1e-6, rtol=1e-6)


def test_continuous_training_uses_distinct_timesteps_per_active_frame() -> None:
    torch.manual_seed(2)
    model = _continuous(min_train_window=2, max_train_window=2).train()
    sample = _sample()
    sequence = model._build_training_sequence(
        text_prompt=sample["text_prompt"],
        discrete_prompt=sample["discrete_prompt"],
        continuous_prompt=sample["continuous_prompt"],
        text_target=sample["text_target"],
        discrete_target=sample["discrete_target"],
        continuous_target=sample["continuous_target"],
    )
    timesteps, active = sequence[3], sequence[7]
    active_times = timesteps[active]
    assert active_times.shape == (2,)
    assert torch.all(active_times > model.time_epsilon)
    assert torch.all(active_times < 1.0 - model.time_epsilon)
    assert not torch.equal(active_times[0], active_times[1])


def test_continuous_generation_uses_requested_patch_windows_and_solver_steps() -> None:
    torch.manual_seed(3)
    model = _continuous().eval()
    sample = _sample()
    target_discrete = torch.tensor(
        [[[5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]], dtype=torch.long
    )
    calls = 0
    original = model._velocity_for_sequence

    def tracked(**kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(**kwargs)  # type: ignore[arg-type]

    model._velocity_for_sequence = tracked  # type: ignore[method-assign]
    try:
        output = model.generate(
            text_prompt=sample["text_prompt"].unsqueeze(0),
            discrete_prompt=sample["discrete_prompt"].unsqueeze(0),
            continuous_prompt=sample["continuous_prompt"].unsqueeze(0),
            text_target=sample["text_target"].unsqueeze(0),
            discrete_target=target_discrete,
            window_size=[2, 3],
            flow_num_steps=3,
        )
    finally:
        model._velocity_for_sequence = original  # type: ignore[method-assign]

    assert output.continuous_latents is not None
    assert output.continuous_latents.shape == (1, 5, 8)
    assert calls == 6


def test_checkpoint_stage_metadata_rejects_the_wrong_component(tmp_path: Path) -> None:
    source = _continuous()
    checkpoint = tmp_path / "continuous.ckpt"
    torch.save(
        {
            "prism_stage": "continuous_meanflow",
            "prism_representation": {
                "num_discrete_tokens": 2,
                "discrete_vocab_size": 35,
                "continuous_latent_size": 8,
            },
            "state_dict": {f"model.{name}": value for name, value in source.state_dict().items()},
        },
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="Checkpoint stage mismatch"):
        generate_utils.load_checkpoint(_discrete(), checkpoint, use_ema=False)


if __name__ == "__main__":
    print_configured_stage_model_sizes()
