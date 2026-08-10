from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import (  # noqa: E402
    ASR_TASK_ID,
    PREDICTION_DISCRETE,
    PREDICTION_TEXT,
    TTS_TASK_ID,
    BatchCollate,
    MultitaskBatchCollate,
)
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


def _multitask_discrete() -> PrismDiscreteTTS:
    return PrismDiscreteTTS(
        llama_config=_config(),
        num_discrete_tokens=2,
        discrete_vocab_size=35,
        tts_token_id=94,
        asr_token_id=95,
    )


def _multitask_collate(task_mode: str) -> MultitaskBatchCollate:
    return MultitaskBatchCollate(
        discrete_token_count=32,
        tts_token_id=94,
        asr_token_id=95,
        task_mode=task_mode,
        text_token_offset=35,
        text_vocab_size=59,
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


def test_multitask_collate_builds_tts_and_asr_with_typed_targets() -> None:
    batch = _multitask_collate("both")([_sample()])
    assert batch["flat_task_ids"].tolist() == [TTS_TASK_ID, ASR_TASK_ID]

    # TTS prepends its task token and retains one-frame-shifted speech targets.
    tts_mask = batch["attention_mask"][0]
    tts_len = int(tts_mask.sum().item())
    assert batch["flat_token_ids"][0, :9].tolist() == [94, 40, 41, 32, 34, 34, 33, 42, 32]
    assert batch["flat_prediction_kind"][0, :tts_len].tolist() == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        PREDICTION_DISCRETE,
        PREDICTION_DISCRETE,
        PREDICTION_DISCRETE,
        PREDICTION_DISCRETE,
    ]
    assert batch["flat_target_block_ids"][0, :tts_len].tolist() == [
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        0,
        1,
        2,
        3,
    ]
    assert batch["flat_target_discrete_values"][0, 8:12].tolist() == [
        [5, 6],
        [7, 8],
        [9, 10],
        [33, 33],
    ]

    # ASR observes the full speech utterance, then scalar EOS anchors target
    # text.  The last transcript input anchors its EOT target.
    asr_mask = batch["attention_mask"][1]
    asr_len = int(asr_mask.sum().item())
    assert batch["flat_token_ids"][1, :asr_len].tolist() == [95, 34, 34, 34, 33, 42]
    assert batch["flat_token_type_ids"][1, :asr_len].tolist() == [0, 3, 3, 3, 0, 0]
    assert batch["flat_prediction_kind"][1, :asr_len].tolist() == [
        0,
        0,
        0,
        0,
        PREDICTION_TEXT,
        PREDICTION_TEXT,
    ]
    assert batch["flat_target_text_values"][1, 4:6].tolist() == [42, 32]
    assert torch.all(batch["flat_target_block_ids"][1, :asr_len] == -1)


def test_multitask_random_collation_is_per_sample_and_seed_reproducible() -> None:
    samples = [_sample(), _sample(), _sample(), _sample()]
    torch.manual_seed(17)
    first = _multitask_collate("random")(samples)
    torch.manual_seed(17)
    second = _multitask_collate("random")(samples)
    assert first["flat_task_ids"].shape == (4,)
    assert torch.equal(first["flat_task_ids"], second["flat_task_ids"])
    assert set(first["flat_task_ids"].tolist()).issubset({TTS_TASK_ID, ASR_TASK_ID})


def test_multitask_model_uses_typed_losses_and_is_causal_for_asr_text() -> None:
    torch.manual_seed(4)
    model = _multitask_discrete().eval()
    sample = _sample()
    sample["text_target"] = torch.tensor([42, 43], dtype=torch.long)
    first = _multitask_collate("both")([sample])
    changed_sample = {key: value.clone() for key, value in sample.items()}
    changed_sample["text_target"] = torch.tensor([42, 44], dtype=torch.long)
    second = _multitask_collate("both")([changed_sample])

    outputs = model(
        flat_token_ids=first["flat_token_ids"],
        flat_discrete_values=first["flat_discrete_values"],
        flat_token_type_ids=first["flat_token_type_ids"],
        flat_target_discrete_values=first["flat_target_discrete_values"],
        flat_target_block_ids=first["flat_target_block_ids"],
        flat_target_text_values=first["flat_target_text_values"],
        flat_prediction_kind=first["flat_prediction_kind"],
        flat_task_ids=first["flat_task_ids"],
        attention_mask=first["attention_mask"],
    )
    assert torch.isfinite(outputs.loss)
    assert torch.isfinite(outputs.tts_loss)
    assert torch.isfinite(outputs.asr_loss)
    assert torch.allclose(outputs.loss, (outputs.tts_loss + outputs.asr_loss) / 2.0)
    assert outputs.tts_sample_count.item() == 1
    assert outputs.asr_sample_count.item() == 1

    def first_asr_text_logits(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            hidden = model._encode(
                token_ids=batch["flat_token_ids"],
                discrete_values=batch["flat_discrete_values"],
                token_type_ids=batch["flat_token_type_ids"],
                attention_mask=batch["attention_mask"],
            )
        asr_index = 1
        first_text_anchor = torch.nonzero(
            batch["flat_prediction_kind"][asr_index] == PREDICTION_TEXT,
            as_tuple=False,
        )[0].item()
        return model._text_logits(hidden[asr_index, first_text_anchor]).detach()

    # The second transcript character is a future input position, so it cannot
    # affect the EOS anchor used to predict the first character.
    assert torch.allclose(
        first_asr_text_logits(first), first_asr_text_logits(second), atol=1e-6, rtol=1e-6
    )


def test_multitask_asr_generation_starts_from_the_asr_prefix() -> None:
    torch.manual_seed(5)
    model = _multitask_discrete().eval()
    speech = _sample()["discrete_target"].unsqueeze(0)
    prefix_ids, _, prefix_types = model._build_asr_generation_prefix(
        discrete_speech=speech[0]
    )
    assert prefix_ids[0].tolist() == [95, 34, 34, 34, 33]
    assert prefix_types[0].tolist() == [0, 3, 3, 3, 0]

    transcription = model.transcribe(
        discrete_speech=speech,
        max_new_tokens=2,
        do_sample=False,
    )
    assert transcription.text_ids is not None
    assert transcription.text_ids.shape[0] == 1
    if transcription.text_ids.numel():
        assert torch.all((transcription.text_ids >= 35) & (transcription.text_ids < 94))


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
