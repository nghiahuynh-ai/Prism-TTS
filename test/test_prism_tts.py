from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import BatchCollate, build_shared_token_layout  # noqa: E402
from models.prism_tts import PrismTTS  # noqa: E402
from utils import generate_utils  # noqa: E402
from utils import model_utils as MU  # noqa: E402


DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "config" / "model.yaml"


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def configured_model_parameter_counts(model: PrismTTS) -> dict[str, int]:
    """Return a non-overlapping parameter breakdown for the configured model."""
    counts = {
        "backbone.embed_tokens": _parameter_count(model.backbone.embed_tokens),
        "backbone.layers": _parameter_count(model.backbone.layers),
        "backbone.norm": _parameter_count(model.backbone.norm),
        "discrete_frame_proj": _parameter_count(model.discrete_frame_proj),
        "continuous_proj": _parameter_count(model.continuous_proj),
        "discrete_lm_head": _parameter_count(model.discrete_lm_head),
        "continuous_prior_head": _parameter_count(model.continuous_prior_head),
        "discrete_stream_embeddings": model.discrete_stream_embeddings.numel(),
        "flow_head": _parameter_count(model.flow_head),
    }
    total = _parameter_count(model)
    covered = sum(counts.values())
    if covered != total:
        raise RuntimeError(
            f"Parameter report does not cover the full model: covered={covered}, total={total}."
        )
    return counts


def _resolve_model_config(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        cwd_candidate = (Path.cwd() / candidate).resolve()
        candidate = cwd_candidate if cwd_candidate.is_file() else (PROJECT_ROOT / candidate).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Model config not found: {candidate}")
    return candidate


def print_configured_model_report(model_config_path: Path, *, device: str = "meta") -> None:
    """Build from model.yaml and print its architecture and parameter breakdown."""
    resolved_config = _resolve_model_config(model_config_path)
    model_config = generate_utils.read_yaml(resolved_config)
    with torch.device(device):
        model = generate_utils.build_model(model_config)

    counts = configured_model_parameter_counts(model)
    total = _parameter_count(model)
    component_width = max(map(len, counts))

    print(f"Model config: {resolved_config}")
    print(f"Construction device: {device}")
    print("\nArchitecture:")
    print(model)
    print("\nParameter counts:")
    print(f"  {'total':<{component_width}} {total:>15,}")
    for component, count in counts.items():
        ratio = 100.0 * count / total
        print(f"  {component:<{component_width}} {count:>15,} ({ratio:6.2f}%)")


def _parse_report_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the PrismTTS architecture and parameter counts from model.yaml."
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Path to model YAML config.",
    )
    parser.add_argument(
        "--device",
        choices=("meta", "cpu"),
        default="meta",
        help="Build on meta for reporting without parameter allocation, or cpu for a real instance.",
    )
    return parser.parse_args()


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


def _model(**kwargs: object) -> PrismTTS:
    return PrismTTS(
        llama_config=_config(),
        num_discrete_tokens=2,
        discrete_vocab_size=35,
        continuous_latent_size=8,
        flow_num_res_blocks=2,
        flow_sample_steps=2,
        **kwargs,
    )


def _sample(*, target_offset: int = 0) -> dict[str, torch.Tensor]:
    return {
        "text_prompt": torch.tensor([40, 41], dtype=torch.long),
        "discrete_prompt": torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long),
        "continuous_prompt": torch.tensor(
            [[0.1] * 8, [0.2] * 8, [0.3] * 8], dtype=torch.float32
        ),
        "text_target": torch.tensor([42, 43], dtype=torch.long),
        "discrete_target": torch.tensor(
            [[7 + target_offset, 8 + target_offset], [9 + target_offset, 10 + target_offset], [11 + target_offset, 12 + target_offset]],
            dtype=torch.long,
        ),
        "continuous_target": torch.tensor(
            [[0.4 + target_offset] * 8, [0.5 + target_offset] * 8, [0.6 + target_offset] * 8],
            dtype=torch.float32,
        ),
    }


def _batch(*samples: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return BatchCollate(discrete_token_count=32)(list(samples))


def _forward(model: PrismTTS, batch: dict[str, torch.Tensor]):
    return model(
        flat_token_ids=batch["flat_token_ids"],
        flat_discrete_values=batch["flat_discrete_values"],
        flat_continuous_values=batch["flat_continuous_values"],
        flat_token_type_ids=batch["flat_token_type_ids"],
        flat_target_discrete_values=batch["flat_target_discrete_values"],
        flat_target_continuous_values=batch["flat_target_continuous_values"],
        flat_target_block_ids=batch["flat_target_block_ids"],
        flat_target_block_counts=batch["flat_target_block_counts"],
        attention_mask=batch["attention_mask"],
    )


def test_forward_computes_fused_ar_losses() -> None:
    torch.manual_seed(0)
    model = _model()
    batch = _batch(_sample(), _sample(target_offset=3))
    outputs = _forward(model, batch)

    assert torch.isfinite(outputs.loss)
    assert torch.isfinite(outputs.discrete_loss)
    assert torch.isfinite(outputs.continuous_loss)
    assert torch.isfinite(outputs.flow_loss)
    assert batch["flat_target_block_counts"].tolist() == [4, 4]
    assert int((batch["flat_target_block_ids"] >= 0).sum().item()) == 8


def test_continuous_latent_normalization_keeps_structural_zeros() -> None:
    model = _model(
        normalize_continuous_latents=True,
        continuous_latent_mean=1.0,
        continuous_latent_std=2.0,
    )
    batch = _batch(_sample())
    flat = MU.build_autoregressive_batch_from_collate(
        flat_token_ids=batch["flat_token_ids"],
        flat_discrete_values=batch["flat_discrete_values"],
        flat_continuous_values=batch["flat_continuous_values"],
        flat_token_type_ids=batch["flat_token_type_ids"],
        flat_target_discrete_values=batch["flat_target_discrete_values"],
        flat_target_continuous_values=batch["flat_target_continuous_values"],
        flat_target_block_ids=batch["flat_target_block_ids"],
        flat_target_block_counts=batch["flat_target_block_counts"],
        attention_mask=batch["attention_mask"],
        num_discrete_tokens=model.num_discrete_tokens,
        continuous_latent_size=model.continuous_latent_size,
    )

    normalized = model._normalize_autoregressive_batch(flat)
    first_prompt_pos = torch.nonzero(
        flat.token_type_ids[0] == MU.SPEECH_FRAME_TOKEN_TYPE,
        as_tuple=False,
    )[0].item()
    first_target_pos = torch.nonzero(flat.target_block_ids[0] == 0, as_tuple=False).item()
    eos_target_pos = torch.nonzero(flat.target_block_ids[0] == 3, as_tuple=False).item()

    assert torch.allclose(
        normalized.continuous_values[0, first_prompt_pos],
        torch.full((8,), -0.45),
    )
    assert torch.allclose(
        normalized.target_continuous_values[0, first_target_pos],
        torch.full((8,), -0.3),
    )
    assert torch.equal(
        normalized.target_continuous_values[0, eos_target_pos],
        torch.zeros(8),
    )


def test_first_frame_prediction_cannot_see_future_target_frames() -> None:
    torch.manual_seed(1)
    model = _model().eval()
    first = _batch(_sample())
    changed = _sample()
    changed["discrete_target"][1:] = torch.tensor([[20, 21], [22, 23]], dtype=torch.long)
    changed["continuous_target"][1:] = 9.0
    second = _batch(changed)

    def first_logits(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        flat = MU.build_autoregressive_batch_from_collate(
            flat_token_ids=batch["flat_token_ids"],
            flat_discrete_values=batch["flat_discrete_values"],
            flat_continuous_values=batch["flat_continuous_values"],
            flat_token_type_ids=batch["flat_token_type_ids"],
            flat_target_discrete_values=batch["flat_target_discrete_values"],
            flat_target_continuous_values=batch["flat_target_continuous_values"],
            flat_target_block_ids=batch["flat_target_block_ids"],
            flat_target_block_counts=batch["flat_target_block_counts"],
            attention_mask=batch["attention_mask"],
            num_discrete_tokens=model.num_discrete_tokens,
            continuous_latent_size=model.continuous_latent_size,
        )
        with torch.no_grad():
            hidden = model._encode(flat)
            discrete_hidden, _ = model._split_hidden(hidden)
            position = torch.nonzero(flat.target_block_ids[0] == 0, as_tuple=False).item()
            return model._discrete_logits(discrete_hidden[0, position]).detach()

    assert torch.allclose(first_logits(first), first_logits(second), atol=1e-6, rtol=1e-6)


def test_generation_is_deterministic_with_fixed_seed_and_uses_cache() -> None:
    torch.manual_seed(2)
    model = _model().eval()
    batch = _batch(_sample(), _sample(target_offset=2))
    kwargs = dict(
        text_prompt=batch["text_prompt"],
        discrete_prompt=batch["discrete_prompt"].transpose(1, 2),
        continuous_prompt=batch["continuous_prompt"],
        text_target=batch["text_target"],
        text_prompt_lengths=batch["text_prompt_lengths"],
        speech_prompt_lengths=batch["speech_prompt_lengths"],
        text_target_lengths=batch["text_target_lengths"],
        max_new_blocks=3,
        do_sample=False,
        flow_num_steps=2,
    )
    original_run_backbone = model._run_backbone
    calls: list[tuple[bool, bool]] = []

    def tracked_run_backbone(**call_kwargs: object):
        calls.append(
            (
                bool(call_kwargs.get("use_cache", False)),
                call_kwargs.get("past_key_values") is not None,
            )
        )
        return original_run_backbone(**call_kwargs)  # type: ignore[arg-type]

    model._run_backbone = tracked_run_backbone  # type: ignore[method-assign]
    torch.manual_seed(9)
    try:
        first = model.generate(**kwargs)
    finally:
        model._run_backbone = original_run_backbone  # type: ignore[method-assign]
    torch.manual_seed(9)
    second = model.generate(**kwargs)

    assert first.discrete_ids.shape == (2, 2, 3)
    assert first.continuous_latents.shape == (2, 3, 8)
    assert torch.equal(first.discrete_ids, second.discrete_ids)
    assert torch.allclose(first.continuous_latents, second.continuous_latents)
    assert len(first.discrete_logits) == 3
    assert calls[0] == (True, False)
    assert any(use_cache and has_past for use_cache, has_past in calls[1:])


def test_generation_stops_when_the_discrete_head_emits_eos() -> None:
    torch.manual_seed(3)
    model = _model().eval()
    _, eos_id, _, _ = build_shared_token_layout(32)
    batch = _batch(_sample())
    original_sampler = model._sample_discrete_ids

    def emit_eos(logits: torch.Tensor, **_: object) -> torch.LongTensor:
        return torch.full(logits.shape[:-1], eos_id, dtype=torch.long, device=logits.device)

    model._sample_discrete_ids = emit_eos  # type: ignore[method-assign]
    try:
        generated = model.generate(
            text_prompt=batch["text_prompt"],
            discrete_prompt=batch["discrete_prompt"].transpose(1, 2),
            continuous_prompt=batch["continuous_prompt"],
            text_target=batch["text_target"],
            max_new_blocks=6,
            do_sample=False,
            flow_num_steps=2,
            discrete_eos_token_id=eos_id,
        )
    finally:
        model._sample_discrete_ids = original_sampler  # type: ignore[method-assign]

    assert generated.discrete_ids.shape == (1, 2, 1)
    assert torch.equal(generated.discrete_ids[0, :, 0], torch.tensor([eos_id, eos_id]))
    assert torch.equal(generated.continuous_latents, torch.zeros_like(generated.continuous_latents))


def test_generation_denormalizes_continuous_latents_before_return() -> None:
    torch.manual_seed(4)
    model = _model(
        normalize_continuous_latents=True,
        continuous_latent_mean=3.0,
        continuous_latent_std=2.0,
    ).eval()
    batch = _batch(_sample())
    original_sampler = model._sample_discrete_ids
    original_sample_continuous = model.sample_continuous_latent

    def emit_regular(logits: torch.Tensor, **_: object) -> torch.LongTensor:
        return torch.ones(logits.shape[:-1], dtype=torch.long, device=logits.device)

    def emit_zero_latent(cond: torch.Tensor, num_steps: int | None = None) -> torch.FloatTensor:
        del num_steps
        return torch.zeros(
            (*cond.shape[:-1], model.continuous_latent_size),
            dtype=cond.dtype,
            device=cond.device,
        )

    model._sample_discrete_ids = emit_regular  # type: ignore[method-assign]
    model.sample_continuous_latent = emit_zero_latent  # type: ignore[method-assign]
    try:
        generated = model.generate(
            text_prompt=batch["text_prompt"],
            discrete_prompt=batch["discrete_prompt"].transpose(1, 2),
            continuous_prompt=batch["continuous_prompt"],
            text_target=batch["text_target"],
            max_new_blocks=1,
            do_sample=False,
            flow_num_steps=2,
        )
    finally:
        model._sample_discrete_ids = original_sampler  # type: ignore[method-assign]
        model.sample_continuous_latent = original_sample_continuous  # type: ignore[method-assign]

    assert generated.continuous_latents.shape == (1, 1, 8)
    assert torch.allclose(generated.continuous_latents, torch.full((1, 1, 8), 3.0))


def test_generation_defaults_to_the_configured_eos_id() -> None:
    torch.manual_seed(4)
    model = _model().eval()
    batch = _batch(_sample())
    original_sampler = model._sample_discrete_ids

    def emit_configured_eos(logits: torch.Tensor, **_: object) -> torch.LongTensor:
        return torch.full(
            logits.shape[:-1],
            model.eos_token_id,
            dtype=torch.long,
            device=logits.device,
        )

    model._sample_discrete_ids = emit_configured_eos  # type: ignore[method-assign]
    try:
        generated = model.generate(
            text_prompt=batch["text_prompt"],
            discrete_prompt=batch["discrete_prompt"].transpose(1, 2),
            continuous_prompt=batch["continuous_prompt"],
            text_target=batch["text_target"],
            max_new_blocks=6,
            do_sample=False,
            flow_num_steps=2,
        )
    finally:
        model._sample_discrete_ids = original_sampler  # type: ignore[method-assign]

    assert generated.discrete_ids.shape == (1, 2, 1)
    assert torch.equal(
        generated.discrete_ids[0, :, 0],
        torch.tensor([model.eos_token_id, model.eos_token_id]),
    )


def test_parallel_generation_is_rejected_for_the_ar_model() -> None:
    model = _model().eval()
    batch = _batch(_sample())
    with pytest.raises(ValueError, match="{'ar', 'causal'}"):
        model.generate(
            text_prompt=batch["text_prompt"],
            discrete_prompt=batch["discrete_prompt"].transpose(1, 2),
            continuous_prompt=batch["continuous_prompt"],
            text_target=batch["text_target"],
            generation_method="parallel",
        )


def test_parameter_component_counts_cover_the_model() -> None:
    model = _model()
    counts = configured_model_parameter_counts(model)
    assert sum(counts.values()) == _parameter_count(model)


def test_model_yaml_architecture_and_parameter_report() -> None:
    """Exercise the configured-model report without allocating its parameters."""
    print_configured_model_report(DEFAULT_MODEL_CONFIG)


if __name__ == "__main__":
    report_args = _parse_report_args()
    print_configured_model_report(report_args.model_config, device=report_args.device)
