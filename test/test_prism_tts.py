from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import BatchCollate, build_shared_token_layout
from models.prism_tts import PrismTTS
from utils import generate_utils as GU
from utils import model_utils as MU

DISCRETE_TOKEN_COUNT = 16
CONTINUOUS_DIM = 4
EOT, EOS, PAD, TEXT_OFFSET = build_shared_token_layout(DISCRETE_TOKEN_COUNT)
VOCAB_SIZE = TEXT_OFFSET + 20
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"


def build_tiny_model(sample_steps: int = 1, tangent_warmup_steps: int = 0) -> PrismTTS:
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        pad_token_id=PAD,
        eos_token_id=EOS,
        use_cache=False,
        tie_word_embeddings=False,
        _attn_implementation="eager",
    )
    return PrismTTS(
        llama_config=cfg,
        continuous_latent_size=CONTINUOUS_DIM,
        flow_num_res_blocks=2,
        sample_steps=sample_steps,
        tangent_warmup_steps=tangent_warmup_steps,
        short_context_layers=1,
        short_context_window=4,
    )


def build_config_model() -> PrismTTS:
    torch.manual_seed(0)
    return GU.build_model(GU.read_yaml(MODEL_CONFIG_PATH))


def make_sample(lt_p, lp, lt_t, lg):
    return {
        "text_prompt": torch.randint(TEXT_OFFSET, VOCAB_SIZE, (lt_p,)),
        "discrete_prompt": torch.randint(0, DISCRETE_TOKEN_COUNT, (lp, 1)),
        "continuous_prompt": torch.randn(lp, CONTINUOUS_DIM),
        "text_target": torch.randint(TEXT_OFFSET, VOCAB_SIZE, (lt_t,)),
        "discrete_target": torch.randint(0, DISCRETE_TOKEN_COUNT, (lg, 1)),
        "continuous_target": torch.randn(lg, CONTINUOUS_DIM),
    }


def make_batch():
    collate = BatchCollate(discrete_token_count=DISCRETE_TOKEN_COUNT)
    return collate([make_sample(3, 4, 5, 6), make_sample(2, 3, 4, 5)])


def _print_model(model: PrismTTS) -> int:
    def emit(value: object = "") -> None:
        print(value, flush=True)

    total_params = sum(param.numel() for param in model.parameters())
    emit("=" * 100)
    emit("Model:")
    emit(model)
    emit("-" * 100)
    emit(f"Total parameters: {total_params:,}")
    emit("=" * 100)
    return total_params


def _forward(model, batch):
    return model(
        flat_token_ids=batch["flat_token_ids"],
        flat_continuous_values=batch["flat_continuous_values"],
        flat_token_type_ids=batch["flat_token_type_ids"],
        flat_target_block_ids=batch["flat_target_block_ids"],
        flat_target_block_counts=batch["flat_target_block_counts"],
        attention_mask=batch["attention_mask"],
        return_dict=True,
    )


def test_print_config_model(pytestconfig):
    torch.manual_seed(0)
    model = build_config_model().eval()
    capture_manager = pytestconfig.pluginmanager.getplugin("capturemanager")
    if capture_manager is None:
        total_params = _print_model(model)
    else:
        with capture_manager.global_and_fixture_disabled():
            total_params = _print_model(model)
    assert total_params == sum(p.numel() for p in model.parameters())
    assert total_params > 0


def test_forward_returns_finite_losses_and_backward_flows():
    torch.manual_seed(0)
    model = build_tiny_model().train()
    out = _forward(model, make_batch())
    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.consistency_loss)
    assert torch.isfinite(out.eos_loss)

    out.loss.backward()
    for name in ("continuous_proj", "eos_head", "flow_head", "short_encoder"):
        module = getattr(model, name)
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f"no gradient reached {name}"
        assert any(g.abs().sum() > 0 for g in grads), f"zero gradient in {name}"
    # The adaptive-weighting logvar head lives inside FlowHead and must receive gradient.
    assert model.flow_head.logvar_linear.weight.grad is not None
    backbone_grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert backbone_grads and any(g.abs().sum() > 0 for g in backbone_grads)


def test_latent_stats_update_during_training():
    torch.manual_seed(0)
    model = build_tiny_model().train()
    assert not bool(model.stats_initialized.item())
    _forward(model, make_batch())
    assert bool(model.stats_initialized.item())
    assert model.emb_std.mean().item() != pytest.approx(1.0, abs=1e-4)
    assert int(model.train_step.item()) == 1


def test_eval_forward_does_not_update_stats():
    torch.manual_seed(0)
    model = build_tiny_model().eval()
    _forward(model, make_batch())
    assert not bool(model.stats_initialized.item())
    assert int(model.train_step.item()) == 0


def test_generate_shapes_and_finite():
    torch.manual_seed(0)
    model = build_tiny_model().eval()
    gen = model.generate(
        text_prompt=torch.randint(TEXT_OFFSET, VOCAB_SIZE, (2, 3)),
        continuous_prompt=torch.randn(2, 4, CONTINUOUS_DIM),
        text_target=torch.randint(TEXT_OFFSET, VOCAB_SIZE, (2, 5)),
        text_prompt_lengths=torch.tensor([3, 3]),
        speech_prompt_lengths=torch.tensor([4, 4]),
        text_target_lengths=torch.tensor([5, 5]),
        max_new_frames=8,
        eos_threshold=2.0,  # never stop -> full length
        return_dict=True,
    )
    assert gen.continuous_latents.shape == (2, 8, CONTINUOUS_DIM)
    assert gen.eos_scores.shape == (2, 8)
    assert torch.isfinite(gen.continuous_latents).all()
    assert gen.text_ids.shape == (2, 5)


def test_generate_stops_on_eos_threshold():
    torch.manual_seed(0)
    model = build_tiny_model().eval()
    gen = model.generate(
        text_prompt=torch.randint(TEXT_OFFSET, VOCAB_SIZE, (2, 3)),
        continuous_prompt=torch.randn(2, 4, CONTINUOUS_DIM),
        text_target=torch.randint(TEXT_OFFSET, VOCAB_SIZE, (2, 4)),
        text_prompt_lengths=torch.tensor([3, 3]),
        speech_prompt_lengths=torch.tensor([4, 4]),
        text_target_lengths=torch.tensor([4, 4]),
        max_new_frames=8,
        eos_threshold=-1.0,  # any probability exceeds -1 -> stop after one frame
        return_dict=True,
    )
    assert gen.continuous_latents.shape == (2, 1, CONTINUOUS_DIM)


def test_generation_first_frame_conditioning_matches_teacher_forcing():
    """Causal parity: the conditioning for target frame 0 is the EOT-after-text-target
    hidden state and is independent of the (future) target frames."""
    torch.manual_seed(0)
    model = build_tiny_model().eval()

    text_prompt = torch.randint(TEXT_OFFSET, VOCAB_SIZE, (1, 3))
    continuous_prompt = torch.randn(1, 4, CONTINUOUS_DIM)
    text_target = torch.randint(TEXT_OFFSET, VOCAB_SIZE, (1, 5))
    continuous_target = torch.randn(1, 6, CONTINUOUS_DIM)
    lengths = dict(
        text_prompt_lengths=torch.tensor([3]),
        speech_prompt_lengths=torch.tensor([4]),
        text_target_lengths=torch.tensor([5]),
    )

    flat_full = MU.assemble_flat_batch(
        text_prompt=text_prompt,
        continuous_prompt=continuous_prompt,
        text_target=text_target,
        continuous_target=continuous_target,
        speech_target_lengths=torch.tensor([6]),
        pad_token_id=model.pad_token_id,
        eos_token_id=model.eos_token_id,
        eot_token_id=model.eot_token_id,
        continuous_latent_size=CONTINUOUS_DIM,
        **lengths,
    )
    with torch.no_grad():
        h_full = model._encode_conditioning(
            flat_full.token_ids,
            flat_full.token_type_ids,
            model._normalize(flat_full.continuous_values),
            model._normalize(flat_full.continuous_values),
            flat_full.attention_mask,
        )
    flat_prefix = MU.assemble_flat_batch(
        text_prompt=text_prompt,
        continuous_prompt=continuous_prompt,
        text_target=text_target,
        continuous_target=continuous_target[:, :0, :],
        speech_target_lengths=torch.tensor([0]),
        pad_token_id=model.pad_token_id,
        eos_token_id=model.eos_token_id,
        eot_token_id=model.eot_token_id,
        continuous_latent_size=CONTINUOUS_DIM,
        **lengths,
    )
    with torch.no_grad():
        h_prefix = model._encode_conditioning(
            flat_prefix.token_ids,
            flat_prefix.token_type_ids,
            model._normalize(flat_prefix.continuous_values),
            model._normalize(flat_prefix.continuous_values),
            flat_prefix.attention_mask,
        )

    prefix_len = flat_prefix.token_ids.shape[1]
    first_frame_pos = int((flat_full.target_block_ids[0] == 0).nonzero()[0].item())
    cond_full = h_full[0, first_frame_pos - 1]
    cond_prefix = h_prefix[0, prefix_len - 1]
    assert torch.allclose(cond_full, cond_prefix, atol=1e-5)


def test_backbone_noise_injection_helper():
    torch.manual_seed(0)
    clean = torch.randn(10, CONTINUOUS_DIM)
    noised = MU.inject_continuous_backbone_noise(clean)
    assert noised.shape == clean.shape
    assert torch.isfinite(noised).all()
    # With k ~ U(0, 1) noise is injected, so the output differs from the clean input.
    assert not torch.allclose(noised, clean)


def test_backbone_noise_flag_is_train_only_and_finite():
    torch.manual_seed(0)
    batch = make_batch()

    # Enabled: training forward stays finite; disabled path also runs.
    for flag in (True, False):
        model = build_tiny_model()
        model.inject_backbone_noise = flag
        model.train()
        out = _forward(model, batch)
        assert torch.isfinite(out.loss)

    # The flag only affects training; eval never injects backbone noise.
    model = build_tiny_model().eval()
    assert model.inject_backbone_noise is True
    out = _forward(model, batch)
    assert torch.isfinite(out.loss)


def test_generate_e2e_tensor_with_injected_components():
    torch.manual_seed(0)
    model = build_tiny_model().eval()

    def fake_tokenizer(text: str):
        return [TEXT_OFFSET + (ord(c) % 20) for c in text][:6] or [TEXT_OFFSET]

    def fake_encoder(_prompt):
        return torch.randn(5, CONTINUOUS_DIM)

    gen = model.generate_e2e(
        raw_text_prompt="hello",
        raw_speech_prompt=object(),
        raw_text_target="world",
        text_tokenizer=fake_tokenizer,
        speech_encoder=fake_encoder,
        output_type="tensor",
        return_dict=True,
        max_new_frames=5,
        eos_threshold=2.0,
    )
    assert gen.continuous_latents.shape == (1, 5, CONTINUOUS_DIM)
    assert torch.isfinite(gen.continuous_latents).all()


if __name__ == "__main__":
    torch.manual_seed(0)
    _print_model(build_config_model().eval())
