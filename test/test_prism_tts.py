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


def build_tiny_model(
    sample_steps: int = 1,
    head_mode: str = "flowmatch",
    noise_mode: str = "ramp",
    default_block_size: int = 2,
    gradient_checkpointing: bool = False,
) -> PrismTTS:
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
        short_context_layers=1,
        short_context_window=4,
        head_mode=head_mode,
        noise_mode=noise_mode,
        default_block_size=default_block_size,
        gradient_checkpointing=gradient_checkpointing,
    )


def build_config_model() -> PrismTTS:
    torch.manual_seed(0)
    return GU.build_model(GU.read_yaml(MODEL_CONFIG_PATH))


def make_sample(lt_p, lp, lt_t, lg):
    # Prompt fields are still required by the collate validator but are ignored by
    # the single-utterance flat layout (only text_target + continuous_target used).
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
    return collate([make_sample(3, 4, 5, 6), make_sample(2, 3, 4, 8)])


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
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0
    assert model.head_mode == "meanflow"
    assert model.gradient_checkpointing
    # The detached JVP target bypasses checkpointing per call, while the
    # differentiable MeanFlow prediction checkpoints its residual blocks.
    assert model.flow_head.grad_checkpointing


def test_single_utterance_flat_layout():
    batch = make_batch()
    token_ids = batch["flat_token_ids"]
    attention = batch["attention_mask"]
    # sample 0: text_target(5) + EOT(1) + frames(6) = 12; sample 1: 4 + 1 + 8 = 13.
    assert int(attention[0].sum()) == 5 + 1 + 6
    assert int(attention[1].sum()) == 4 + 1 + 8
    # No literal EOS token anywhere; exactly one EOT per sample.
    assert int((token_ids == EOS).sum()) == 0
    assert int((token_ids[0, : int(attention[0].sum())] == EOT).sum()) == 1


@pytest.mark.parametrize("head_mode", ["flowmatch", "meanflow"])
def test_forward_returns_finite_losses_and_backward_flows(head_mode):
    torch.manual_seed(0)
    model = build_tiny_model(head_mode=head_mode).train()
    out = _forward(model, make_batch())
    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.flow_loss)
    assert torch.isfinite(out.eos_loss)

    out.loss.backward()
    for name in ("continuous_proj", "eos_head", "flow_head", "short_encoder"):
        module = getattr(model, name)
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f"no gradient reached {name}"
        assert any(g.abs().sum() > 0 for g in grads), f"zero gradient in {name}"
    assert model.masked_speech_embedding.grad is not None
    assert model.masked_speech_embedding.grad.abs().sum() > 0
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


def test_backbone_is_bidirectional():
    model = build_tiny_model().eval()
    embeds = torch.randn(1, 5, model.hidden_size)
    mask = model._backbone_attention_mask(torch.ones(1, 5, dtype=torch.bool), embeds)
    # The singleton query dimension broadcasts over all queries, avoiding a
    # quadratic [B, 1, L, L] allocation for MAR attention.
    assert mask.shape == (1, 1, 1, 5)
    assert mask.untyped_storage().nbytes() == 5 * mask.element_size()
    # A query may attend to a future key (no causal triangle).
    assert mask[0, 0, 0, 4].item() == 0.0


@pytest.mark.parametrize("head_mode", ["flowmatch", "meanflow"])
def test_gradient_checkpointing_backward(head_mode):
    model = build_tiny_model(
        head_mode=head_mode,
        gradient_checkpointing=True,
    ).train()
    out = _forward(model, make_batch())
    out.loss.backward()
    assert model.backbone.gradient_checkpointing
    assert model.short_encoder.gradient_checkpointing
    assert model.continuous_proj.weight.grad is not None


def test_meanflow_jvp_target_does_not_build_reverse_graph(monkeypatch):
    model = build_tiny_model(
        head_mode="meanflow",
        gradient_checkpointing=True,
    ).train()
    real_jvp = torch.func.jvp
    grad_enabled_during_jvp = None

    def recording_jvp(*args, **kwargs):
        nonlocal grad_enabled_during_jvp
        grad_enabled_during_jvp = torch.is_grad_enabled()
        return real_jvp(*args, **kwargs)

    monkeypatch.setattr(torch.func, "jvp", recording_jvp)
    loss = model._meanflow_loss(
        torch.randn(6, CONTINUOUS_DIM),
        torch.randn(6, model.hidden_size, requires_grad=True),
    )
    loss.backward()

    assert grad_enabled_during_jvp is False
    assert model.flow_head.grad_checkpointing
    assert model.flow_head.input_proj.weight.grad is not None


def test_masked_positions_do_not_leak_targets():
    torch.manual_seed(0)
    model = build_tiny_model().eval()
    batch = make_batch()
    flat = MU.build_flat_batch_from_collate(
        flat_token_ids=batch["flat_token_ids"],
        flat_continuous_values=batch["flat_continuous_values"],
        flat_token_type_ids=batch["flat_token_type_ids"],
        flat_target_block_ids=batch["flat_target_block_ids"],
        flat_target_block_counts=batch["flat_target_block_counts"],
        attention_mask=batch["attention_mask"],
        continuous_latent_size=CONTINUOUS_DIM,
    )
    masked = torch.zeros_like(batch["flat_token_type_ids"], dtype=torch.bool)
    masked[0] = (flat.target_block_ids[0] >= 3) & (flat.target_block_ids[0] >= 0)
    norm = model._normalize(flat.continuous_values)
    with torch.no_grad():
        z1 = model._encode_conditioning(
            flat.token_ids, flat.token_type_ids, norm, norm, flat.attention_mask, masked
        )
        perturbed = norm.clone()
        pos = int(torch.nonzero(masked[0], as_tuple=False)[0].item())
        perturbed[0, pos, :] += 1000.0
        z2 = model._encode_conditioning(
            flat.token_ids, flat.token_type_ids, perturbed, perturbed, flat.attention_mask, masked
        )
    assert (z1 - z2).abs().max().item() < 1e-4


@pytest.mark.parametrize("block_size", [1, 3])
def test_generate_shapes_and_finite(block_size):
    torch.manual_seed(0)
    model = build_tiny_model().eval()
    gen = model.generate(
        text_target=torch.randint(TEXT_OFFSET, VOCAB_SIZE, (2, 5)),
        continuous_condition=torch.randn(2, 3, CONTINUOUS_DIM),
        text_target_lengths=torch.tensor([5, 5]),
        condition_lengths=torch.tensor([3, 3]),
        max_new_frames=8,
        block_size=block_size,
        eos_threshold=2.0,  # never stop -> full length
        return_dict=True,
    )
    assert gen.continuous_latents.shape == (2, 8, CONTINUOUS_DIM)
    assert gen.eos_scores.shape == (2, 8)
    assert torch.isfinite(gen.continuous_latents).all()
    assert gen.text_ids.shape == (2, 5)


def test_generate_without_condition_is_plain_tts():
    torch.manual_seed(0)
    model = build_tiny_model().eval()
    gen = model.generate(
        text_target=torch.randint(TEXT_OFFSET, VOCAB_SIZE, (2, 4)),
        text_target_lengths=torch.tensor([4, 4]),
        max_new_frames=5,
        block_size=2,
        eos_threshold=2.0,
        return_dict=True,
    )
    assert gen.continuous_latents.shape == (2, 5, CONTINUOUS_DIM)


def test_generate_stops_on_eos_threshold():
    torch.manual_seed(0)
    model = build_tiny_model().eval()
    gen = model.generate(
        text_target=torch.randint(TEXT_OFFSET, VOCAB_SIZE, (2, 4)),
        continuous_condition=torch.randn(2, 3, CONTINUOUS_DIM),
        text_target_lengths=torch.tensor([4, 4]),
        condition_lengths=torch.tensor([3, 3]),
        max_new_frames=8,
        block_size=1,
        eos_threshold=-1.0,  # any probability exceeds -1 -> stop after one frame
        return_dict=True,
    )
    assert gen.continuous_latents.shape == (2, 1, CONTINUOUS_DIM)


def test_backbone_noise_injection_helper():
    torch.manual_seed(0)
    clean = torch.randn(10, CONTINUOUS_DIM)
    noised = MU.inject_continuous_backbone_noise(clean)
    assert noised.shape == clean.shape
    assert torch.isfinite(noised).all()
    assert not torch.allclose(noised, clean)


def test_context_noise_ramp_keeps_prefix_and_mask_clean():
    torch.manual_seed(0)
    batch_size, seq_len = 1, 10
    latents = torch.randn(batch_size, seq_len, CONTINUOUS_DIM)
    # frames 0..9 are all speech targets; protect first 30% (=3), mask suffix 8..9.
    target_block_ids = torch.arange(seq_len).view(1, seq_len)
    counts = torch.tensor([seq_len])
    masked = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    masked[0, 8:] = True
    speech = torch.ones(batch_size, seq_len, dtype=torch.bool)
    out = MU.apply_context_noise_ramp(
        latents, target_block_ids, counts, masked, speech,
        protected_prefix_ratio=0.3, k_max=1.0, random_intensity=False,
    )
    # Protected prefix (0..2) and masked suffix (8..9) stay exactly clean.
    assert torch.allclose(out[0, :3], latents[0, :3])
    assert torch.allclose(out[0, 8:], latents[0, 8:])
    # Eligible visible region (3..7) is perturbed somewhere.
    assert not torch.allclose(out[0, 3:8], latents[0, 3:8])


def test_backbone_noise_modes_train_only_and_finite():
    torch.manual_seed(0)
    batch = make_batch()
    for noise_mode in ("ramp", "iid", "none"):
        model = build_tiny_model(noise_mode=noise_mode).train()
        out = _forward(model, batch)
        assert torch.isfinite(out.loss)


def test_generate_e2e_tensor_with_injected_components():
    torch.manual_seed(0)
    model = build_tiny_model().eval()

    def fake_tokenizer(text: str):
        return [TEXT_OFFSET + (ord(c) % 20) for c in text][:6] or [TEXT_OFFSET]

    def fake_encoder(_prompt):
        return torch.randn(3, CONTINUOUS_DIM)

    gen = model.generate_e2e(
        raw_text_target="world",
        raw_speech_condition=object(),
        text_tokenizer=fake_tokenizer,
        speech_encoder=fake_encoder,
        output_type="tensor",
        return_dict=True,
        max_new_frames=5,
        eos_threshold=2.0,
    )
    assert gen.continuous_latents.shape == (1, 5, CONTINUOUS_DIM)
    assert torch.isfinite(gen.continuous_latents).all()


def test_generate_e2e_prepends_prompt_transcript():
    torch.manual_seed(0)
    model = build_tiny_model().eval()

    tokenized_texts: list[str] = []

    def fake_tokenizer(text: str):
        tokenized_texts.append(text)
        return [TEXT_OFFSET + (ord(c) % 20) for c in text][:6] or [TEXT_OFFSET]

    def fake_encoder(_prompt):
        return torch.randn(3, CONTINUOUS_DIM)

    gen = model.generate_e2e(
        raw_text_target="world",
        raw_speech_condition=object(),
        raw_text_prompt="hello",
        text_tokenizer=fake_tokenizer,
        speech_encoder=fake_encoder,
        output_type="tensor",
        return_dict=True,
        max_new_frames=4,
        eos_threshold=2.0,
    )
    assert tokenized_texts == ["hello", "world"]
    # text_ids carries prompt + target tokens (5 chars each under fake_tokenizer).
    assert gen.text_ids.shape == (1, 10)
    assert gen.continuous_latents.shape == (1, 4, CONTINUOUS_DIM)
    assert torch.isfinite(gen.continuous_latents).all()


def test_generate_e2e_rejects_prompt_without_condition():
    model = build_tiny_model().eval()

    def fake_tokenizer(text: str):
        return [TEXT_OFFSET + (ord(c) % 20) for c in text][:6] or [TEXT_OFFSET]

    with pytest.raises(ValueError, match="raw_text_prompt"):
        model.generate_e2e(
            raw_text_target="world",
            raw_text_prompt="hello",
            text_tokenizer=fake_tokenizer,
            output_type="tensor",
            max_new_frames=2,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    build_config_model().eval()
