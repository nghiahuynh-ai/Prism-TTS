from __future__ import annotations

from functools import lru_cache
import sys
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ImportError(
        "test/test_prism_tts.py requires PyYAML (`pip install pyyaml`)."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.prism_tts import PrismTTS


MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"
MIN_REASONABLE_PARAM_COUNT = 100_000_000


@lru_cache(maxsize=1)
def load_model_config() -> dict:
    config_data = yaml.safe_load(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config_data, dict):
        raise ValueError(f"Invalid YAML structure in {MODEL_CONFIG_PATH}.")
    model_cfg = config_data.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Missing 'model' mapping in {MODEL_CONFIG_PATH}.")
    return model_cfg


def make_large_config() -> LlamaConfig:
    llama_cfg = dict(load_model_config()["llama_config"])
    config = LlamaConfig(**llama_cfg)
    if "_attn_implementation" in llama_cfg:
        config._attn_implementation = llama_cfg["_attn_implementation"]
    return config


def make_small_config() -> LlamaConfig:
    llama_cfg = dict(load_model_config()["llama_config"])
    llama_cfg.update(
        {
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "pad_token_id": 0,
        }
    )
    config = LlamaConfig(**llama_cfg)
    if "_attn_implementation" in llama_cfg:
        config._attn_implementation = llama_cfg["_attn_implementation"]
    return config


class TestPrismTTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.device = torch.device("cpu")
        cls.llama_config = make_large_config()
        prism_cfg = load_model_config()["prism_tts"]
        cls.num_discrete_tokens = int(prism_cfg["num_discrete_tokens"])
        cls.discrete_vocab_size = int(prism_cfg["discrete_vocab_size"])
        cls.continuous_latent_size = int(prism_cfg["continuous_latent_size"])
        cls.text_vocab_size = int(cls.llama_config.vocab_size)
        cls.model = PrismTTS(
            llama_config=cls.llama_config,
            num_discrete_tokens=cls.num_discrete_tokens,
            discrete_vocab_size=cls.discrete_vocab_size,
            continuous_latent_size=cls.continuous_latent_size,
            flow_num_res_blocks=int(prism_cfg.get("flow_num_res_blocks", 4)),
            flow_model_channels=prism_cfg.get("flow_model_channels"),
            flow_loss_weight=float(prism_cfg.get("flow_loss_weight", 1.0)),
            flow_sample_steps=int(prism_cfg.get("flow_sample_steps", 16)),
        ).to(cls.device)
        cls.model_num_params = sum(p.numel() for p in cls.model.parameters())
        cls._print_model_summary()

    @classmethod
    def _print_model_summary(cls):
        component_param_counts = cls._component_param_counts()
        print("=" * 100)
        print("Model Architecture:")
        print(cls.model)
        print("-" * 100)
        print("Model Config:")
        print(cls.llama_config)
        print("-" * 100)
        print(f"Total Parameters: {cls.model_num_params:,}")
        print("-" * 100)
        print("Component Parameters:")
        for component_name in sorted(component_param_counts):
            print(f"{component_name}: {component_param_counts[component_name]:,}")
        print("=" * 100)

    @classmethod
    def _component_param_counts(cls) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name, param in cls.model.named_parameters():
            component = name.split(".", 1)[0]
            counts[component] = counts.get(component, 0) + param.numel()
        return counts

    def setUp(self):
        torch.manual_seed(0)

    def test_model_has_about_100m_parameters(self):
        self.assertGreaterEqual(self.model_num_params, MIN_REASONABLE_PARAM_COUNT)

    def test_forward_returns_losses_only(self):
        batch_size = 2
        prompt_len = 2
        generated_len = 3
        text_vocab_upper = max(2, min(200, self.text_vocab_size))

        text_prompt = torch.randint(
            0, text_vocab_upper, (batch_size, prompt_len), device=self.device
        )
        discrete_prompt = torch.randint(
            0,
            self.discrete_vocab_size,
            (batch_size, self.num_discrete_tokens, prompt_len),
            device=self.device,
        )
        continuous_prompt = torch.randn(
            batch_size, prompt_len, self.continuous_latent_size, device=self.device
        )

        text_target = torch.randint(
            0, text_vocab_upper, (batch_size, generated_len), device=self.device
        )
        discrete_target = torch.randint(
            0,
            self.discrete_vocab_size,
            (batch_size, self.num_discrete_tokens, generated_len),
            device=self.device,
        )
        continuous_target = torch.randn(
            batch_size, generated_len, self.continuous_latent_size, device=self.device
        )

        # Generated region mask (prompt region is always valid).
        attention_mask = torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 0],
            ],
            dtype=torch.long,
            device=self.device,
        )

        full_text = torch.cat([text_prompt, text_target], dim=1)
        full_discrete = torch.cat([discrete_prompt, discrete_target], dim=2)
        full_continuous = torch.cat([continuous_prompt, continuous_target], dim=1)
        prompt_lengths = torch.full(
            (batch_size,),
            prompt_len,
            dtype=torch.long,
            device=self.device,
        )
        target_lengths = torch.full(
            (batch_size,),
            generated_len,
            dtype=torch.long,
            device=self.device,
        )

        outputs = self.model(
            text=full_text,
            discrete=full_discrete,
            continuous=full_continuous,
            prompt_lengths=prompt_lengths,
            target_lengths=target_lengths,
            attention_mask=attention_mask,
            return_dict=True,
        )

        self.assertIsNotNone(outputs.loss)
        self.assertIsNotNone(outputs.discrete_loss)
        self.assertIsNotNone(outputs.flow_loss)
        self.assertIsNotNone(outputs.text_loss)
        self.assertEqual(
            tuple(outputs.keys()),
            ("loss", "discrete_loss", "flow_loss", "text_loss"),
        )

    def test_build_inputs_embeds_uses_block_major_order(self):
        model = PrismTTS(
            llama_config=make_small_config(),
            num_discrete_tokens=2,
            discrete_vocab_size=256,
            continuous_latent_size=1,
            flow_num_res_blocks=1,
            flow_sample_steps=2,
        ).to(self.device)

        with torch.no_grad():
            model.text_embedding.weight.zero_()
            model.discrete_embedding.weight.zero_()
            model.continuous_proj.weight.zero_()
            model.continuous_proj.bias.zero_()
            model.stream_type_embeddings.zero_()

            model.text_embedding.weight[:, 0] = torch.arange(256, device=self.device)
            model.discrete_embedding.weight[:, 0] = torch.arange(256, device=self.device)
            model.continuous_proj.weight[0, 0] = 1.0

        text_tokens = torch.tensor([[5, 6, 7]], dtype=torch.long, device=self.device)
        # [B, L, N]
        discrete_tokens = torch.tensor(
            [[[11, 12], [21, 22], [31, 32]]],
            dtype=torch.long,
            device=self.device,
        )
        continuous_latents = torch.tensor(
            [[[1.0], [2.0], [3.0]]],
            device=self.device,
        )

        inputs_embeds, _ = model._build_inputs_embeds(
            text_tokens=text_tokens,
            discrete_tokens=discrete_tokens,
            continuous_latents=continuous_latents,
        )
        observed = inputs_embeds[0, :, 0].tolist()
        expected = [5.0, 11.0, 12.0, 1.0, 6.0, 21.0, 22.0, 2.0, 7.0, 31.0, 32.0, 3.0]
        self.assertEqual(observed, expected)

    def test_build_dual_attention_masks_match_expected_layouts(self):
        model = PrismTTS(
            llama_config=make_small_config(),
            num_discrete_tokens=2,
            discrete_vocab_size=32,
            continuous_latent_size=1,
            flow_num_res_blocks=1,
            flow_sample_steps=2,
        ).to(self.device)

        stream_mask, block_mask = model._build_dual_attention_masks(
            total_blocks=2,
            batch_size=1,
            device=self.device,
        )

        expected_stream_mask = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
            ],
            dtype=torch.bool,
            device=self.device,
        )
        expected_block_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
            device=self.device,
        )

        self.assertTrue(torch.equal(stream_mask[0, 0], expected_stream_mask))
        self.assertTrue(torch.equal(block_mask[0, 0], expected_block_mask))

    def test_generate_returns_expected_shapes(self):
        batch_size = 1
        prompt_len = 2
        generated_len = 2
        text_vocab_upper = max(2, min(200, self.text_vocab_size))

        text_prompt = torch.randint(
            0, text_vocab_upper, (batch_size, prompt_len), device=self.device
        )
        discrete_prompt = torch.randint(
            0,
            self.discrete_vocab_size,
            (batch_size, self.num_discrete_tokens, prompt_len),
            device=self.device,
        )
        continuous_prompt = torch.randn(
            batch_size, prompt_len, self.continuous_latent_size, device=self.device
        )
        text_target = torch.randint(
            0, text_vocab_upper, (batch_size, generated_len), device=self.device
        )

        outputs = self.model.generate(
            text_prompt=text_prompt,
            discrete_prompt=discrete_prompt,
            continuous_prompt=continuous_prompt,
            text_target=text_target,
            max_new_blocks=generated_len,
            discrete_eos_token_id=-1,
            do_sample=False,
            return_dict=True,
        )

        self.assertEqual(outputs.text_ids.shape, (batch_size, generated_len))
        self.assertEqual(
            outputs.discrete_ids.shape,
            (batch_size, self.num_discrete_tokens, generated_len),
        )
        self.assertEqual(
            outputs.continuous_latents.shape,
            (batch_size, generated_len, self.continuous_latent_size),
        )
        self.assertEqual(len(outputs.discrete_logits), generated_len)
        self.assertEqual(
            outputs.discrete_logits[0].shape,
            (batch_size, self.num_discrete_tokens, self.discrete_vocab_size),
        )

    def test_generate_with_kv_cache_returns_expected_shapes(self):
        batch_size = 1
        prompt_len = 2
        generated_len = 2
        text_vocab_upper = max(2, min(200, self.text_vocab_size))

        text_prompt = torch.randint(
            0, text_vocab_upper, (batch_size, prompt_len), device=self.device
        )
        discrete_prompt = torch.randint(
            0,
            self.discrete_vocab_size,
            (batch_size, self.num_discrete_tokens, prompt_len),
            device=self.device,
        )
        continuous_prompt = torch.randn(
            batch_size, prompt_len, self.continuous_latent_size, device=self.device
        )
        text_target = torch.randint(
            0, text_vocab_upper, (batch_size, generated_len), device=self.device
        )

        outputs = self.model.generate_with_kv_cache(
            text_prompt=text_prompt,
            discrete_prompt=discrete_prompt,
            continuous_prompt=continuous_prompt,
            text_target=text_target,
            max_new_blocks=generated_len,
            discrete_eos_token_id=-1,
            do_sample=False,
            return_dict=True,
        )

        self.assertEqual(outputs.text_ids.shape, (batch_size, generated_len))
        self.assertEqual(
            outputs.discrete_ids.shape,
            (batch_size, self.num_discrete_tokens, generated_len),
        )
        self.assertEqual(
            outputs.continuous_latents.shape,
            (batch_size, generated_len, self.continuous_latent_size),
        )
        self.assertEqual(len(outputs.discrete_logits), generated_len)
        self.assertEqual(
            outputs.discrete_logits[0].shape,
            (batch_size, self.num_discrete_tokens, self.discrete_vocab_size),
        )


class TestPrismTTSGenerationAlignment(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cpu")
        self.num_discrete_tokens = 3
        self.discrete_vocab_size = 64
        self.continuous_latent_size = 8
        self.model = PrismTTS(
            llama_config=make_small_config(),
            num_discrete_tokens=self.num_discrete_tokens,
            discrete_vocab_size=self.discrete_vocab_size,
            continuous_latent_size=self.continuous_latent_size,
            flow_sample_steps=2,
        ).to(self.device)
        self.model.eval()

    def test_generate_and_generate_with_kv_cache_match_under_fixed_seed(self) -> None:
        batch_size = 2
        prompt_len = 3
        generated_len = 4
        text_vocab_upper = max(2, min(200, int(self.model.backbone.config.vocab_size)))

        text_prompt = torch.randint(
            0,
            text_vocab_upper,
            (batch_size, prompt_len),
            device=self.device,
        )
        discrete_prompt = torch.randint(
            0,
            self.discrete_vocab_size,
            (batch_size, self.num_discrete_tokens, prompt_len),
            device=self.device,
        )
        continuous_prompt = torch.randn(
            batch_size,
            prompt_len,
            self.continuous_latent_size,
            device=self.device,
        )
        text_target = torch.randint(
            0,
            text_vocab_upper,
            (batch_size, generated_len),
            device=self.device,
        )

        with torch.no_grad():
            torch.manual_seed(123)
            full_outputs = self.model.generate(
                text_prompt=text_prompt,
                discrete_prompt=discrete_prompt,
                continuous_prompt=continuous_prompt,
                text_target=text_target,
                max_new_blocks=generated_len,
                do_sample=False,
                return_dict=True,
            )
            torch.manual_seed(123)
            kv_outputs = self.model.generate_with_kv_cache(
                text_prompt=text_prompt,
                discrete_prompt=discrete_prompt,
                continuous_prompt=continuous_prompt,
                text_target=text_target,
                max_new_blocks=generated_len,
                do_sample=False,
                return_dict=True,
            )

        self.assertTrue(torch.equal(full_outputs.text_ids, kv_outputs.text_ids))
        self.assertTrue(torch.equal(full_outputs.discrete_ids, kv_outputs.discrete_ids))
        self.assertTrue(
            torch.allclose(
                full_outputs.continuous_latents,
                kv_outputs.continuous_latents,
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertEqual(len(full_outputs.discrete_logits), len(kv_outputs.discrete_logits))
        for full_logits, kv_logits in zip(full_outputs.discrete_logits, kv_outputs.discrete_logits):
            self.assertTrue(torch.allclose(full_logits, kv_logits, atol=1e-5, rtol=1e-5))

    def test_generate_first_step_logits_use_last_prompt_block(self) -> None:
        batch_size = 1
        prompt_len = 3
        generated_len = 2
        text_vocab_upper = max(2, min(200, int(self.model.backbone.config.vocab_size)))

        text_prompt = torch.randint(
            0,
            text_vocab_upper,
            (batch_size, prompt_len),
            device=self.device,
        )
        discrete_prompt = torch.randint(
            0,
            self.discrete_vocab_size,
            (batch_size, self.num_discrete_tokens, prompt_len),
            device=self.device,
        )
        continuous_prompt = torch.randn(
            batch_size,
            prompt_len,
            self.continuous_latent_size,
            device=self.device,
        )
        text_target = torch.randint(
            0,
            text_vocab_upper,
            (batch_size, generated_len),
            device=self.device,
        )

        normalized_discrete_prompt = self.model._normalize_discrete_tokens(
            discrete_prompt,
            "discrete_prompt",
        )
        with torch.no_grad():
            prompt_embeds, _ = self.model._build_inputs_embeds(
                text_tokens=text_prompt,
                discrete_tokens=normalized_discrete_prompt,
                continuous_latents=continuous_prompt,
            )
            prompt_stream_mask, prompt_block_mask = self.model._build_dual_attention_masks(
                total_blocks=prompt_len,
                batch_size=batch_size,
                device=prompt_embeds.device,
            )
            prompt_position_embeddings = self.model._build_two_level_rope_position_embeddings(
                inputs_embeds=prompt_embeds,
                total_blocks=prompt_len,
            )
            prompt_outputs = self.model.backbone(
                inputs_embeds=prompt_embeds,
                streamwise_attention_mask=prompt_stream_mask,
                blockwise_attention_mask=prompt_block_mask,
                position_embeddings=prompt_position_embeddings,
                return_dict=True,
            )
            last_prompt_block_hidden = prompt_outputs.last_hidden_state[
                :,
                -self.model.block_size :,
                :,
            ]
            expected_first_logits = self.model.discrete_lm_head(
                last_prompt_block_hidden[:, 1 : 1 + self.num_discrete_tokens, :]
            )

            generation = self.model.generate(
                text_prompt=text_prompt,
                discrete_prompt=discrete_prompt,
                continuous_prompt=continuous_prompt,
                text_target=text_target,
                max_new_blocks=generated_len,
                do_sample=False,
                return_dict=True,
            )

        self.assertGreaterEqual(len(generation.discrete_logits), 1)
        self.assertTrue(
            torch.allclose(
                generation.discrete_logits[0],
                expected_first_logits,
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_generate_with_kv_cache_respects_prompt_and_target_lengths(self) -> None:
        batch_size = 1
        prompt_len = 3
        padded_prompt_len = 6
        target_len = 2
        padded_target_len = 5
        text_vocab_upper = max(2, min(200, int(self.model.backbone.config.vocab_size)))
        text_pad_value = int(self.model.backbone.config.pad_token_id or 0)

        text_prompt_trim = torch.randint(
            1,
            text_vocab_upper,
            (batch_size, prompt_len),
            device=self.device,
        )
        discrete_prompt_trim = torch.randint(
            0,
            self.discrete_vocab_size,
            (batch_size, self.num_discrete_tokens, prompt_len),
            device=self.device,
        )
        continuous_prompt_trim = torch.randn(
            batch_size,
            prompt_len,
            self.continuous_latent_size,
            device=self.device,
        )
        text_target_trim = torch.randint(
            1,
            text_vocab_upper,
            (batch_size, target_len),
            device=self.device,
        )

        text_prompt_padded = torch.full(
            (batch_size, padded_prompt_len),
            text_pad_value,
            dtype=torch.long,
            device=self.device,
        )
        text_prompt_padded[:, :prompt_len] = text_prompt_trim

        discrete_prompt_padded = torch.full(
            (batch_size, self.num_discrete_tokens, padded_prompt_len),
            text_pad_value,
            dtype=torch.long,
            device=self.device,
        )
        discrete_prompt_padded[:, :, :prompt_len] = discrete_prompt_trim

        continuous_prompt_padded = torch.zeros(
            batch_size,
            padded_prompt_len,
            self.continuous_latent_size,
            device=self.device,
        )
        continuous_prompt_padded[:, :prompt_len, :] = continuous_prompt_trim

        text_target_padded = torch.full(
            (batch_size, padded_target_len),
            text_pad_value,
            dtype=torch.long,
            device=self.device,
        )
        text_target_padded[:, :target_len] = text_target_trim

        with torch.no_grad():
            torch.manual_seed(77)
            trimmed_outputs = self.model.generate_with_kv_cache(
                text_prompt=text_prompt_trim,
                discrete_prompt=discrete_prompt_trim,
                continuous_prompt=continuous_prompt_trim,
                text_target=text_target_trim,
                max_new_blocks=target_len,
                do_sample=False,
                return_dict=True,
            )
            torch.manual_seed(77)
            padded_outputs = self.model.generate_with_kv_cache(
                text_prompt=text_prompt_padded,
                discrete_prompt=discrete_prompt_padded,
                continuous_prompt=continuous_prompt_padded,
                text_target=text_target_padded,
                prompt_lengths=torch.tensor([prompt_len], device=self.device),
                target_lengths=torch.tensor([target_len], device=self.device),
                max_new_blocks=target_len,
                do_sample=False,
                return_dict=True,
            )

        self.assertTrue(torch.equal(trimmed_outputs.text_ids, padded_outputs.text_ids))
        self.assertTrue(torch.equal(trimmed_outputs.discrete_ids, padded_outputs.discrete_ids))
        self.assertTrue(
            torch.allclose(
                trimmed_outputs.continuous_latents,
                padded_outputs.continuous_latents,
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_special_discrete_blocks_produce_silent_continuous_latents(self) -> None:
        batch_size = 1
        prompt_len = 2
        generated_len = 1
        text_vocab_upper = max(2, min(200, int(self.model.backbone.config.vocab_size)))
        special_id = self.discrete_vocab_size - 1

        text_prompt = torch.randint(
            1,
            text_vocab_upper,
            (batch_size, prompt_len),
            device=self.device,
        )
        discrete_prompt = torch.randint(
            0,
            self.discrete_vocab_size,
            (batch_size, self.num_discrete_tokens, prompt_len),
            device=self.device,
        )
        continuous_prompt = torch.randn(
            batch_size,
            prompt_len,
            self.continuous_latent_size,
            device=self.device,
        )
        text_target = torch.randint(
            1,
            text_vocab_upper,
            (batch_size, generated_len),
            device=self.device,
        )

        original_sampler = self.model._sample_discrete_ids

        def _always_special(
            logits: torch.Tensor,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.95,
            do_sample: bool = True,
        ) -> torch.LongTensor:
            del temperature, top_k, top_p, do_sample
            return torch.full(
                logits.shape[:-1],
                fill_value=special_id,
                dtype=torch.long,
                device=logits.device,
            )

        self.model._sample_discrete_ids = _always_special
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    text_prompt=text_prompt,
                    discrete_prompt=discrete_prompt,
                    continuous_prompt=continuous_prompt,
                    text_target=text_target,
                    max_new_blocks=generated_len,
                    do_sample=False,
                    discrete_eos_token_id=special_id,
                    force_silent_special_tokens=True,
                    return_dict=True,
                )
        finally:
            self.model._sample_discrete_ids = original_sampler

        self.assertTrue(
            torch.all(
                outputs.continuous_latents
                == torch.zeros_like(outputs.continuous_latents)
            )
        )

    def test_generation_teacher_forced_delay_prefix_is_respected(self) -> None:
        batch_size = 1
        prompt_len = 2
        generated_len = 4
        delay_prefix_len = 2
        text_vocab_upper = max(2, min(200, int(self.model.backbone.config.vocab_size)))
        delay_token_id = self.discrete_vocab_size - 2

        text_prompt = torch.randint(
            1,
            text_vocab_upper,
            (batch_size, prompt_len),
            device=self.device,
        )
        discrete_prompt = torch.randint(
            0,
            self.discrete_vocab_size,
            (batch_size, self.num_discrete_tokens, prompt_len),
            device=self.device,
        )
        continuous_prompt = torch.randn(
            batch_size,
            prompt_len,
            self.continuous_latent_size,
            device=self.device,
        )
        text_target = torch.randint(
            1,
            text_vocab_upper,
            (batch_size, generated_len),
            device=self.device,
        )

        teacher_discrete_prefix = torch.full(
            (batch_size, self.num_discrete_tokens, delay_prefix_len),
            fill_value=delay_token_id,
            dtype=torch.long,
            device=self.device,
        )
        teacher_continuous_prefix = torch.randn(
            batch_size,
            delay_prefix_len,
            self.continuous_latent_size,
            device=self.device,
        )

        for method_name in ("generate", "generate_with_kv_cache"):
            generation_method = getattr(self.model, method_name)
            with torch.no_grad():
                outputs = generation_method(
                    text_prompt=text_prompt,
                    discrete_prompt=discrete_prompt,
                    continuous_prompt=continuous_prompt,
                    text_target=text_target,
                    max_new_blocks=generated_len,
                    do_sample=False,
                    discrete_eos_token_id=-1,
                    teacher_forced_discrete_prefix=teacher_discrete_prefix,
                    teacher_forced_continuous_prefix=teacher_continuous_prefix,
                    return_dict=True,
                )

            self.assertTrue(
                torch.equal(
                    outputs.discrete_ids[:, :, :delay_prefix_len],
                    teacher_discrete_prefix,
                )
            )
            self.assertTrue(
                torch.allclose(
                    outputs.continuous_latents[:, :delay_prefix_len, :],
                    teacher_continuous_prefix,
                    atol=1e-6,
                    rtol=1e-6,
                )
            )


if __name__ == "__main__":
    unittest.main()
