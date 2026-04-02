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
        self.assertEqual(tuple(outputs.keys()), ("loss", "discrete_loss", "flow_loss"))

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


if __name__ == "__main__":
    unittest.main()
