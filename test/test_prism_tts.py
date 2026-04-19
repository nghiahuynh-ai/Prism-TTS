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

import models.prism_tts as prism_tts_module
from models.prism_tts import PrismTTS
from dataset.dataset import BatchCollate
from utils.model_utils import resolve_generation_discrete_eos_token_id


MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"
MIN_REASONABLE_PARAM_COUNT = 90_000_000


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
            flow_sample_steps=int(prism_cfg.get("flow_sample_steps", 64)),
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

        collate = BatchCollate(
            discrete_token_count=max(1, int(self.discrete_vocab_size) - 3)
        )
        samples = []
        for sample_idx in range(batch_size):
            samples.append(
                {
                    "text_prompt": text_prompt[sample_idx].detach().cpu(),
                    "discrete_prompt": discrete_prompt[sample_idx]
                    .transpose(0, 1)
                    .contiguous()
                    .detach()
                    .cpu(),
                    "continuous_prompt": continuous_prompt[sample_idx].detach().cpu(),
                    "text_target": text_target[sample_idx].detach().cpu(),
                    "discrete_target": discrete_target[sample_idx]
                    .transpose(0, 1)
                    .contiguous()
                    .detach()
                    .cpu(),
                    "continuous_target": continuous_target[sample_idx].detach().cpu(),
                }
            )
        flat_batch = collate(samples)

        outputs = self.model(
            flat_token_ids=flat_batch["flat_token_ids"].to(self.device),
            flat_continuous_values=flat_batch["flat_continuous_values"].to(self.device),
            flat_token_type_ids=flat_batch["flat_token_type_ids"].to(self.device),
            flat_speech_stream_ids=flat_batch["flat_speech_stream_ids"].to(self.device),
            flat_target_block_ids=flat_batch["flat_target_block_ids"].to(self.device),
            flat_target_block_counts=flat_batch["flat_target_block_counts"].to(self.device),
            attention_mask=flat_batch["attention_mask"].to(self.device),
            return_dict=True,
        )

        self.assertIsNotNone(outputs.loss)
        self.assertIsNotNone(outputs.discrete_loss)
        self.assertIsNotNone(outputs.continuous_loss)
        self.assertIsNotNone(outputs.flow_loss)
        self.assertEqual(
            tuple(outputs.keys()),
            ("loss", "discrete_loss", "continuous_loss", "flow_loss"),
        )

    def test_forward_samples_mask_ratio_from_zero_to_one_when_unspecified(self):
        batch_size = 2
        prompt_len = 2
        target_len = 3
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
            0, text_vocab_upper, (batch_size, target_len), device=self.device
        )
        discrete_target = torch.randint(
            0,
            self.discrete_vocab_size,
            (batch_size, self.num_discrete_tokens, target_len),
            device=self.device,
        )
        continuous_target = torch.randn(
            batch_size, target_len, self.continuous_latent_size, device=self.device
        )

        collate = BatchCollate(
            discrete_token_count=max(1, int(self.discrete_vocab_size) - 3)
        )
        samples = []
        for sample_idx in range(batch_size):
            samples.append(
                {
                    "text_prompt": text_prompt[sample_idx].detach().cpu(),
                    "discrete_prompt": discrete_prompt[sample_idx]
                    .transpose(0, 1)
                    .contiguous()
                    .detach()
                    .cpu(),
                    "continuous_prompt": continuous_prompt[sample_idx].detach().cpu(),
                    "text_target": text_target[sample_idx].detach().cpu(),
                    "discrete_target": discrete_target[sample_idx]
                    .transpose(0, 1)
                    .contiguous()
                    .detach()
                    .cpu(),
                    "continuous_target": continuous_target[sample_idx].detach().cpu(),
                }
            )
        flat_batch = collate(samples)

        captured_ratios: list[float] = []
        original_sampler = prism_tts_module.MU.sample_masked_target_blocks

        def _capture_ratio(
            target_block_counts: torch.LongTensor,
            mask_ratio: float,
            masked_target_blocks: torch.BoolTensor | None,
        ) -> torch.BoolTensor:
            captured_ratios.append(float(mask_ratio))
            return original_sampler(
                target_block_counts=target_block_counts,
                mask_ratio=mask_ratio,
                masked_target_blocks=masked_target_blocks,
            )

        prism_tts_module.MU.sample_masked_target_blocks = _capture_ratio
        try:
            self.model(
                flat_token_ids=flat_batch["flat_token_ids"].to(self.device),
                flat_continuous_values=flat_batch["flat_continuous_values"].to(self.device),
                flat_token_type_ids=flat_batch["flat_token_type_ids"].to(self.device),
                flat_speech_stream_ids=flat_batch["flat_speech_stream_ids"].to(self.device),
                flat_target_block_ids=flat_batch["flat_target_block_ids"].to(self.device),
                flat_target_block_counts=flat_batch["flat_target_block_counts"].to(self.device),
                attention_mask=flat_batch["attention_mask"].to(self.device),
                return_dict=True,
            )
            self.model(
                flat_token_ids=flat_batch["flat_token_ids"].to(self.device),
                flat_continuous_values=flat_batch["flat_continuous_values"].to(self.device),
                flat_token_type_ids=flat_batch["flat_token_type_ids"].to(self.device),
                flat_speech_stream_ids=flat_batch["flat_speech_stream_ids"].to(self.device),
                flat_target_block_ids=flat_batch["flat_target_block_ids"].to(self.device),
                flat_target_block_counts=flat_batch["flat_target_block_counts"].to(self.device),
                attention_mask=flat_batch["attention_mask"].to(self.device),
                return_dict=True,
            )
        finally:
            prism_tts_module.MU.sample_masked_target_blocks = original_sampler

        self.assertEqual(len(captured_ratios), 2)
        for ratio in captured_ratios:
            self.assertGreaterEqual(ratio, 0.0)
            self.assertLess(ratio, 1.0)

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
        self.assertEqual(len(outputs.discrete_logits), max(0, generated_len - 1))
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

    def test_generate_is_deterministic_under_fixed_seed(self) -> None:
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
            second_outputs = self.model.generate(
                text_prompt=text_prompt,
                discrete_prompt=discrete_prompt,
                continuous_prompt=continuous_prompt,
                text_target=text_target,
                max_new_blocks=generated_len,
                do_sample=False,
                return_dict=True,
            )

        self.assertTrue(torch.equal(full_outputs.text_ids, second_outputs.text_ids))
        self.assertTrue(torch.equal(full_outputs.discrete_ids, second_outputs.discrete_ids))
        self.assertTrue(
            torch.allclose(
                full_outputs.continuous_latents,
                second_outputs.continuous_latents,
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertEqual(len(full_outputs.discrete_logits), len(second_outputs.discrete_logits))
        for full_logits, second_logits in zip(
            full_outputs.discrete_logits,
            second_outputs.discrete_logits,
        ):
            self.assertTrue(torch.allclose(full_logits, second_logits, atol=1e-5, rtol=1e-5))

    def test_generate_uses_explicit_terminal_eos_block(self) -> None:
        batch_size = 1
        prompt_len = 2
        max_new_blocks = 6
        text_vocab_upper = max(2, min(200, int(self.model.backbone.config.vocab_size)))
        eos_id = int(
            resolve_generation_discrete_eos_token_id(
                None,
                backbone_eos_token_id=self.model.backbone.config.eos_token_id,
                discrete_vocab_size=self.model.discrete_vocab_size,
            )
        )
        content_id = 0 if eos_id != 0 else 1

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
            (batch_size, prompt_len),
            device=self.device,
        )

        original_sampler = self.model._sample_discrete_ids

        def _always_non_eos(
            logits: torch.Tensor,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.95,
            do_sample: bool = True,
        ) -> torch.LongTensor:
            del temperature, top_k, top_p, do_sample
            return torch.full(
                logits.shape[:-1],
                fill_value=content_id,
                dtype=torch.long,
                device=logits.device,
            )

        self.model._sample_discrete_ids = _always_non_eos
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    text_prompt=text_prompt,
                    discrete_prompt=discrete_prompt,
                    continuous_prompt=continuous_prompt,
                    text_target=text_target,
                    max_new_blocks=max_new_blocks,
                    discrete_eos_token_id=eos_id,
                    do_sample=False,
                    return_dict=True,
                )
        finally:
            self.model._sample_discrete_ids = original_sampler

        self.assertEqual(outputs.discrete_ids.shape[-1], max_new_blocks)
        self.assertEqual(outputs.continuous_latents.shape[1], max_new_blocks)
        self.assertEqual(len(outputs.discrete_logits), max(0, max_new_blocks - 1))
        self.assertTrue(
            torch.eq(outputs.discrete_ids[0, :, max_new_blocks - 1], eos_id).all()
        )
        if max_new_blocks > 1:
            self.assertTrue(
                torch.eq(outputs.discrete_ids[0, :, : max_new_blocks - 1], content_id).all()
            )
        self.assertTrue(
            torch.eq(
                outputs.continuous_latents[0, max_new_blocks - 1, :],
                0.0,
            ).all()
        )

    def test_generate_parallel_returns_expected_shapes(self) -> None:
        batch_size = 1
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
            outputs = self.model.generate(
                text_prompt=text_prompt,
                discrete_prompt=discrete_prompt,
                continuous_prompt=continuous_prompt,
                text_target=text_target,
                speech_target_lengths=torch.tensor([generated_len], device=self.device),
                max_new_blocks=generated_len,
                discrete_eos_token_id=-1,
                do_sample=False,
                generation_method="parallel",
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
        self.assertGreaterEqual(len(outputs.discrete_logits), 1)

    def test_generate_parallel_stable_returns_expected_shapes(self) -> None:
        batch_size = 1
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
            outputs = self.model.generate(
                text_prompt=text_prompt,
                discrete_prompt=discrete_prompt,
                continuous_prompt=continuous_prompt,
                text_target=text_target,
                speech_target_lengths=torch.tensor([generated_len], device=self.device),
                max_new_blocks=generated_len,
                discrete_eos_token_id=-1,
                do_sample=False,
                generation_method="parallel_stable",
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
        self.assertGreaterEqual(len(outputs.discrete_logits), 1)

    def test_generate_parallel_estimates_target_lengths_from_prompt_ratio(self) -> None:
        batch_size = 1
        prompt_text_len = 4
        prompt_speech_len = 8
        target_text_len = 3
        expected_generated_len = 6
        text_vocab_upper = max(2, min(200, int(self.model.backbone.config.vocab_size)))

        text_prompt = torch.randint(
            0,
            text_vocab_upper,
            (batch_size, prompt_text_len),
            device=self.device,
        )
        discrete_prompt = torch.randint(
            0,
            self.discrete_vocab_size,
            (batch_size, self.num_discrete_tokens, prompt_speech_len),
            device=self.device,
        )
        continuous_prompt = torch.randn(
            batch_size,
            prompt_speech_len,
            self.continuous_latent_size,
            device=self.device,
        )
        text_target = torch.randint(
            0,
            text_vocab_upper,
            (batch_size, target_text_len),
            device=self.device,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                text_prompt=text_prompt,
                discrete_prompt=discrete_prompt,
                continuous_prompt=continuous_prompt,
                text_target=text_target,
                max_new_blocks=None,
                discrete_eos_token_id=-1,
                do_sample=False,
                generation_method="parallel",
                return_dict=True,
            )

        self.assertEqual(outputs.discrete_ids.shape[-1], expected_generated_len)
        self.assertEqual(outputs.continuous_latents.shape[1], expected_generated_len)

    def test_generate_parallel_respects_parallel_num_steps(self) -> None:
        batch_size = 1
        prompt_len = 3
        generated_len = 16
        parallel_steps = 2
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
            outputs = self.model.generate(
                text_prompt=text_prompt,
                discrete_prompt=discrete_prompt,
                continuous_prompt=continuous_prompt,
                text_target=text_target,
                speech_target_lengths=torch.tensor([generated_len], device=self.device),
                max_new_blocks=generated_len,
                do_sample=False,
                generation_method="parallel",
                parallel_num_steps=parallel_steps,
                return_dict=True,
            )

        self.assertGreaterEqual(len(outputs.discrete_logits), 1)
        self.assertLessEqual(len(outputs.discrete_logits), parallel_steps)

    def test_generate_parallel_stable_respects_parallel_num_steps(self) -> None:
        batch_size = 1
        prompt_len = 3
        generated_len = 16
        parallel_steps = 2
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
            outputs = self.model.generate(
                text_prompt=text_prompt,
                discrete_prompt=discrete_prompt,
                continuous_prompt=continuous_prompt,
                text_target=text_target,
                speech_target_lengths=torch.tensor([generated_len], device=self.device),
                max_new_blocks=generated_len,
                do_sample=False,
                generation_method="parallel_stable",
                parallel_num_steps=parallel_steps,
                return_dict=True,
            )

        self.assertGreaterEqual(len(outputs.discrete_logits), 1)
        self.assertLessEqual(len(outputs.discrete_logits), parallel_steps)

    def test_generate_rejects_unknown_generation_method(self) -> None:
        batch_size = 1
        prompt_len = 2
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
            (batch_size, prompt_len),
            device=self.device,
        )

        with self.assertRaisesRegex(ValueError, "generation_method must be one of"):
            self.model.generate(
                text_prompt=text_prompt,
                discrete_prompt=discrete_prompt,
                continuous_prompt=continuous_prompt,
                text_target=text_target,
                do_sample=False,
                generation_method="nonexistent",
                return_dict=True,
            )

    def test_generate_rejects_invalid_parallel_num_steps(self) -> None:
        batch_size = 1
        prompt_len = 2
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
            (batch_size, prompt_len),
            device=self.device,
        )

        with self.assertRaisesRegex(ValueError, "parallel_num_steps must be >= 1"):
            self.model.generate(
                text_prompt=text_prompt,
                discrete_prompt=discrete_prompt,
                continuous_prompt=continuous_prompt,
                text_target=text_target,
                do_sample=False,
                generation_method="parallel",
                parallel_num_steps=0,
                return_dict=True,
            )

    def test_discrete_loss_applies_special_token_weight(self) -> None:
        model = PrismTTS(
            llama_config=make_small_config(),
            num_discrete_tokens=1,
            discrete_vocab_size=16,
            continuous_latent_size=8,
            discrete_regular_token_loss_weight=1.0,
            discrete_special_token_loss_weight=0.05,
            flow_sample_steps=2,
        ).to(self.device)
        model.eval()

        self.assertGreater(len(model.training_special_discrete_token_ids), 0)
        special_id = int(model.training_special_discrete_token_ids[0])
        regular_id = 1 if special_id != 1 else 2

        hidden_states = torch.zeros(1, 4, model.hidden_size, device=self.device)
        token_ids = torch.tensor(
            [[regular_id, special_id, regular_id, regular_id]],
            dtype=torch.long,
            device=self.device,
        )
        masked_positions = torch.tensor(
            [[True, True, True, False]],
            dtype=torch.bool,
            device=self.device,
        )

        fixed_logits = torch.zeros(
            3,
            model.discrete_vocab_size,
            dtype=hidden_states.dtype,
            device=self.device,
        )
        fixed_logits[0, regular_id] = 4.0
        fixed_logits[1, regular_id] = 4.0
        fixed_logits[2, regular_id] = 1.0

        class _FixedHead(torch.nn.Module):
            def __init__(self, logits: torch.Tensor) -> None:
                super().__init__()
                self.register_buffer("logits", logits)

            def forward(self, hidden: torch.Tensor) -> torch.Tensor:
                if hidden.shape[0] != self.logits.shape[0]:
                    raise AssertionError("Unexpected hidden batch size in fixed head.")
                return self.logits

        model.discrete_lm_head = _FixedHead(fixed_logits)
        loss, logits_out = model._compute_discrete_loss(
            hidden_states=hidden_states,
            token_ids=token_ids,
            masked_discrete_positions=masked_positions,
        )

        selected_targets = token_ids[masked_positions]
        per_token_loss = torch.nn.functional.cross_entropy(
            fixed_logits,
            selected_targets,
            reduction="none",
        )
        expected_weights = torch.tensor(
            [1.0, 0.05, 1.0],
            dtype=per_token_loss.dtype,
            device=per_token_loss.device,
        )
        expected_loss = (per_token_loss * expected_weights).sum() / expected_weights.sum()

        self.assertTrue(torch.allclose(logits_out, fixed_logits, atol=0.0, rtol=0.0))
        self.assertTrue(torch.allclose(loss, expected_loss, atol=1e-7, rtol=1e-6))

    def test_generate_e2e_accepts_string_and_raw_speech_prompt(self) -> None:
        eos_id = int(
            resolve_generation_discrete_eos_token_id(
                None,
                backbone_eos_token_id=self.model.backbone.config.eos_token_id,
                discrete_vocab_size=self.model.discrete_vocab_size,
            )
        )

        def _tokenizer(text: str) -> list[int]:
            return [5 + (ord(ch) % 13) for ch in text]

        def _speech_encoder(raw_prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            frame_count = int(raw_prompt.shape[0])
            discrete = (
                torch.arange(
                    frame_count * self.num_discrete_tokens,
                    device=raw_prompt.device,
                    dtype=torch.long,
                )
                .view(frame_count, self.num_discrete_tokens)
                .remainder(self.discrete_vocab_size)
            )
            continuous = torch.randn(
                frame_count,
                self.continuous_latent_size,
                device=raw_prompt.device,
            )
            return discrete, continuous

        raw_prompt_audio = torch.tensor([0.1, -0.2, 0.3, -0.4], device=self.device)
        with torch.no_grad():
            outputs = self.model.generate_e2e(
                raw_text_prompt="ab",
                raw_speech_prompt=raw_prompt_audio,
                raw_text_target="cd",
                text_tokenizer=_tokenizer,
                speech_encoder=_speech_encoder,
                max_new_blocks=3,
                discrete_eos_token_id=eos_id,
                do_sample=False,
                return_dict=True,
            )

        self.assertEqual(outputs.text_ids.shape[0], 1)
        self.assertEqual(outputs.discrete_ids.shape[0], 1)
        self.assertEqual(outputs.continuous_latents.shape[0], 1)
        self.assertLessEqual(outputs.discrete_ids.shape[-1], 3)
        self.assertEqual(
            len(outputs.discrete_logits),
            max(0, int(outputs.discrete_ids.shape[-1]) - 1),
        )

    def test_generate_e2e_supports_batched_inputs(self) -> None:
        eos_id = int(
            resolve_generation_discrete_eos_token_id(
                None,
                backbone_eos_token_id=self.model.backbone.config.eos_token_id,
                discrete_vocab_size=self.model.discrete_vocab_size,
            )
        )

        def _tokenizer(text: str) -> list[int]:
            return [7 + (ord(ch) % 17) for ch in text]

        def _speech_encoder(raw_prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            frame_count = int(raw_prompt.shape[0])
            discrete = (
                torch.arange(
                    frame_count * self.num_discrete_tokens,
                    dtype=torch.long,
                    device=raw_prompt.device,
                )
                .view(frame_count, self.num_discrete_tokens)
                .remainder(self.discrete_vocab_size)
            )
            continuous = torch.randn(
                frame_count,
                self.continuous_latent_size,
                device=raw_prompt.device,
            )
            return discrete, continuous

        raw_prompts = [
            torch.tensor([0.2, 0.1, -0.1], device=self.device),
            torch.tensor([0.3, -0.3], device=self.device),
        ]
        with torch.no_grad():
            outputs = self.model.generate_e2e(
                raw_text_prompt=["hello", "yo"],
                raw_speech_prompt=raw_prompts,
                raw_text_target=["world", "ha"],
                text_tokenizer=_tokenizer,
                speech_encoder=_speech_encoder,
                max_new_blocks=2,
                discrete_eos_token_id=eos_id,
                do_sample=False,
                return_dict=True,
            )

        self.assertEqual(outputs.text_ids.shape[0], 2)
        self.assertEqual(outputs.discrete_ids.shape[0], 2)
        self.assertEqual(outputs.continuous_latents.shape[0], 2)
        self.assertLessEqual(outputs.discrete_ids.shape[-1], 2)

    def test_generate_respects_prompt_and_target_lengths(self) -> None:
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
            trimmed_outputs = self.model.generate(
                text_prompt=text_prompt_trim,
                discrete_prompt=discrete_prompt_trim,
                continuous_prompt=continuous_prompt_trim,
                text_target=text_target_trim,
                max_new_blocks=target_len,
                do_sample=False,
                return_dict=True,
            )
            torch.manual_seed(77)
            padded_outputs = self.model.generate(
                text_prompt=text_prompt_padded,
                discrete_prompt=discrete_prompt_padded,
                continuous_prompt=continuous_prompt_padded,
                text_target=text_target_padded,
                text_prompt_lengths=torch.tensor([prompt_len], device=self.device),
                speech_prompt_lengths=torch.tensor([prompt_len], device=self.device),
                text_target_lengths=torch.tensor([target_len], device=self.device),
                speech_target_lengths=torch.tensor([target_len], device=self.device),
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

if __name__ == "__main__":
    unittest.main()
