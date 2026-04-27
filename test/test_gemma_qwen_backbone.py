from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.gemma_backbone import GemmaBackbone
from models.qwen_backbone import QwenBackbone

try:
    from transformers import Gemma3TextConfig, Qwen3Config
except Exception:  # pragma: no cover - optional dependency by environment
    Gemma3TextConfig = None  # type: ignore[assignment]
    Qwen3Config = None  # type: ignore[assignment]


def _print_trainable_and_non_trainable_params(model: torch.nn.Module, *, label: str) -> None:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    non_trainable = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    print(f"[{label}] Trainable Parameters: {trainable:,}")
    print(f"[{label}] Non-trainable Parameters: {non_trainable:,}")


@unittest.skipIf(Gemma3TextConfig is None, "Gemma3 config support is unavailable.")
class TestGemmaBackboneMaskedAttention(unittest.TestCase):
    def test_forces_non_causal_full_attention(self):
        config = Gemma3TextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
            max_position_embeddings=256,
            pad_token_id=0,
            eos_token_id=1,
            use_cache=False,
        )
        config.head_dim = config.hidden_size // config.num_attention_heads
        config.query_pre_attn_scalar = config.head_dim
        config.layer_types = ["sliding_attention", "full_attention"]
        config.use_bidirectional_attention = False
        config._attn_implementation = "eager"

        backbone = GemmaBackbone(config)
        _print_trainable_and_non_trainable_params(backbone, label="GemmaBackbone")

        self.assertEqual(
            list(backbone.layer_types),
            ["full_attention"] * int(config.num_hidden_layers),
        )
        self.assertTrue(bool(backbone.config.use_bidirectional_attention))

        for layer in backbone.layers:
            self.assertEqual(layer.attention_type, "full_attention")
            self.assertFalse(bool(layer.self_attn.is_causal))
            self.assertIsNone(layer.self_attn.sliding_window)

        input_ids = torch.randint(0, int(config.vocab_size), (2, 6), dtype=torch.long)
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]],
            dtype=torch.bool,
        )
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        self.assertEqual(tuple(outputs.last_hidden_state.shape), (2, 6, int(config.hidden_size)))


@unittest.skipIf(Qwen3Config is None, "Qwen3 config support is unavailable.")
class TestQwenBackboneMaskedAttention(unittest.TestCase):
    def test_forces_non_causal_full_attention(self):
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            pad_token_id=0,
            eos_token_id=1,
            use_cache=False,
        )
        config.head_dim = config.hidden_size // config.num_attention_heads
        config.layer_types = ["sliding_attention", "full_attention"]
        config.sliding_window = 64
        config.use_sliding_window = True
        config._attn_implementation = "eager"

        backbone = QwenBackbone(config)
        _print_trainable_and_non_trainable_params(backbone, label="QwenBackbone")

        self.assertEqual(
            list(backbone.layer_types),
            ["full_attention"] * int(config.num_hidden_layers),
        )
        self.assertFalse(bool(backbone.has_sliding_layers))

        for layer in backbone.layers:
            self.assertEqual(layer.attention_type, "full_attention")
            self.assertFalse(bool(layer.self_attn.is_causal))
            self.assertIsNone(layer.self_attn.sliding_window)

        input_ids = torch.randint(0, int(config.vocab_size), (2, 6), dtype=torch.long)
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]],
            dtype=torch.bool,
        )
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        self.assertEqual(tuple(outputs.last_hidden_state.shape), (2, 6, int(config.hidden_size)))
