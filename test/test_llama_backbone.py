from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "llama_backbone.py"


def _print_trainable_and_non_trainable_params(model: torch.nn.Module, *, label: str) -> None:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    non_trainable = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    print(f"[{label}] Trainable Parameters: {trainable:,}")
    print(f"[{label}] Non-trainable Parameters: {non_trainable:,}")


def load_llama_backbone():
    spec = importlib.util.spec_from_file_location("prism_llama_backbone", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LlamaBackbone


def make_config() -> LlamaConfig:
    config = LlamaConfig(
        vocab_size=2048,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    config._attn_implementation = "eager"
    return config


class TestLlamaBackbone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.LlamaBackbone = load_llama_backbone()
        cls._printed_param_summary = False

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cpu")
        self.model = self.LlamaBackbone(make_config()).to(self.device).eval()
        if not type(self)._printed_param_summary:
            _print_trainable_and_non_trainable_params(self.model, label="LlamaBackbone")
            type(self)._printed_param_summary = True

    def test_forward_with_attention_mask(self):
        input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5, 0],
                [6, 7, 8, 9, 0, 0],
            ],
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0],
            ],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
            )

        self.assertEqual(outputs.last_hidden_state.shape, (2, 6, 64))
        self.assertEqual(len(outputs.hidden_states), 3)
        self.assertEqual(len(outputs.attentions), 2)
        self.assertEqual(outputs.attentions[0].shape[-2:], (6, 6))

    def test_attention_mask_is_forwarded_to_attention_layer(self):
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=self.device)
        custom_additive_mask = torch.zeros((1, 1, 4, 4), dtype=torch.float32, device=self.device)
        custom_additive_mask[:, :, :, -1] = torch.finfo(torch.float32).min

        first_layer = self.model.layers[0]
        original_forward = first_layer.self_attn.forward
        call_log: list[torch.Tensor] = []

        def wrapped_forward(*args, **kwargs):
            mask = kwargs.get("attention_mask")
            if mask is not None:
                call_log.append(mask.detach().clone())
            return original_forward(*args, **kwargs)

        first_layer.self_attn.forward = wrapped_forward
        try:
            with torch.no_grad():
                self.model(
                    input_ids=input_ids,
                    attention_mask=custom_additive_mask,
                )
        finally:
            first_layer.self_attn.forward = original_forward

        self.assertGreaterEqual(len(call_log), 1)
        self.assertTrue(torch.equal(call_log[0], custom_additive_mask))

    def test_cache_path_runs_and_returns_expected_shapes(self):
        prefill_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=self.device)
        next_token = torch.tensor([[6]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            prefill_outputs = self.model(input_ids=prefill_ids, use_cache=True)
            step_outputs = self.model(
                input_ids=next_token,
                past_key_values=prefill_outputs.past_key_values,
                use_cache=True,
            )

        self.assertEqual(step_outputs.last_hidden_state.shape, (1, 1, 64))
        self.assertEqual(step_outputs.past_key_values.get_seq_length(), 6)


if __name__ == "__main__":
    unittest.main()
