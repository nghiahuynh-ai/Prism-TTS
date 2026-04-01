from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "llama_backbone.py"


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


def make_4d_masks(
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device))

    streamwise_mask = causal.clone()
    streamwise_mask[0::2, 1::2] = 0

    blockwise_mask = torch.zeros_like(causal)
    for idx in range(seq_len):
        blockwise_mask[idx, max(0, idx - 1) : idx + 1] = 1

    streamwise_mask = (
        streamwise_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    )
    blockwise_mask = (
        blockwise_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    )
    return streamwise_mask, blockwise_mask


class TestLlamaBackbone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.LlamaBackbone = load_llama_backbone()

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cpu")
        self.model = self.LlamaBackbone(make_config()).to(self.device).eval()
        print('='*100)
        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))

    def test_forward_with_dual_attention_masks(self):
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
        streamwise_mask, blockwise_mask = make_4d_masks(
            batch_size=input_ids.shape[0],
            seq_len=input_ids.shape[1],
            device=self.device,
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                streamwise_attention_mask=streamwise_mask,
                blockwise_attention_mask=blockwise_mask,
                output_hidden_states=True,
                output_attentions=True,
            )

        self.assertEqual(outputs.last_hidden_state.shape, (2, 6, 64))
        self.assertEqual(len(outputs.hidden_states), 3)
        self.assertEqual(len(outputs.attentions), 2)
        self.assertEqual(len(outputs.attentions[0]), 2)
        self.assertEqual(outputs.attentions[0][0].shape[-2:], (6, 6))
        self.assertEqual(outputs.attentions[0][1].shape[-2:], (6, 6))

    def test_streamwise_runs_before_blockwise_and_receives_matching_masks(self):
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=self.device)
        streamwise_mask, blockwise_mask = make_4d_masks(
            batch_size=1,
            seq_len=input_ids.shape[1],
            device=self.device,
        )

        first_layer = self.model.layers[0]
        streamwise_forward = first_layer.streamwise_attn.forward
        blockwise_forward = first_layer.blockwise_attn.forward
        call_log = []

        def wrap_streamwise(*args, **kwargs):
            call_log.append(("streamwise", kwargs["attention_mask"].detach().clone()))
            return streamwise_forward(*args, **kwargs)

        def wrap_blockwise(*args, **kwargs):
            call_log.append(("blockwise", kwargs["attention_mask"].detach().clone()))
            return blockwise_forward(*args, **kwargs)

        first_layer.streamwise_attn.forward = wrap_streamwise
        first_layer.blockwise_attn.forward = wrap_blockwise

        try:
            with torch.no_grad():
                self.model(
                    input_ids=input_ids,
                    streamwise_attention_mask=streamwise_mask,
                    blockwise_attention_mask=blockwise_mask,
                )
        finally:
            first_layer.streamwise_attn.forward = streamwise_forward
            first_layer.blockwise_attn.forward = blockwise_forward

        self.assertGreaterEqual(len(call_log), 2)
        self.assertEqual(call_log[0][0], "streamwise")
        self.assertEqual(call_log[1][0], "blockwise")
        self.assertTrue(torch.equal(call_log[0][1], streamwise_mask))
        self.assertTrue(torch.equal(call_log[1][1], blockwise_mask))

    def test_autoregressive_cache_matches_full_decode_last_token(self):
        prefill_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=self.device)
        next_token = torch.tensor([[6]], dtype=torch.long, device=self.device)
        full_ids = torch.cat([prefill_ids, next_token], dim=1)

        with torch.no_grad():
            full_outputs = self.model(input_ids=full_ids, use_cache=False)
            prefill_outputs = self.model(input_ids=prefill_ids, use_cache=True)
            step_outputs = self.model(
                input_ids=next_token,
                past_key_values=prefill_outputs.past_key_values,
                use_cache=True,
            )

        expected_last = full_outputs.last_hidden_state[:, -1:, :]
        self.assertTrue(
            torch.allclose(step_outputs.last_hidden_state, expected_last, atol=1e-5)
        )
        self.assertEqual(step_outputs.past_key_values.get_seq_length(), full_ids.shape[1])


if __name__ == "__main__":
    unittest.main()
