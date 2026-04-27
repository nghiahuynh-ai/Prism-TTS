from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.backbone_utils import (
    build_backbone,
    normalize_backbone_name,
    resolve_backbone_spec,
    should_use_checkpoint_text_tokenizer,
)


def test_llada_aliases_are_supported() -> None:
    assert normalize_backbone_name("llada") == "llada"
    assert should_use_checkpoint_text_tokenizer("llada")


def test_resolve_backbone_spec_reads_llada_lora_config() -> None:
    model_cfg = {
        "backbone": {
            "name": "llada",
            "config_key": "llada_config",
            "hf_checkpoint": "GSAI-ML/LLaDA-8B-Base",
            "lora": {
                "enabled": True,
                "r": 8,
                "alpha": 16,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
            },
        },
        "llada_config": {
            "pad_token_id": 2050,
            "eos_token_id": 2049,
            "use_cache": False,
        },
    }

    spec = resolve_backbone_spec(model_cfg)
    assert spec.name == "llada"
    assert spec.hf_checkpoint == "GSAI-ML/LLaDA-8B-Base"
    assert spec.lora_config["enabled"] is True
    assert spec.lora_config["r"] == 8
    assert spec.config == model_cfg["llada_config"]


def test_non_llada_backbone_rejects_enabled_lora() -> None:
    with pytest.raises(ValueError, match="only supported with model.backbone.name='llada'"):
        build_backbone(
            backbone_name="llama",
            backbone_config={
                "vocab_size": 128,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "pad_token_id": 0,
                "eos_token_id": 1,
                "use_cache": False,
            },
            backbone_hf_checkpoint=None,
            backbone_hf_strict=True,
            backbone_hf_kwargs=None,
            backbone_lora_config={"enabled": True, "r": 8},
        )


def test_non_llada_backbone_rejects_target_modules_in_lora_config() -> None:
    with pytest.raises(ValueError, match="target_modules is only supported"):
        build_backbone(
            backbone_name="llama",
            backbone_config={
                "vocab_size": 128,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "pad_token_id": 0,
                "eos_token_id": 1,
                "use_cache": False,
            },
            backbone_hf_checkpoint=None,
            backbone_hf_strict=True,
            backbone_hf_kwargs=None,
            backbone_lora_config={"enabled": False, "target_modules": ["q_proj"]},
        )


def test_resolve_backbone_spec_rejects_target_modules_for_non_llada() -> None:
    model_cfg = {
        "backbone": {
            "name": "llama",
            "lora": {
                "target_modules": ["q_proj"],
            },
        },
        "llama_config": {
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "use_cache": False,
        },
    }
    with pytest.raises(ValueError, match="target_modules is only supported"):
        resolve_backbone_spec(model_cfg)


def test_resolve_backbone_spec_rejects_disabled_lora_for_llada() -> None:
    model_cfg = {
        "backbone": {
            "name": "llada",
            "hf_checkpoint": "GSAI-ML/LLaDA-8B-Base",
            "lora": {
                "enabled": False,
                "target_modules": ["q_proj", "v_proj"],
            },
        },
        "llada_config": {
            "pad_token_id": 2050,
            "eos_token_id": 2049,
            "use_cache": False,
        },
    }
    with pytest.raises(ValueError, match="must be true when model.backbone.name='llada'"):
        resolve_backbone_spec(model_cfg)
