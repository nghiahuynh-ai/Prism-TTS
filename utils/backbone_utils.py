from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch.nn as nn
from transformers import LlamaConfig

if TYPE_CHECKING:
    from models.llama_backbone import LlamaBackbone

__all__ = [
    "BackboneSpec",
    "build_backbone_text_tokenizer",
    "build_backbone",
    "normalize_backbone_name",
    "normalize_optional_string",
    "resolve_backbone_config_key",
    "resolve_backbone_spec",
    "should_use_checkpoint_text_tokenizer",
]


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    config_key: str
    config: Optional[dict[str, Any]]
    hf_checkpoint: Optional[str]
    hf_strict: bool
    hf_kwargs: dict[str, Any]


_BACKBONE_NAME_ALIASES: dict[str, str] = {
    "llama": "llama",
    "llama2": "llama",
    "llama3": "llama",
    "gemma": "gemma",
    "gemma3": "gemma",
    "qwen": "qwen",
    "qwen3": "qwen",
}

_HF_FLAT_KWARG_MAP: dict[str, str] = {
    "hf_token": "token",
    "hf_revision": "revision",
    "hf_cache_dir": "cache_dir",
    "hf_subfolder": "subfolder",
    "hf_trust_remote_code": "trust_remote_code",
    "hf_local_files_only": "local_files_only",
    "hf_force_download": "force_download",
}

_HF_KWARG_KEYS: tuple[str, ...] = (
    "token",
    "revision",
    "cache_dir",
    "subfolder",
    "trust_remote_code",
    "local_files_only",
    "force_download",
)

_HF_BOOL_KWARG_KEYS: tuple[str, ...] = (
    "trust_remote_code",
    "local_files_only",
    "force_download",
)

_HF_ENV_KWARG_MAP: dict[str, tuple[str, ...]] = {
    "token": ("PRISM_TTS_HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"),
    "revision": ("PRISM_TTS_HF_REVISION",),
    "cache_dir": ("PRISM_TTS_HF_CACHE_DIR",),
    "subfolder": ("PRISM_TTS_HF_SUBFOLDER",),
    "trust_remote_code": ("PRISM_TTS_HF_TRUST_REMOTE_CODE",),
    "local_files_only": ("PRISM_TTS_HF_LOCAL_FILES_ONLY",),
    "force_download": ("PRISM_TTS_HF_FORCE_DOWNLOAD",),
}


def normalize_optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized


def _read_first_env_var(*keys: str) -> Optional[str]:
    for key in keys:
        value = normalize_optional_string(os.environ.get(key))
        if value is not None:
            return value
    return None


def _parse_bool_text(raw_value: str, *, field_name: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean value for {field_name}: {raw_value!r}. "
        "Expected one of: 1/0, true/false, yes/no, on/off."
    )


def _read_optional_env_bool(*keys: str) -> Optional[bool]:
    for key in keys:
        raw = normalize_optional_string(os.environ.get(key))
        if raw is not None:
            return _parse_bool_text(raw, field_name=key)
    return None


def normalize_backbone_name(backbone_name: Any) -> str:
    candidate = "llama" if backbone_name is None else str(backbone_name).strip().lower()
    normalized = _BACKBONE_NAME_ALIASES.get(candidate)
    if normalized is not None:
        return normalized

    supported = ", ".join(sorted(set(_BACKBONE_NAME_ALIASES.values())))
    raise ValueError(
        f"Unsupported model.backbone.name={backbone_name!r}. Supported backbones: {supported}."
    )


def should_use_checkpoint_text_tokenizer(backbone_name: Any) -> bool:
    """Return whether text tokenization must come from the HF backbone checkpoint."""
    return normalize_backbone_name(backbone_name) in {"gemma", "qwen"}


def build_backbone_text_tokenizer(
    *,
    backbone_name: Any,
    backbone_hf_checkpoint: Optional[str],
    backbone_hf_kwargs: Optional[Mapping[str, Any]] = None,
    require_checkpoint: bool = False,
) -> Optional[Any]:
    """
    Load the text tokenizer from a backbone checkpoint when required.

    Gemma and Qwen backbones rely on checkpoint-native tokenization so their
    pretrained text embedding table stays aligned with token ids.
    """
    if not should_use_checkpoint_text_tokenizer(backbone_name):
        return None

    checkpoint = normalize_optional_string(backbone_hf_checkpoint)
    if checkpoint is None:
        if require_checkpoint:
            raise ValueError(
                "Gemma/Qwen backbones require model.backbone.hf_checkpoint so text "
                "tokenization can be inherited from the pretrained checkpoint."
            )
        return None

    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise ImportError(
            "Loading Gemma/Qwen text tokenizer requires transformers AutoTokenizer support."
        ) from exc

    loading_kwargs: dict[str, Any] = {}
    for key in _HF_KWARG_KEYS:
        if backbone_hf_kwargs is not None and key in backbone_hf_kwargs:
            value = backbone_hf_kwargs[key]
            if value is not None:
                loading_kwargs[key] = value

    try:
        return AutoTokenizer.from_pretrained(checkpoint, **loading_kwargs)
    except Exception as exc:
        normalized_name = normalize_backbone_name(backbone_name)
        raise RuntimeError(
            "Unable to load text tokenizer from checkpoint "
            f"{checkpoint!r} for backbone {normalized_name!r}: {exc}"
        ) from exc


def resolve_backbone_config_key(backbone_name: str) -> str:
    return f"{normalize_backbone_name(backbone_name)}_config"


def resolve_backbone_spec(model_cfg: Mapping[str, Any]) -> BackboneSpec:
    backbone_payload = model_cfg.get("backbone")
    backbone_overrides: Mapping[str, Any]
    configured_backbone_name_raw: Any = "llama"

    if backbone_payload is None:
        backbone_overrides = {}
    elif isinstance(backbone_payload, str):
        backbone_overrides = {}
        configured_backbone_name_raw = backbone_payload
    elif isinstance(backbone_payload, Mapping):
        backbone_overrides = backbone_payload
        configured_backbone_name_raw = backbone_overrides.get("name", "llama")
    else:
        raise ValueError("model.backbone must be a mapping, string, or null.")

    configured_backbone_name = normalize_backbone_name(configured_backbone_name_raw)
    env_backbone_name_raw = _read_first_env_var(
        "PRISM_TTS_BACKBONE_NAME",
        "PRISM_TTS_BACKBONE",
    )
    backbone_name_raw = env_backbone_name_raw or configured_backbone_name_raw
    backbone_name = normalize_backbone_name(backbone_name_raw)

    config_key_raw = _read_first_env_var("PRISM_TTS_BACKBONE_CONFIG_KEY")
    if config_key_raw is None:
        config_key_raw = backbone_overrides.get("config_key")
    if config_key_raw is None:
        config_key = resolve_backbone_config_key(backbone_name)
    else:
        if not isinstance(config_key_raw, str) or not config_key_raw.strip():
            raise ValueError("model.backbone.config_key must be a non-empty string when provided.")
        config_key = config_key_raw.strip()

    configured_hf_checkpoint = normalize_optional_string(
        backbone_overrides.get(
            "hf_checkpoint",
            backbone_overrides.get(
                "checkpoint",
                backbone_overrides.get("pretrained_model_name_or_path"),
            ),
        )
    )
    env_hf_checkpoint = _read_first_env_var(
        "PRISM_TTS_BACKBONE_HF_CHECKPOINT",
        "PRISM_TTS_HF_CHECKPOINT",
    )
    hf_checkpoint = env_hf_checkpoint or configured_hf_checkpoint
    if (
        env_backbone_name_raw is not None
        and backbone_name != configured_backbone_name
        and env_hf_checkpoint is None
    ):
        # Avoid accidentally reusing a checkpoint configured for another backbone.
        hf_checkpoint = None
    hf_strict = bool(backbone_overrides.get("hf_strict", True))
    hf_strict_env = _read_optional_env_bool(
        "PRISM_TTS_BACKBONE_HF_STRICT",
        "PRISM_TTS_HF_STRICT",
    )
    if hf_strict_env is not None:
        hf_strict = hf_strict_env

    hf_kwargs: dict[str, Any] = {}
    hf_payload = backbone_overrides.get("hf")
    if hf_payload is not None:
        if not isinstance(hf_payload, Mapping):
            raise ValueError("model.backbone.hf must be a mapping when provided.")
        for key in _HF_KWARG_KEYS:
            value = hf_payload.get(key)
            if value is not None:
                hf_kwargs[key] = value

    for src_key, dst_key in _HF_FLAT_KWARG_MAP.items():
        value = backbone_overrides.get(src_key)
        if value is not None:
            hf_kwargs[dst_key] = value

    for key in ("token", "revision", "cache_dir", "subfolder"):
        if key in hf_kwargs:
            normalized = normalize_optional_string(hf_kwargs[key])
            if normalized is None:
                hf_kwargs.pop(key, None)
            else:
                hf_kwargs[key] = normalized

    for key in _HF_BOOL_KWARG_KEYS:
        if key in hf_kwargs:
            hf_kwargs[key] = bool(hf_kwargs[key])

    for key, env_var_names in _HF_ENV_KWARG_MAP.items():
        if key in _HF_BOOL_KWARG_KEYS:
            env_bool = _read_optional_env_bool(*env_var_names)
            if env_bool is not None:
                hf_kwargs[key] = env_bool
            continue
        env_value = _read_first_env_var(*env_var_names)
        if env_value is not None:
            hf_kwargs[key] = env_value

    config_payload = model_cfg.get(config_key)
    resolved_config: Optional[dict[str, Any]]
    if config_payload is None:
        if hf_checkpoint is None:
            raise ValueError(
                f"Missing required mapping model.{config_key} for selected backbone "
                f"{backbone_name!r}. Add model.{config_key} or set model.backbone.hf_checkpoint."
            )
        resolved_config = None
    elif isinstance(config_payload, Mapping):
        resolved_config = dict(config_payload)
    else:
        raise ValueError(f"model.{config_key} must be a mapping when provided.")

    return BackboneSpec(
        name=backbone_name,
        config_key=config_key,
        config=resolved_config,
        hf_checkpoint=hf_checkpoint,
        hf_strict=hf_strict,
        hf_kwargs=hf_kwargs,
    )


def _coerce_backbone_config(backbone_name: str, backbone_config: Any) -> Any:
    if backbone_name == "llama":
        if isinstance(backbone_config, LlamaConfig):
            return backbone_config
        if isinstance(backbone_config, Mapping):
            return LlamaConfig(**dict(backbone_config))
        raise TypeError(
            "Llama backbone expects a LlamaConfig or mapping-compatible payload, "
            f"got {type(backbone_config).__name__}."
        )

    if backbone_name == "gemma":
        try:
            from transformers import Gemma3Config, Gemma3TextConfig
        except Exception as exc:
            raise ImportError(
                "Gemma backbone requires Gemma3 config support in transformers. "
                "Install/upgrade with `python -m pip install -U 'transformers>=4.50.0'`."
            ) from exc

        if isinstance(backbone_config, (Gemma3TextConfig, Gemma3Config)):
            return backbone_config
        if isinstance(backbone_config, Mapping):
            payload = dict(backbone_config)
            if isinstance(payload.get("text_config"), Mapping):
                return Gemma3Config(**payload)
            return Gemma3TextConfig(**payload)
        raise TypeError(
            "Gemma backbone expects Gemma3TextConfig/Gemma3Config or mapping-compatible payload, "
            f"got {type(backbone_config).__name__}."
        )

    if backbone_name == "qwen":
        try:
            from transformers import Qwen3Config
        except Exception as exc:
            raise ImportError(
                "Qwen backbone requires Qwen3 config support in transformers. "
                "Install/upgrade with `python -m pip install -U 'transformers>=4.51.0'`."
            ) from exc

        if isinstance(backbone_config, Qwen3Config):
            return backbone_config
        if isinstance(backbone_config, Mapping):
            return Qwen3Config(**dict(backbone_config))
        raise TypeError(
            "Qwen backbone expects Qwen3Config or mapping-compatible payload, "
            f"got {type(backbone_config).__name__}."
        )

    raise ValueError(f"Unsupported backbone {backbone_name!r}.")


def _load_llama_backbone_from_pretrained(
    checkpoint_name: str,
    *,
    config: Optional[LlamaConfig],
    strict: bool,
    kwargs: dict[str, Any],
) -> "LlamaBackbone":
    from models.llama_backbone import LlamaBackbone

    loading_kwargs = dict(kwargs)
    if config is not None:
        loading_kwargs["config"] = config
    try:
        model, loading_info = LlamaBackbone.from_pretrained(
            checkpoint_name,
            output_loading_info=True,
            **loading_kwargs,
        )
    except TypeError:
        # Older transformers may not support output_loading_info.
        return LlamaBackbone.from_pretrained(checkpoint_name, **loading_kwargs)

    if strict:
        missing = list(loading_info.get("missing_keys", []))
        unexpected = list(loading_info.get("unexpected_keys", []))
        mismatched = list(loading_info.get("mismatched_keys", []))
        if missing or unexpected or mismatched:
            missing_preview = ", ".join(missing[:10]) if missing else "none"
            unexpected_preview = ", ".join(unexpected[:10]) if unexpected else "none"
            mismatched_preview = ", ".join(str(item) for item in mismatched[:3]) if mismatched else "none"
            raise RuntimeError(
                "Unable to strictly load Llama backbone from checkpoint. "
                f"Missing: {missing_preview}. Unexpected: {unexpected_preview}. "
                f"Mismatched: {mismatched_preview}."
            )
    return model


def build_backbone(
    *,
    backbone_name: str,
    backbone_config: Optional[Any],
    backbone_hf_checkpoint: Optional[str],
    backbone_hf_strict: bool,
    backbone_hf_kwargs: Optional[dict[str, Any]],
) -> nn.Module:
    resolved_name = normalize_backbone_name(backbone_name)
    resolved_config: Optional[Any] = None
    if backbone_config is not None:
        resolved_config = _coerce_backbone_config(resolved_name, backbone_config)
    resolved_hf_kwargs = dict(backbone_hf_kwargs or {})

    if not backbone_hf_checkpoint:
        if resolved_config is None:
            raise ValueError(
                "backbone_config is required when model.backbone.hf_checkpoint is not set."
            )
        if resolved_name == "llama":
            from models.llama_backbone import LlamaBackbone

            return LlamaBackbone(resolved_config)
        if resolved_name == "gemma":
            from models.gemma_backbone import GemmaBackbone

            return GemmaBackbone(resolved_config)
        from models.qwen_backbone import QwenBackbone

        return QwenBackbone(resolved_config)

    if resolved_name == "llama":
        return _load_llama_backbone_from_pretrained(
            backbone_hf_checkpoint,
            config=resolved_config,
            strict=backbone_hf_strict,
            kwargs=resolved_hf_kwargs,
        )
    if resolved_name == "gemma":
        from models.gemma_backbone import GemmaBackbone

        loading_kwargs = dict(resolved_hf_kwargs)
        if resolved_config is not None:
            loading_kwargs["config"] = resolved_config
        return GemmaBackbone.from_gemma3_pretrained(
            backbone_hf_checkpoint,
            strict=backbone_hf_strict,
            **loading_kwargs,
        )
    from models.qwen_backbone import QwenBackbone

    loading_kwargs = dict(resolved_hf_kwargs)
    if resolved_config is not None:
        loading_kwargs["config"] = resolved_config
    return QwenBackbone.from_qwen3_pretrained(
        backbone_hf_checkpoint,
        strict=backbone_hf_strict,
        **loading_kwargs,
    )
