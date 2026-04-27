from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_utils import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast

try:
    from peft import LoraConfig, get_peft_model

    _PEFT_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    LoraConfig = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]
    _PEFT_IMPORT_ERROR = exc


DEFAULT_LLADA_CHECKPOINT = "GSAI-ML/LLaDA-8B-Base"

_HF_KWARG_KEYS: tuple[str, ...] = (
    "cache_dir",
    "force_download",
    "local_files_only",
    "proxies",
    "revision",
    "subfolder",
    "token",
)

_DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "attn_out",
    "ff_proj",
    "up_proj",
    "ff_out",
)


def _require_peft_support() -> None:
    if _PEFT_IMPORT_ERROR is None:
        return
    raise ImportError(
        "LLaDA LoRA adapters require `peft` to be installed in this environment."
    ) from _PEFT_IMPORT_ERROR


def _ensure_transformers_tied_weights_compat() -> None:
    """
    Compatibility shim for transformers builds where internal loading code
    expects `PreTrainedModel.all_tied_weights_keys`, but the attribute is absent.
    """
    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    def _all_tied_weights_keys(self: PreTrainedModel) -> dict[str, str]:
        return self.get_expanded_tied_weights_keys(all_submodels=True)

    PreTrainedModel.all_tied_weights_keys = property(_all_tied_weights_keys)  # type: ignore[attr-defined]


def _patch_llada_dynamic_model_compat(
    *,
    checkpoint_name: str,
    config: Any,
    hf_kwargs: Mapping[str, Any],
) -> None:
    auto_map = getattr(config, "auto_map", None)
    if not isinstance(auto_map, Mapping):
        return

    class_reference = auto_map.get("AutoModel") or auto_map.get("AutoModelForCausalLM")
    if not isinstance(class_reference, str):
        return

    dynamic_import_kwargs: dict[str, Any] = {}
    for key in ("cache_dir", "force_download", "local_files_only", "proxies", "revision", "token"):
        if key in hf_kwargs:
            dynamic_import_kwargs[key] = hf_kwargs[key]

    model_class = get_class_from_dynamic_module(
        class_reference,
        checkpoint_name,
        **dynamic_import_kwargs,
    )
    tie_weights = getattr(model_class, "tie_weights", None)
    if not callable(tie_weights):
        return

    signature = inspect.signature(tie_weights)
    if "missing_keys" in signature.parameters and "recompute_mapping" in signature.parameters:
        return
    if getattr(tie_weights, "_prism_tts_compat_patched", False):
        return

    def _tie_weights_compat(self: nn.Module, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("missing_keys", None)
        kwargs.pop("recompute_mapping", None)
        return tie_weights(self, *args, **kwargs)

    _tie_weights_compat._prism_tts_compat_patched = True  # type: ignore[attr-defined]
    model_class.tie_weights = _tie_weights_compat


def _to_float_additive_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if mask.dtype.is_floating_point:
        return mask.to(dtype=dtype)

    cast_mask = mask.to(dtype=dtype)
    min_dtype = torch.finfo(dtype).min
    return (1.0 - cast_mask).masked_fill((1.0 - cast_mask).to(torch.bool), min_dtype)


class LladaBackbone(nn.Module):
    """
    Prism-TTS wrapper around trust-remote-code LLaDA checkpoints.

    This wrapper exposes:
    - `embed_tokens` like other backbones
    - `forward(...)->BaseModelOutputWithPast` with `last_hidden_state`
    - mandatory PEFT LoRA injection for adapter-only fine-tuning.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        config: Any,
        lora_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.gradient_checkpointing = False
        self.lora_config = dict(lora_config or {})
        self.lora_config.setdefault("enabled", True)
        self.lora_config.setdefault("target_modules", list(_DEFAULT_LORA_TARGET_MODULES))
        self.lora_enabled = False
        self._apply_lora(self.lora_config)

    @property
    def embed_tokens(self) -> nn.Embedding:
        input_embeddings = self.model.get_input_embeddings()
        if not isinstance(input_embeddings, nn.Embedding):
            raise TypeError(
                "LLaDA backbone input embedding is expected to be nn.Embedding, "
                f"got {type(input_embeddings).__name__}."
            )
        return input_embeddings

    @staticmethod
    def _resolve_trust_remote_code_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
        resolved = dict(kwargs)
        # LLaDA checkpoints rely on custom HF code modules; this must stay enabled.
        resolved["trust_remote_code"] = True
        return resolved

    @staticmethod
    def _build_config_from_payload(
        *,
        config_payload: Any,
        checkpoint_name: str,
        hf_kwargs: Mapping[str, Any],
    ) -> Any:
        if config_payload is not None and not isinstance(config_payload, Mapping):
            model_type = getattr(config_payload, "model_type", None)
            if model_type != "llada":
                raise TypeError(
                    "LladaBackbone expects an LLaDA config object or mapping overrides, "
                    f"got config with model_type={model_type!r}."
                )
            return config_payload

        config_overrides = dict(config_payload or {})
        config_loading_kwargs = {
            key: hf_kwargs[key]
            for key in _HF_KWARG_KEYS
            if key in hf_kwargs
        }
        config_loading_kwargs = LladaBackbone._resolve_trust_remote_code_kwargs(config_loading_kwargs)
        return AutoConfig.from_pretrained(
            checkpoint_name,
            **config_loading_kwargs,
            **config_overrides,
        )

    @classmethod
    def from_llada_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        config: Optional[Any] = None,
        strict: bool = True,
        lora_config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "LladaBackbone":
        del strict  # Trust-remote-code checkpoints do not expose strict loading info here.

        _ensure_transformers_tied_weights_compat()
        loading_kwargs = cls._resolve_trust_remote_code_kwargs(kwargs)
        resolved_config = cls._build_config_from_payload(
            config_payload=config,
            checkpoint_name=pretrained_model_name_or_path,
            hf_kwargs=loading_kwargs,
        )
        _patch_llada_dynamic_model_compat(
            checkpoint_name=pretrained_model_name_or_path,
            config=resolved_config,
            hf_kwargs=loading_kwargs,
        )
        loading_kwargs["config"] = resolved_config

        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **loading_kwargs,
        )
        return cls(model=model, config=resolved_config, lora_config=lora_config)

    @classmethod
    def from_llada_config(
        cls,
        config: Any,
        *,
        config_code_source: str = DEFAULT_LLADA_CHECKPOINT,
        lora_config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "LladaBackbone":
        """
        Build a randomly initialized LLaDA backbone from config.

        If `config` is a mapping, the LLaDA config class is resolved from
        `config_code_source` (default: official 8B checkpoint).
        """
        resolved_config = cls._build_config_from_payload(
            config_payload=config,
            checkpoint_name=config_code_source,
            hf_kwargs=kwargs,
        )
        model_loading_kwargs = cls._resolve_trust_remote_code_kwargs({})
        model = AutoModel.from_config(
            resolved_config,
            **model_loading_kwargs,
        )
        return cls(model=model, config=resolved_config, lora_config=lora_config)

    def _apply_lora(self, lora_config: Mapping[str, Any]) -> None:
        enabled = bool(lora_config.get("enabled", True))
        if not enabled:
            raise ValueError(
                "model.backbone.lora.enabled must be true when model.backbone.name='llada'."
            )

        _require_peft_support()

        raw_r = int(lora_config.get("r", 16))
        if raw_r < 1:
            raise ValueError("model.backbone.lora.r must be >= 1.")
        raw_alpha = int(lora_config.get("alpha", lora_config.get("lora_alpha", 32)))
        if raw_alpha < 1:
            raise ValueError("model.backbone.lora.alpha must be >= 1.")
        raw_dropout = float(lora_config.get("dropout", lora_config.get("lora_dropout", 0.05)))
        if raw_dropout < 0.0:
            raise ValueError("model.backbone.lora.dropout must be >= 0.")
        bias = str(lora_config.get("bias", "none"))
        if bias not in {"none", "all", "lora_only"}:
            raise ValueError(
                "model.backbone.lora.bias must be one of: none, all, lora_only."
            )

        target_modules_raw = lora_config.get("target_modules")
        if target_modules_raw is None:
            target_modules_raw = _DEFAULT_LORA_TARGET_MODULES
        if isinstance(target_modules_raw, str):
            target_modules = [target_modules_raw]
        elif isinstance(target_modules_raw, (list, tuple)):
            target_modules = [str(module_name) for module_name in target_modules_raw]
        else:
            raise ValueError(
                "model.backbone.lora.target_modules must be a string or list of strings."
            )
        if len(target_modules) == 0:
            raise ValueError("model.backbone.lora.target_modules must not be empty.")

        modules_to_save_raw = lora_config.get("modules_to_save")
        modules_to_save: Optional[list[str]]
        if modules_to_save_raw is None:
            modules_to_save = None
        elif isinstance(modules_to_save_raw, (list, tuple)):
            modules_to_save = [str(module_name) for module_name in modules_to_save_raw]
        else:
            raise ValueError("model.backbone.lora.modules_to_save must be a list when provided.")

        lora_init = lora_config.get("init_lora_weights", True)
        use_rslora = bool(lora_config.get("use_rslora", False))

        peft_config = LoraConfig(
            r=raw_r,
            lora_alpha=raw_alpha,
            lora_dropout=raw_dropout,
            bias=bias,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            init_lora_weights=lora_init,
            use_rslora=use_rslora,
            task_type=None,
        )
        self.model = get_peft_model(self.model, peft_config)
        self.lora_enabled = True

    def _resolve_final_layer_norm_module(self) -> nn.Module:
        candidates = [
            ("model", "transformer", "ln_f"),
            ("transformer", "ln_f"),
        ]
        for path in candidates:
            cursor: Any = self.model
            valid = True
            for attr in path:
                if not hasattr(cursor, attr):
                    valid = False
                    break
                cursor = getattr(cursor, attr)
            if valid and isinstance(cursor, nn.Module):
                return cursor

        # If LoRA wraps the base model, use the unwrapped model namespace.
        get_base_model = getattr(self.model, "get_base_model", None)
        if callable(get_base_model):
            base_model = get_base_model()
            for path in candidates:
                cursor = base_model
                valid = True
                for attr in path:
                    if not hasattr(cursor, attr):
                        valid = False
                        break
                    cursor = getattr(cursor, attr)
                if valid and isinstance(cursor, nn.Module):
                    return cursor

        raise RuntimeError(
            "Unable to resolve LLaDA final layer norm module for hidden-state capture."
        )

    def _forward_with_last_hidden_hook(
        self,
        *,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.Tensor],
        attention_bias: Optional[torch.Tensor],
    ) -> torch.FloatTensor:
        captured: dict[str, torch.Tensor] = {}
        final_norm = self._resolve_final_layer_norm_module()

        def _capture_hidden(
            module: nn.Module, module_inputs: tuple[torch.Tensor, ...], module_output: torch.Tensor
        ) -> None:
            del module, module_inputs
            captured["last_hidden_state"] = module_output

        hook = final_norm.register_forward_hook(_capture_hidden)
        try:
            self.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            hook.remove()

        if "last_hidden_state" not in captured:
            raise RuntimeError("Failed to capture LLaDA hidden states from forward pass.")
        return captured["last_hidden_state"]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast | tuple:
        del position_ids, position_embeddings, cache_position

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
        if past_key_values is not None:
            raise ValueError("LLaDA backbone does not support past_key_values in Prism-TTS mode.")
        if use_cache:
            raise ValueError("LLaDA backbone use_cache=True is unsupported in Prism-TTS mode.")
        if output_attentions:
            raise ValueError("LLaDA backbone does not expose attention weights.")

        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments for LladaBackbone.forward: {unknown}")

        model_attention_mask: Optional[torch.Tensor] = None
        attention_bias: Optional[torch.Tensor] = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                model_attention_mask = attention_mask
            elif attention_mask.dim() == 4:
                model_attention_mask = None
                dtype = inputs_embeds.dtype if inputs_embeds is not None else torch.float32
                attention_bias = _to_float_additive_mask(attention_mask, dtype=dtype)
            else:
                raise ValueError("attention_mask must be 2D padding mask or 4D additive mask.")

        if output_hidden_states:
            outputs = self.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=model_attention_mask,
                attention_bias=attention_bias,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            if outputs.hidden_states is None or len(outputs.hidden_states) == 0:
                raise RuntimeError("LLaDA did not return hidden_states despite output_hidden_states=True.")
            hidden_states = outputs.hidden_states
            last_hidden_state = hidden_states[-1]
        else:
            hidden_states = None
            last_hidden_state = self._forward_with_last_hidden_hook(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=model_attention_mask,
                attention_bias=attention_bias,
            )

        if not return_dict:
            payload = [last_hidden_state, None]
            if output_hidden_states:
                payload.append(hidden_states)
            return tuple(payload)

        return BaseModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            past_key_values=None,
            hidden_states=hidden_states,
            attentions=None,
        )
