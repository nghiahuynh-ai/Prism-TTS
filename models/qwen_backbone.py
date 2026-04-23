from __future__ import annotations

import sys
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

try:
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention,
        Qwen3ForCausalLM,
        Qwen3MLP,
        Qwen3Model,
        Qwen3PreTrainedModel,
        Qwen3RMSNorm,
        Qwen3RotaryEmbedding,
    )

    _QWEN3_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    Qwen3Config = type("Qwen3Config", (), {})  # type: ignore[assignment]
    Qwen3Attention = None  # type: ignore[assignment]
    Qwen3ForCausalLM = None  # type: ignore[assignment]
    Qwen3MLP = None  # type: ignore[assignment]
    Qwen3Model = None  # type: ignore[assignment]
    Qwen3PreTrainedModel = None  # type: ignore[assignment]
    Qwen3RMSNorm = None  # type: ignore[assignment]
    Qwen3RotaryEmbedding = None  # type: ignore[assignment]
    _QWEN3_IMPORT_ERROR = exc


def _require_qwen3_support() -> None:
    if _QWEN3_IMPORT_ERROR is None:
        return
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unknown"

    raise ImportError(
        "QwenBackbone requires Qwen3 support from `transformers`, but the current "
        "environment cannot import `transformers.models.qwen3`. "
        f"Current python: {sys.executable}. "
        f"Current transformers: {transformers_version}. "
        "Install/upgrade with `python -m pip install -U 'transformers>=4.51.0'` "
        "in this same environment."
    ) from _QWEN3_IMPORT_ERROR


_QWEN_BACKBONE_BASE = Qwen3PreTrainedModel if Qwen3PreTrainedModel is not None else nn.Module


def _to_additive_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if mask.dtype.is_floating_point:
        return mask.to(dtype=dtype)

    mask = mask.to(dtype=dtype)
    min_dtype = torch.finfo(dtype).min
    return (1.0 - mask).masked_fill((1.0 - mask).to(torch.bool), min_dtype)


def _expand_padding_mask_to_4d(
    attention_mask: torch.Tensor,
    query_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if attention_mask.dim() != 2:
        raise ValueError("Padding attention_mask must be 2D [batch, key_len].")

    batch_size, key_len = attention_mask.shape
    if key_len < target_length:
        raise ValueError(
            f"attention_mask key length ({key_len}) is shorter than target_length ({target_length})."
        )

    key_mask = attention_mask[:, :target_length].to(device=device, dtype=dtype)
    additive = _to_additive_mask(key_mask, dtype=dtype)
    return additive[:, None, None, :].expand(batch_size, 1, query_length, target_length)


class FullAttentionQwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        _require_qwen3_support()

        self.hidden_size = config.hidden_size
        self.attention_type = "full_attention"
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        # Prism-TTS uses masked reconstruction, so attention must stay bidirectional.
        self.self_attn.is_causal = False
        self.self_attn.sliding_window = None
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _run_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Cache],
        use_cache: Optional[bool],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        output_attentions: bool,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_kwargs = dict(kwargs)
        attn_kwargs["is_causal"] = False

        try:
            outputs = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                **attn_kwargs,
            )
        except TypeError:
            outputs = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **attn_kwargs,
            )

        hidden_states = outputs[0]
        attn_weights = outputs[1] if len(outputs) > 1 else None
        return hidden_states, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        if position_embeddings is None:
            raise ValueError("position_embeddings must be provided for Qwen3 attention layers.")

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self._run_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if output_attentions:
            return hidden_states, attn_weights
        return hidden_states


class QwenBackbone(_QWEN_BACKBONE_BASE):
    config_class = Qwen3Config

    def __init__(self, config: Qwen3Config):
        _require_qwen3_support()
        if not isinstance(config, Qwen3Config):
            raise TypeError(
                "QwenBackbone expects Qwen3Config, "
                f"got {type(config).__name__}."
            )
        # Force non-causal full attention for masked reconstruction across all layers.
        config.layer_types = ["full_attention"] * int(config.num_hidden_layers)
        if hasattr(config, "use_sliding_window"):
            config.use_sliding_window = False
        if len(config.layer_types) < int(config.num_hidden_layers):
            raise ValueError("Qwen3 config layer_types must match num_hidden_layers.")

        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.has_sliding_layers = "sliding_attention" in config.layer_types

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                FullAttentionQwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    @property
    def layer_types(self) -> tuple[str, ...]:
        return tuple(self.config.layer_types[: self.config.num_hidden_layers])

    @property
    def unique_layer_types(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.layer_types))

    @staticmethod
    def _resolve_single_attention_mask(
        attention_mask: Optional[torch.Tensor],
        *,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[Cache],
    ) -> Optional[torch.Tensor]:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        sequence_length = int(inputs_embeds.shape[1])
        target_length = past_seen_tokens + sequence_length

        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            return _expand_padding_mask_to_4d(
                attention_mask=attention_mask,
                query_length=sequence_length,
                target_length=target_length,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
        if attention_mask.dim() == 4:
            return _to_additive_mask(attention_mask.to(device=inputs_embeds.device), inputs_embeds.dtype)
        raise ValueError("attention_mask must be either 2D padding mask or 4D additive mask.")

    def _resolve_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor | dict[str, Optional[torch.Tensor]]],
        *,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[Cache],
    ) -> Optional[torch.Tensor | dict[str, Optional[torch.Tensor]]]:
        if attention_mask is None or isinstance(attention_mask, torch.Tensor):
            return self._resolve_single_attention_mask(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
            )

        if not isinstance(attention_mask, dict):
            raise ValueError(
                "attention_mask must be None, a 2D/4D tensor, or a dict[layer_type -> mask tensor]."
            )

        resolved: dict[str, Optional[torch.Tensor]] = {}
        for layer_type, mask in attention_mask.items():
            resolved[layer_type] = self._resolve_single_attention_mask(
                attention_mask=mask,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
            )
        return resolved

    def _resolve_position_embeddings(
        self,
        *,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor] | dict[str, tuple[torch.Tensor, torch.Tensor]]],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        if position_embeddings is None:
            shared_embeddings = self.rotary_emb(hidden_states, position_ids)
            return {layer_type: shared_embeddings for layer_type in self.unique_layer_types}

        if isinstance(position_embeddings, tuple):
            return {layer_type: position_embeddings for layer_type in self.unique_layer_types}

        if not isinstance(position_embeddings, dict):
            raise ValueError(
                "position_embeddings must be None, a (cos, sin) tuple, or a dict[layer_type -> (cos, sin)]."
            )

        default_embedding = position_embeddings.get("default")
        resolved: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_type in self.unique_layer_types:
            layer_embedding = position_embeddings.get(layer_type, default_embedding)
            if layer_embedding is None:
                raise ValueError(
                    "Missing position embeddings for layer type "
                    f"{layer_type!r}. Provide tuple or dict with all required layer types."
                )
            resolved[layer_type] = layer_embedding
        return resolved

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor | dict[str, Optional[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor] | dict[str, tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> BaseModelOutputWithPast | tuple:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
            position_ids = position_ids.unsqueeze(0)

        resolved_attention_mask = self._resolve_attention_mask(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
        )
        resolved_position_embeddings = self._resolve_position_embeddings(
            hidden_states=inputs_embeds,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        default_mask: Optional[torch.Tensor] = None
        if isinstance(resolved_attention_mask, dict):
            default_mask = resolved_attention_mask.get("default")
            if default_mask is None:
                default_mask = resolved_attention_mask.get("full_attention")

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_type = self.config.layer_types[layer_idx]
            if isinstance(resolved_attention_mask, dict):
                layer_attention_mask = resolved_attention_mask.get(layer_type, default_mask)
            else:
                layer_attention_mask = resolved_attention_mask

            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=resolved_position_embeddings[layer_type],
                output_attentions=output_attentions,
                **kwargs,
            )

            if output_attentions:
                hidden_states, attn_weights = layer_outputs
                all_attentions += (attn_weights,)
            else:
                hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            outputs = [hidden_states, past_key_values]
            if output_hidden_states:
                outputs.append(all_hidden_states)
            if output_attentions:
                outputs.append(all_attentions)
            return tuple(outputs)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    @classmethod
    def from_qwen3_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        strict: bool = True,
        **kwargs: Any,
    ) -> "QwenBackbone":
        """
        Load Qwen3 text-backbone weights into a masked-LM QwenBackbone.

        This method first attempts direct loading via `QwenBackbone.from_pretrained`.
        If the checkpoint is wrapped by a causal LM head, it falls back to
        extracting `Qwen3ForCausalLM.model` weights.
        """
        _require_qwen3_support()

        config = kwargs.pop("config", None)
        if config is None:
            auto_config_kwargs = {
                key: kwargs[key]
                for key in (
                    "cache_dir",
                    "force_download",
                    "local_files_only",
                    "proxies",
                    "revision",
                    "subfolder",
                    "token",
                    "trust_remote_code",
                )
                if key in kwargs
            }
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                **auto_config_kwargs,
            )
        if not isinstance(config, Qwen3Config):
            raise TypeError(
                "from_qwen3_pretrained expects a Qwen3 checkpoint/config. "
                f"Got config type: {type(config).__name__}."
            )

        loading_kwargs = dict(kwargs)
        loading_kwargs["config"] = config
        direct_loading_error: Optional[Exception] = None

        try:
            direct_model, direct_info = cls.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                output_loading_info=True,
                **loading_kwargs,
            )
            if not direct_info.get("missing_keys"):
                return direct_model
            direct_loading_error = RuntimeError(
                "Direct load completed with missing keys: "
                + ", ".join(direct_info.get("missing_keys", [])[:10])
            )
        except Exception as exc:
            direct_loading_error = exc

        candidates: list[tuple[str, Any, Any]] = [
            ("Qwen3Model", Qwen3Model, lambda model: model),
            ("Qwen3ForCausalLM", Qwen3ForCausalLM, lambda model: model.model),
        ]

        attempt_errors: list[str] = []
        if direct_loading_error is not None:
            attempt_errors.append(
                f"Direct QwenBackbone.from_pretrained failed: "
                f"{type(direct_loading_error).__name__}: {direct_loading_error}"
            )

        for candidate_name, candidate_cls, candidate_extractor in candidates:
            reference_model = None
            try:
                reference_model, reference_info = candidate_cls.from_pretrained(
                    pretrained_model_name_or_path,
                    *model_args,
                    config=config,
                    output_loading_info=True,
                    **kwargs,
                )
                missing_candidate_keys = reference_info.get("missing_keys", [])
                if missing_candidate_keys:
                    missing_preview = ", ".join(missing_candidate_keys[:10])
                    raise RuntimeError(
                        f"Candidate produced missing keys ({len(missing_candidate_keys)}): {missing_preview}"
                    )

                reference_state = candidate_extractor(reference_model).state_dict()
                model = cls(config)
                missing, unexpected = model.load_state_dict(reference_state, strict=strict)
                if strict and (missing or unexpected):
                    missing_preview = ", ".join(missing[:10]) if missing else "none"
                    unexpected_preview = ", ".join(unexpected[:10]) if unexpected else "none"
                    raise RuntimeError(
                        "Unexpected key mismatch when loading fallback reference state. "
                        f"Missing: {missing_preview}. Unexpected: {unexpected_preview}."
                    )
                return model
            except Exception as exc:
                attempt_errors.append(f"{candidate_name}: {type(exc).__name__}: {exc}")
            finally:
                del reference_model

        joined_errors = "\n".join(attempt_errors)
        raise RuntimeError(
            "Unable to load Qwen3 pretrained backbone weights. "
            f"Attempts:\n{joined_errors}"
        )
