from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)


def _to_additive_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if mask.dtype.is_floating_point:
        return mask.to(dtype=dtype)

    mask = mask.to(dtype=dtype)
    min_dtype = torch.finfo(dtype).min
    return (1.0 - mask).masked_fill((1.0 - mask).to(torch.bool), min_dtype)


def _build_causal_4d_mask(
    padding_mask: Optional[torch.Tensor],
    batch_size: int,
    query_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_position: Optional[torch.LongTensor],
    window: Optional[int] = None,
) -> torch.Tensor:
    """Build a 4D additive causal mask, optionally combined with a key-padding mask.

    PrismTTS is a purely autoregressive model, so attention must be causal. Right
    padding plus causality guarantees every query attends to at least position 0,
    avoiding all-`-inf` rows. When `window` is set, attention is additionally limited
    to the most recent `window` positions (sliding-window causal, used by the
    short-context conditioner).
    """
    min_value = torch.finfo(dtype).min

    if cache_position is not None:
        query_positions = cache_position.to(device=device).view(query_length, 1)
    else:
        # No KV cache: queries occupy the last `query_length` positions.
        offset = target_length - query_length
        query_positions = (
            torch.arange(query_length, device=device) + offset
        ).view(query_length, 1)
    key_positions = torch.arange(target_length, device=device).view(1, target_length)

    # True where a key is in the future of the query (must be masked).
    causal_bool = key_positions > query_positions  # [q, kv]
    if window is not None and window > 0:
        causal_bool = causal_bool | ((query_positions - key_positions) >= window)
    causal = torch.zeros(query_length, target_length, dtype=dtype, device=device)
    causal = causal.masked_fill(causal_bool, min_value)
    mask = causal[None, None].expand(batch_size, 1, query_length, target_length).clone()

    if padding_mask is not None:
        if padding_mask.dim() != 2:
            raise ValueError("Padding attention_mask must be 2D [batch, key_len].")
        if padding_mask.shape[1] < target_length:
            raise ValueError(
                f"attention_mask key length ({padding_mask.shape[1]}) is shorter than "
                f"target_length ({target_length})."
            )
        key_pad = ~padding_mask[:, :target_length].to(device=device, dtype=torch.bool)
        mask = mask.masked_fill(key_pad[:, None, None, :], min_value)

    return mask


def build_bidirectional_4d_mask(
    padding_mask: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a 4D additive *bidirectional* mask (only padded keys are masked).

    Used by the masked-generative (MAR) variant: every query may attend to every
    non-padded key (no causal triangle). The singleton query dimension is
    intentional: attention kernels broadcast it over all queries, so this stores
    ``O(batch * sequence)`` values rather than a dense ``O(batch * sequence^2)``
    mask. The latter was a multi-GiB allocation for long training examples.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    if padding_mask is not None:
        if padding_mask.dim() != 2:
            raise ValueError("Padding attention_mask must be 2D [batch, key_len].")
        if padding_mask.shape[0] != batch_size:
            raise ValueError(
                f"attention_mask batch size ({padding_mask.shape[0]}) does not match "
                f"batch_size ({batch_size})."
            )
        if padding_mask.shape[1] < seq_len:
            raise ValueError(
                f"attention_mask key length ({padding_mask.shape[1]}) is shorter than "
                f"seq_len ({seq_len})."
            )
        return _to_additive_mask(
            padding_mask[:, :seq_len].to(device=device), dtype=dtype
        )[:, None, None, :]

    # A non-null zero mask explicitly selects bidirectional SDPA. Its singleton
    # batch/query/key dimensions broadcast without a quadratic allocation.
    return torch.zeros((1, 1, 1, 1), dtype=dtype, device=device)


class FullAttentionLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _run_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Cache],
        use_cache: Optional[bool],
        cache_position: Optional[torch.LongTensor],
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
        output_attentions: bool,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        try:
            outputs = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        except TypeError:
            outputs = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
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
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self._run_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
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


class WindowedLlamaDecoderLayer(FullAttentionLlamaDecoderLayer):
    """Llama decoder block with memory-bounded sliding-window attention.

    ``scaled_dot_product_attention`` cannot express a sliding window using its
    ``is_causal`` flag. Constructing a dense [B, H, L, L] mask to do so defeats
    the point of a local conditioner. This implementation gathers only the local
    keys/values for a small query chunk, keeping the temporary footprint linear
    in sequence length (and bounded by ``chunk_size * window`` per operation).
    """

    def _windowed_attention(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        window: int,
        chunk_size: int,
    ) -> torch.Tensor:
        attention = self.self_attn
        batch_size, seq_len, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, attention.head_dim)

        query_states = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        try:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        except TypeError:
            # transformers<=4.x accepted position_ids as a required argument.
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        if key_states.shape[1] != query_states.shape[1]:
            if query_states.shape[1] % key_states.shape[1] != 0:
                raise ValueError("The query-head count must be divisible by the KV-head count.")
            repeat = query_states.shape[1] // key_states.shape[1]
            key_states = key_states.repeat_interleave(repeat, dim=1)
            value_states = value_states.repeat_interleave(repeat, dim=1)

        padding = None
        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.shape != (batch_size, seq_len):
                raise ValueError("attention_mask must have shape [batch, sequence].")
            padding = attention_mask.to(device=hidden_states.device, dtype=torch.bool)

        offsets = torch.arange(window, device=hidden_states.device) - (window - 1)
        dropout_p = float(getattr(attention, "attention_dropout", 0.0)) if self.training else 0.0
        scaling = getattr(attention, "scaling", attention.head_dim ** -0.5)
        outputs: list[torch.Tensor] = []
        for start in range(0, seq_len, chunk_size):
            end = min(seq_len, start + chunk_size)
            query_positions = torch.arange(start, end, device=hidden_states.device)
            key_positions = query_positions[:, None] + offsets[None, :]
            valid_keys = key_positions >= 0
            key_positions = key_positions.clamp_min(0)

            # [B, H, Q, W, D]; Q is a batch-like dimension for SDPA below.
            local_keys = key_states[:, :, key_positions, :]
            local_values = value_states[:, :, key_positions, :]
            allowed = valid_keys[None, None, :, None, :]
            if padding is not None:
                allowed = allowed & padding[:, None, key_positions].unsqueeze(-2)

            local_output = F.scaled_dot_product_attention(
                query_states[:, :, start:end, :].unsqueeze(-2),
                local_keys,
                local_values,
                attn_mask=allowed,
                dropout_p=dropout_p,
                scale=scaling,
                is_causal=False,
            ).squeeze(-2)
            outputs.append(local_output)

        attn_output = torch.cat(outputs, dim=2)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1).contiguous()
        return attention.o_proj(attn_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        window: int,
        chunk_size: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._windowed_attention(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            window=window,
            chunk_size=chunk_size,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class LlamaBackbone(LlamaPreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [FullAttentionLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def _resolve_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[Cache],
        cache_position: Optional[torch.LongTensor],
    ) -> Optional[torch.Tensor]:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        sequence_length = int(inputs_embeds.shape[1])
        target_length = past_seen_tokens + sequence_length
        batch_size = int(inputs_embeds.shape[0])

        # A 4D mask is treated as a fully specified additive mask (already causal).
        if attention_mask is not None and attention_mask.dim() == 4:
            return _to_additive_mask(
                attention_mask.to(device=inputs_embeds.device),
                inputs_embeds.dtype,
            )
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError("attention_mask must be either 2D padding mask or 4D additive mask.")

        return _build_causal_4d_mask(
            padding_mask=attention_mask,
            batch_size=batch_size,
            query_length=sequence_length,
            target_length=target_length,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
            cache_position=cache_position,
        )

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
        **kwargs,
    ) -> BaseModelOutputWithPast | tuple:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        resolved_attention_mask = self._resolve_attention_mask(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

        hidden_states = inputs_embeds
        if position_embeddings is None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        checkpoint_layers = self.gradient_checkpointing and self.training and not output_attentions
        if checkpoint_layers and use_cache:
            # Replaying a checkpointed layer must not append to a KV cache twice.
            use_cache = False

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if checkpoint_layers:
                layer_outputs = checkpoint(
                    lambda states, layer=decoder_layer: layer(
                        hidden_states=states,
                        attention_mask=resolved_attention_mask,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        output_attentions=False,
                        **kwargs,
                    ),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=resolved_attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
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


class WindowedCausalEncoder(nn.Module):
    """Lightweight sliding-window causal transformer for local (short-context) conditioning.

    Each position attends only to the most recent `window` positions (including
    itself). Used by PrismTTS as the CALM-style short-context branch whose output is
    added to the long-context backbone hidden state before the per-frame head.
    """

    def __init__(
        self,
        config: LlamaConfig,
        num_layers: int,
        window: int,
        chunk_size: int = 128,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("WindowedCausalEncoder num_layers must be >= 1.")
        if window < 1:
            raise ValueError("WindowedCausalEncoder window must be >= 1.")
        if chunk_size < 1:
            raise ValueError("WindowedCausalEncoder chunk_size must be >= 1.")
        self.window = int(window)
        self.chunk_size = int(chunk_size)
        self.layers = nn.ModuleList(
            [WindowedLlamaDecoderLayer(config, layer_idx) for layer_idx in range(num_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        cache_position = torch.arange(seq_len, device=device)
        position_ids = cache_position.unsqueeze(0)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    lambda states, layer=decoder_layer: layer(
                        hidden_states=states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        window=self.window,
                        chunk_size=self.chunk_size,
                    ),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    window=self.window,
                    chunk_size=self.chunk_size,
                )
        return self.norm(hidden_states)
