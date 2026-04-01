from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


def _to_additive_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if mask.dtype.is_floating_point:
        return mask.to(dtype=dtype)

    mask = mask.to(dtype=dtype)
    inverted_mask = 1.0 - mask
    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_position: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if attention_mask is not None and attention_mask.dim() == 4:
        return _to_additive_mask(attention_mask.to(device=device), dtype)

    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full(
        (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
    )
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

    if attention_mask is not None:
        causal_mask = causal_mask.clone()
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask, min_dtype
        )

    return causal_mask


class DualAttentionLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Use unique cache slots for the two attention modules inside one layer.
        self.streamwise_attn = LlamaAttention(config=config, layer_idx=2 * layer_idx)
        self.blockwise_attn = LlamaAttention(
            config=config, layer_idx=2 * layer_idx + 1
        )

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_blockwise_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def _run_attention(
        self,
        attn: LlamaAttention,
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
            outputs = attn(
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
            outputs = attn(
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
        streamwise_attention_mask: Optional[torch.Tensor] = None,
        blockwise_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, streamwise_attn_weights = self._run_attention(
            self.streamwise_attn,
            hidden_states=hidden_states,
            attention_mask=streamwise_attention_mask,
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
        hidden_states = self.pre_blockwise_layernorm(hidden_states)
        hidden_states, blockwise_attn_weights = self._run_attention(
            self.blockwise_attn,
            hidden_states=hidden_states,
            attention_mask=blockwise_attention_mask,
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
            return hidden_states, (streamwise_attn_weights, blockwise_attn_weights)

        return hidden_states


class LlamaBackbone(LlamaPreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                DualAttentionLlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def _resolve_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        custom_attention_mask: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
        cache_position: torch.LongTensor,
        past_key_values: Optional[Cache],
        position_ids: torch.LongTensor,
    ) -> Optional[torch.Tensor]:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        sequence_length = inputs_embeds.shape[1]

        if custom_attention_mask is None and attention_mask is None:
            target_length = past_seen_tokens + sequence_length
            return _prepare_4d_causal_attention_mask(
                attention_mask=None,
                sequence_length=sequence_length,
                target_length=target_length,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                cache_position=cache_position,
                batch_size=inputs_embeds.shape[0],
            )

        if custom_attention_mask is None:
            return _prepare_4d_causal_attention_mask(
                attention_mask=attention_mask,
                sequence_length=sequence_length,
                target_length=attention_mask.shape[-1],
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                cache_position=cache_position,
                batch_size=inputs_embeds.shape[0],
            )

        if custom_attention_mask.dim() == 2:
            return _prepare_4d_causal_attention_mask(
                attention_mask=custom_attention_mask,
                sequence_length=sequence_length,
                target_length=custom_attention_mask.shape[-1],
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                cache_position=cache_position,
                batch_size=inputs_embeds.shape[0],
            )

        if custom_attention_mask.dim() != 4:
            raise ValueError(
                "Custom attention masks must be 2D padding masks or 4D additive masks."
            )

        custom_attention_mask = _to_additive_mask(
            custom_attention_mask.to(device=inputs_embeds.device),
            inputs_embeds.dtype,
        )

        if attention_mask is None:
            return custom_attention_mask

        base_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            sequence_length=sequence_length,
            target_length=attention_mask.shape[-1],
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
            cache_position=cache_position,
            batch_size=inputs_embeds.shape[0],
        )
        return base_attention_mask + custom_attention_mask

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
        streamwise_attention_mask: Optional[torch.Tensor] = None,
        blockwise_attention_mask: Optional[torch.Tensor] = None,
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
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_seen_tokens
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        streamwise_attention_mask = self._resolve_attention_mask(
            attention_mask=attention_mask,
            custom_attention_mask=streamwise_attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        blockwise_attention_mask = self._resolve_attention_mask(
            attention_mask=attention_mask,
            custom_attention_mask=blockwise_attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        if position_embeddings is None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                streamwise_attention_mask=streamwise_attention_mask,
                blockwise_attention_mask=blockwise_attention_mask,
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
