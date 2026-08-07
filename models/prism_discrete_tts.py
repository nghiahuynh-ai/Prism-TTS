from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig

from models.llama_backbone import LlamaBackbone
from utils import model_utils as MU


class PrismDiscreteTTS(nn.Module):
    """Causal, discrete-only speech-token model used by Prism-TTS stage 1.

    The class intentionally has no continuous-latent input or parameters.  It
    retains the project's fused *frame* representation for the discrete streams
    (one transformer position per codec frame), while target frames use the
    same one-frame shift as the former joint model.
    """

    stage = "discrete"

    def __init__(
        self,
        *,
        llama_config: LlamaConfig,
        num_discrete_tokens: int,
        discrete_vocab_size: int,
        discrete_regular_token_loss_weight: float = 1.0,
        discrete_special_token_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if num_discrete_tokens < 1:
            raise ValueError("num_discrete_tokens must be >= 1.")
        if discrete_vocab_size < 1 or discrete_vocab_size > llama_config.vocab_size:
            raise ValueError("discrete_vocab_size must be in [1, llama_config.vocab_size].")
        if discrete_regular_token_loss_weight < 0 or discrete_special_token_loss_weight < 0:
            raise ValueError("Discrete loss weights must be non-negative.")
        if discrete_regular_token_loss_weight == 0 and discrete_special_token_loss_weight == 0:
            raise ValueError("At least one discrete loss weight must be positive.")

        self.backbone = LlamaBackbone(llama_config)
        self.hidden_size = int(llama_config.hidden_size)
        self.num_discrete_tokens = int(num_discrete_tokens)
        self.discrete_vocab_size = int(discrete_vocab_size)
        self.discrete_regular_token_loss_weight = float(discrete_regular_token_loss_weight)
        self.discrete_special_token_loss_weight = float(discrete_special_token_loss_weight)

        self.discrete_frame_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.discrete_lm_head = nn.Linear(
            self.hidden_size,
            self.num_discrete_tokens * self.discrete_vocab_size,
            bias=False,
        )
        self.discrete_stream_embeddings = nn.Parameter(
            torch.empty(self.num_discrete_tokens, self.hidden_size)
        )

        vocab_size = int(self.backbone.config.vocab_size)
        self.pad_token_id = int(self.backbone.config.pad_token_id or 0)
        if not 0 <= self.pad_token_id < vocab_size:
            self.pad_token_id = 0
        self.eos_token_id = int(self.backbone.config.eos_token_id or 0)
        if not 0 <= self.eos_token_id < vocab_size:
            self.eos_token_id = 0
        self.eot_token_id = max(0, self.eos_token_id - 1)
        self.training_special_discrete_token_ids = MU.infer_special_discrete_token_ids(
            MU.resolve_generation_discrete_eos_token_id(
                None,
                backbone_eos_token_id=self.backbone.config.eos_token_id,
                discrete_vocab_size=self.discrete_vocab_size,
            ),
            backbone_eos_token_id=self.backbone.config.eos_token_id,
            backbone_pad_token_id=self.backbone.config.pad_token_id,
            discrete_vocab_size=self.discrete_vocab_size,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = float(getattr(self.backbone.config, "initializer_range", 0.02))
        for layer in (self.discrete_frame_proj, self.discrete_lm_head):
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.normal_(self.discrete_stream_embeddings, mean=0.0, std=std)

    @property
    def text_embedding(self) -> nn.Embedding:
        return self.backbone.embed_tokens

    @property
    def discrete_embedding(self) -> nn.Embedding:
        return self.backbone.embed_tokens

    def _build_inputs_embeds(
        self,
        *,
        token_ids: torch.LongTensor,
        discrete_values: torch.LongTensor,
        token_type_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        is_frame = token_type_ids == MU.SPEECH_FRAME_TOKEN_TYPE
        text_embeds = self.text_embedding(token_ids)
        safe_discrete = discrete_values.clamp(min=0, max=self.backbone.config.vocab_size - 1)
        per_stream = self.discrete_embedding(safe_discrete)
        per_stream = per_stream + self.discrete_stream_embeddings.view(
            1, 1, self.num_discrete_tokens, self.hidden_size
        )
        frames = per_stream.sum(dim=2) / math.sqrt(float(self.num_discrete_tokens))
        frames = self.discrete_frame_proj(frames)
        return torch.where(is_frame.unsqueeze(-1), frames, text_embeds)

    def _encode(
        self,
        *,
        token_ids: torch.LongTensor,
        discrete_values: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        inputs_embeds = self._build_inputs_embeds(
            token_ids=token_ids,
            discrete_values=discrete_values,
            token_type_ids=token_type_ids,
        )
        batch_size, seq_len, _ = inputs_embeds.shape
        causal_mask = MU.build_causal_attention_mask(attention_mask, dtype=inputs_embeds.dtype)
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.backbone.rotary_emb(inputs_embeds, position_ids=position_ids)
        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            return_dict=True,
        ).last_hidden_state

    def _run_backbone(
        self,
        *,
        token_ids: torch.LongTensor,
        discrete_values: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Any = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> Any:
        inputs_embeds = self._build_inputs_embeds(
            token_ids=token_ids,
            discrete_values=discrete_values,
            token_type_ids=token_type_ids,
        )
        batch_size, seq_len, _ = inputs_embeds.shape
        position_ids = torch.arange(
            position_offset, position_offset + seq_len, device=inputs_embeds.device
        ).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.backbone.rotary_emb(inputs_embeds, position_ids=position_ids)
        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )

    def _discrete_logits(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        logits = self.discrete_lm_head(hidden_states)
        return logits.view(*logits.shape[:-1], self.num_discrete_tokens, self.discrete_vocab_size)

    def _compute_loss(
        self,
        *,
        hidden_states: torch.FloatTensor,
        targets: torch.LongTensor,
        prediction_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.FloatTensor]:
        if not prediction_mask.any():
            return hidden_states.new_zeros(()), hidden_states.new_zeros(
                (0, self.num_discrete_tokens, self.discrete_vocab_size)
            )
        logits = self._discrete_logits(hidden_states[prediction_mask])
        flat_targets = targets[prediction_mask].reshape(-1)
        per_token_loss = F.cross_entropy(logits.flatten(0, 1), flat_targets, reduction="none")
        weights = torch.full_like(per_token_loss, self.discrete_regular_token_loss_weight)
        if self.training_special_discrete_token_ids:
            special_ids = torch.tensor(
                self.training_special_discrete_token_ids,
                dtype=flat_targets.dtype,
                device=flat_targets.device,
            )
            weights = torch.where(
                torch.isin(flat_targets, special_ids),
                weights.new_full(weights.shape, self.discrete_special_token_loss_weight),
                weights,
            )
        return (per_token_loss * weights).sum() / weights.sum().clamp_min(1e-12), logits

    def forward(
        self,
        *,
        flat_token_ids: torch.LongTensor,
        flat_discrete_values: torch.LongTensor,
        flat_token_type_ids: torch.LongTensor,
        flat_target_discrete_values: torch.LongTensor,
        flat_target_block_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> MU.PrismTTSOutput | tuple[torch.Tensor, torch.Tensor]:
        if flat_token_ids.dim() != 2:
            raise ValueError("flat_token_ids must have shape [batch, sequence].")
        batch_size, seq_len = flat_token_ids.shape
        expected_discrete_shape = (batch_size, seq_len, self.num_discrete_tokens)
        for name, tensor in (
            ("flat_discrete_values", flat_discrete_values),
            ("flat_target_discrete_values", flat_target_discrete_values),
        ):
            if tuple(tensor.shape) != expected_discrete_shape:
                raise ValueError(f"{name} must have shape {expected_discrete_shape}.")
        for name, tensor in (
            ("flat_token_type_ids", flat_token_type_ids),
            ("flat_target_block_ids", flat_target_block_ids),
        ):
            if tuple(tensor.shape) != (batch_size, seq_len):
                raise ValueError(f"{name} must have shape [batch, sequence].")
        if attention_mask is None:
            resolved_attention = torch.ones_like(flat_token_ids, dtype=torch.bool)
        else:
            if tuple(attention_mask.shape) != (batch_size, seq_len):
                raise ValueError("attention_mask must have shape [batch, sequence].")
            resolved_attention = attention_mask.to(dtype=torch.bool, device=flat_token_ids.device)

        hidden_states = self._encode(
            token_ids=flat_token_ids.to(dtype=torch.long),
            discrete_values=flat_discrete_values.to(dtype=torch.long),
            token_type_ids=flat_token_type_ids.to(dtype=torch.long),
            attention_mask=resolved_attention,
        )
        prediction_mask = resolved_attention & (flat_target_block_ids >= 0)
        discrete_loss, _ = self._compute_loss(
            hidden_states=hidden_states,
            targets=flat_target_discrete_values.to(dtype=torch.long),
            prediction_mask=prediction_mask,
        )
        if not return_dict:
            return discrete_loss, discrete_loss
        zero = discrete_loss.new_zeros(())
        return MU.PrismTTSOutput(
            loss=discrete_loss,
            discrete_loss=discrete_loss,
            continuous_loss=zero,
            flow_loss=zero,
        )

    def _sample_discrete_ids(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
    ) -> torch.LongTensor:
        return MU.sample_discrete_ids(
            logits=logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )

    def _build_generation_prefix(
        self,
        *,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        text_target: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        prompt_speech_len = int(discrete_prompt.shape[0])
        total = int(text_prompt.shape[0]) + 1 + prompt_speech_len + 1 + int(text_target.shape[0]) + 1
        device = text_prompt.device
        token_ids = torch.full((1, total), self.pad_token_id, dtype=torch.long, device=device)
        discrete_values = torch.full(
            (1, total, self.num_discrete_tokens), self.pad_token_id, dtype=torch.long, device=device
        )
        token_types = torch.full((1, total), MU.TEXT_TOKEN_TYPE, dtype=torch.long, device=device)
        cursor = 0
        token_ids[0, cursor : cursor + text_prompt.shape[0]] = text_prompt
        cursor += text_prompt.shape[0]
        token_ids[0, cursor] = self.eot_token_id
        cursor += 1
        if prompt_speech_len:
            token_types[0, cursor : cursor + prompt_speech_len] = MU.SPEECH_FRAME_TOKEN_TYPE
            discrete_values[0, cursor : cursor + prompt_speech_len] = discrete_prompt
            cursor += prompt_speech_len
        token_ids[0, cursor] = self.eos_token_id
        cursor += 1
        token_ids[0, cursor : cursor + text_target.shape[0]] = text_target
        cursor += text_target.shape[0]
        token_ids[0, cursor] = self.eot_token_id
        return token_ids, discrete_values, token_types

    @torch.no_grad()
    def _generate_one(
        self,
        *,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        text_target: torch.LongTensor,
        max_new_blocks: int,
        discrete_eos_id: int,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
    ) -> tuple[torch.LongTensor, list[torch.Tensor]]:
        token_ids, discrete_values, token_types = self._build_generation_prefix(
            text_prompt=text_prompt,
            discrete_prompt=discrete_prompt,
            text_target=text_target,
        )
        prefix_len = int(token_ids.shape[1])
        causal_mask = MU.build_causal_attention_mask(
            torch.ones((1, prefix_len), dtype=torch.bool, device=token_ids.device),
            dtype=self.text_embedding.weight.dtype,
        )
        outputs = self._run_backbone(
            token_ids=token_ids,
            discrete_values=discrete_values,
            token_type_ids=token_types,
            attention_mask=causal_mask,
            use_cache=True,
        )
        hidden = outputs.last_hidden_state[:, -1]
        cache = outputs.past_key_values
        generated: list[torch.Tensor] = []
        logits_by_step: list[torch.Tensor] = []
        for step_idx in range(max_new_blocks):
            logits = self._discrete_logits(hidden)[0]
            sampled = self._sample_discrete_ids(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            generated.append(sampled)
            logits_by_step.append(logits)
            if bool(torch.eq(sampled, int(discrete_eos_id)).all().item()):
                break
            outputs = self._run_backbone(
                token_ids=torch.full((1, 1), self.pad_token_id, dtype=torch.long, device=hidden.device),
                discrete_values=sampled.view(1, 1, self.num_discrete_tokens),
                token_type_ids=torch.full(
                    (1, 1), MU.SPEECH_FRAME_TOKEN_TYPE, dtype=torch.long, device=hidden.device
                ),
                attention_mask=None,
                past_key_values=cache,
                use_cache=True,
                position_offset=prefix_len + step_idx,
            )
            hidden = outputs.last_hidden_state[:, -1]
            cache = outputs.past_key_values
        if not generated:
            return torch.empty((0, self.num_discrete_tokens), dtype=torch.long, device=token_ids.device), logits_by_step
        return torch.stack(generated), logits_by_step

    @torch.no_grad()
    def generate(
        self,
        *,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        text_target: Optional[torch.LongTensor] = None,
        text_prompt_lengths: Optional[torch.Tensor | int] = None,
        speech_prompt_lengths: Optional[torch.Tensor | int] = None,
        text_target_lengths: Optional[torch.Tensor | int] = None,
        max_new_blocks: int = 128,
        discrete_eos_token_id: Optional[int] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        return_dict: bool = True,
    ) -> MU.PrismTTSGenerationOutput | torch.LongTensor:
        if max_new_blocks < 0:
            raise ValueError("max_new_blocks must be >= 0.")
        text_prompt = MU.normalize_text_tokens(text_prompt, "text_prompt")
        discrete_prompt = MU.normalize_discrete_tokens(
            discrete_prompt, "discrete_prompt", num_discrete_tokens=self.num_discrete_tokens
        )
        batch_size = int(text_prompt.shape[0])
        if text_target is None:
            text_target = text_prompt.new_empty((batch_size, 0))
        else:
            text_target = MU.normalize_text_tokens(text_target, "text_target")
        if text_target.shape[0] != batch_size:
            raise ValueError("text_target batch size must match text_prompt.")
        text_prompt_lengths = MU.normalize_lengths(
            text_prompt_lengths, batch_size, text_prompt.shape[1], "text_prompt_lengths", text_prompt.device,
            default_value=text_prompt.shape[1],
        )
        speech_prompt_lengths = MU.normalize_lengths(
            speech_prompt_lengths, batch_size, discrete_prompt.shape[1], "speech_prompt_lengths", text_prompt.device,
            default_value=discrete_prompt.shape[1],
        )
        text_target_lengths = MU.normalize_lengths(
            text_target_lengths, batch_size, text_target.shape[1], "text_target_lengths", text_prompt.device,
            default_value=text_target.shape[1],
        )
        eos_id = MU.resolve_generation_discrete_eos_token_id(
            discrete_eos_token_id,
            backbone_eos_token_id=self.backbone.config.eos_token_id,
            discrete_vocab_size=self.discrete_vocab_size,
        )
        results: list[tuple[torch.LongTensor, list[torch.Tensor]]] = []
        for index in range(batch_size):
            results.append(self._generate_one(
                text_prompt=text_prompt[index, : int(text_prompt_lengths[index])],
                discrete_prompt=discrete_prompt[index, : int(speech_prompt_lengths[index])],
                text_target=text_target[index, : int(text_target_lengths[index])],
                max_new_blocks=int(max_new_blocks),
                discrete_eos_id=eos_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            ))
        max_len = max((int(item[0].shape[0]) for item in results), default=0)
        output = torch.full(
            (batch_size, max_len, self.num_discrete_tokens), self.pad_token_id,
            dtype=torch.long, device=text_prompt.device,
        )
        logits_by_step: list[torch.Tensor] = []
        for batch_idx, (tokens, logits_list) in enumerate(results):
            if tokens.numel():
                output[batch_idx, : tokens.shape[0]] = tokens
            for step_idx, logits in enumerate(logits_list):
                while len(logits_by_step) <= step_idx:
                    logits_by_step.append(torch.full(
                        (batch_size, self.num_discrete_tokens, self.discrete_vocab_size),
                        float("-inf"), dtype=logits.dtype, device=logits.device,
                    ))
                logits_by_step[step_idx][batch_idx] = logits
        if not return_dict:
            return output.transpose(1, 2).contiguous()
        return MU.PrismTTSGenerationOutput(
            text_ids=text_target,
            discrete_ids=output.transpose(1, 2).contiguous(),
            discrete_logits=tuple(logits_by_step),
        )
