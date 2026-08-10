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
        tts_token_id: int | None = None,
        asr_token_id: int | None = None,
        text_loss_weight: float = 1.0,
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
        if (tts_token_id is None) != (asr_token_id is None):
            raise ValueError("tts_token_id and asr_token_id must be provided together.")
        if text_loss_weight < 0:
            raise ValueError("text_loss_weight must be non-negative.")

        self.backbone = LlamaBackbone(llama_config)
        self.hidden_size = int(llama_config.hidden_size)
        self.num_discrete_tokens = int(num_discrete_tokens)
        self.discrete_vocab_size = int(discrete_vocab_size)
        self.discrete_regular_token_loss_weight = float(discrete_regular_token_loss_weight)
        self.discrete_special_token_loss_weight = float(discrete_special_token_loss_weight)
        self.tts_token_id = None if tts_token_id is None else int(tts_token_id)
        self.asr_token_id = None if asr_token_id is None else int(asr_token_id)
        self.text_loss_weight = float(text_loss_weight)

        self.discrete_frame_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.discrete_lm_head = nn.Linear(
            self.hidden_size,
            self.num_discrete_tokens * self.discrete_vocab_size,
            bias=False,
        )
        self.text_lm_head = (
            nn.Linear(self.hidden_size, int(llama_config.vocab_size), bias=False)
            if self.tts_token_id is not None
            else None
        )
        self.discrete_stream_embeddings = nn.Parameter(
            torch.empty(self.num_discrete_tokens, self.hidden_size)
        )

        vocab_size = int(self.backbone.config.vocab_size)
        if self.tts_token_id is not None:
            assert self.asr_token_id is not None
            if not 0 <= self.tts_token_id < vocab_size or not 0 <= self.asr_token_id < vocab_size:
                raise ValueError("Task token ids must be in [0, llama_config.vocab_size).")
            if self.tts_token_id == self.asr_token_id:
                raise ValueError("tts_token_id and asr_token_id must be distinct.")
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
        layers: tuple[nn.Linear, ...] = (self.discrete_frame_proj, self.discrete_lm_head)
        if self.text_lm_head is not None:
            layers = (*layers, self.text_lm_head)
        for layer in layers:
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

    def _text_logits(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        if self.text_lm_head is None:
            raise RuntimeError("Text logits require Stage-1 multitask task-token configuration.")
        return self.text_lm_head(hidden_states)

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

    def _compute_discrete_sample_losses(
        self,
        *,
        hidden_states: torch.FloatTensor,
        targets: torch.LongTensor,
        prediction_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        """Return one normalized discrete CE value per sample.

        Normalizing within a sample before reducing across the batch prevents a
        long TTS utterance from outweighing a short ASR transcript solely
        because it contains more target positions.
        """
        losses = hidden_states.new_zeros((hidden_states.shape[0],))
        active = prediction_mask.any(dim=1)
        for batch_idx in torch.nonzero(active, as_tuple=False).flatten().tolist():
            logits = self._discrete_logits(hidden_states[batch_idx, prediction_mask[batch_idx]])
            flat_targets = targets[batch_idx, prediction_mask[batch_idx]].reshape(-1)
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
            losses[batch_idx] = (per_token_loss * weights).sum() / weights.sum().clamp_min(1e-12)
        return losses, active

    def _compute_text_sample_losses(
        self,
        *,
        hidden_states: torch.FloatTensor,
        targets: torch.LongTensor,
        prediction_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        losses = hidden_states.new_zeros((hidden_states.shape[0],))
        active = prediction_mask.any(dim=1)
        for batch_idx in torch.nonzero(active, as_tuple=False).flatten().tolist():
            logits = self._text_logits(hidden_states[batch_idx, prediction_mask[batch_idx]])
            labels = targets[batch_idx, prediction_mask[batch_idx]]
            losses[batch_idx] = F.cross_entropy(logits, labels, reduction="mean")
        return losses, active

    def forward(
        self,
        *,
        flat_token_ids: torch.LongTensor,
        flat_discrete_values: torch.LongTensor,
        flat_token_type_ids: torch.LongTensor,
        flat_target_discrete_values: torch.LongTensor,
        flat_target_block_ids: torch.LongTensor,
        flat_target_text_values: Optional[torch.LongTensor] = None,
        flat_prediction_kind: Optional[torch.LongTensor] = None,
        flat_task_ids: Optional[torch.LongTensor] = None,
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
        typed_inputs = (
            flat_target_text_values is not None
            or flat_prediction_kind is not None
            or flat_task_ids is not None
        )
        if typed_inputs:
            if (
                flat_target_text_values is None
                or flat_prediction_kind is None
                or flat_task_ids is None
            ):
                raise ValueError(
                    "Multitask batches require flat_target_text_values, "
                    "flat_prediction_kind, and flat_task_ids together."
                )
            if self.text_lm_head is None or self.tts_token_id is None or self.asr_token_id is None:
                raise RuntimeError(
                    "Received multitask labels but the model has no configured TTS/ASR task tokens."
                )
            if tuple(flat_target_text_values.shape) != (batch_size, seq_len):
                raise ValueError("flat_target_text_values must have shape [batch, sequence].")
            if tuple(flat_prediction_kind.shape) != (batch_size, seq_len):
                raise ValueError("flat_prediction_kind must have shape [batch, sequence].")
            if tuple(flat_task_ids.shape) != (batch_size,):
                raise ValueError("flat_task_ids must have shape [batch].")
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
        if typed_inputs:
            assert flat_prediction_kind is not None
            assert flat_target_text_values is not None
            assert flat_task_ids is not None
            kinds = flat_prediction_kind.to(device=hidden_states.device, dtype=torch.long)
            task_ids = flat_task_ids.to(device=hidden_states.device, dtype=torch.long)
            allowed_kinds = {0, 1, 2}
            if not bool(torch.isin(kinds, torch.tensor(tuple(allowed_kinds), device=kinds.device)).all()):
                raise ValueError("flat_prediction_kind contains an unknown prediction kind.")
            if not bool(torch.isin(task_ids, torch.tensor((0, 1), device=task_ids.device)).all()):
                raise ValueError("flat_task_ids must contain only TTS=0 or ASR=1.")
            expected_task_tokens = torch.where(
                task_ids == 0,
                torch.full_like(task_ids, self.tts_token_id),
                torch.full_like(task_ids, self.asr_token_id),
            )
            if bool((flat_token_ids[:, 0].to(device=task_ids.device) != expected_task_tokens).any()):
                raise ValueError("The first token of each multitask sequence must be its task token.")
            discrete_mask = resolved_attention & (kinds == 1)
            text_mask = resolved_attention & (kinds == 2)
            tts_samples = task_ids == 0
            asr_samples = task_ids == 1
            if bool((discrete_mask & ~tts_samples.unsqueeze(1)).any()):
                raise ValueError("Only TTS samples may contain discrete prediction labels.")
            if bool((text_mask & ~asr_samples.unsqueeze(1)).any()):
                raise ValueError("Only ASR samples may contain text prediction labels.")
            if bool((flat_target_block_ids.to(device=kinds.device) >= 0).ne(discrete_mask).any()):
                raise ValueError(
                    "flat_target_block_ids must mark exactly the discrete prediction positions."
                )
            discrete_sample_losses, discrete_active = self._compute_discrete_sample_losses(
                hidden_states=hidden_states,
                targets=flat_target_discrete_values.to(dtype=torch.long),
                prediction_mask=discrete_mask,
            )
            text_sample_losses, text_active = self._compute_text_sample_losses(
                hidden_states=hidden_states,
                targets=flat_target_text_values.to(device=hidden_states.device, dtype=torch.long),
                prediction_mask=text_mask,
            )
            if bool((tts_samples & ~discrete_active).any()) or bool((asr_samples & ~text_active).any()):
                raise ValueError("Every multitask sample must contain its task's target labels.")
            per_sample_loss = torch.where(
                tts_samples,
                discrete_sample_losses,
                self.text_loss_weight * text_sample_losses,
            )
            total_loss = per_sample_loss.mean()
            tts_loss = (
                discrete_sample_losses[tts_samples].mean()
                if bool(tts_samples.any())
                else total_loss.new_zeros(())
            )
            asr_loss = (
                text_sample_losses[asr_samples].mean()
                if bool(asr_samples.any())
                else total_loss.new_zeros(())
            )
            discrete_loss = tts_loss
            text_loss = asr_loss
            tts_sample_count = tts_samples.sum()
            asr_sample_count = asr_samples.sum()
        else:
            prediction_mask = resolved_attention & (flat_target_block_ids >= 0)
            discrete_loss, _ = self._compute_loss(
                hidden_states=hidden_states,
                targets=flat_target_discrete_values.to(dtype=torch.long),
                prediction_mask=prediction_mask,
            )
            total_loss = discrete_loss
            tts_loss = discrete_loss
            text_loss = discrete_loss.new_zeros(())
            asr_loss = text_loss
            tts_sample_count = torch.tensor(batch_size, device=hidden_states.device)
            asr_sample_count = torch.tensor(0, device=hidden_states.device)
        if not return_dict:
            return total_loss, discrete_loss
        zero = discrete_loss.new_zeros(())
        return MU.PrismTTSOutput(
            loss=total_loss,
            discrete_loss=discrete_loss,
            continuous_loss=zero,
            flow_loss=zero,
            tts_loss=tts_loss,
            asr_loss=asr_loss,
            text_loss=text_loss,
            tts_sample_count=tts_sample_count,
            asr_sample_count=asr_sample_count,
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

    def _sample_text_id(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
    ) -> torch.LongTensor:
        """Sample only character ids and EOT from the full-vocabulary text head."""
        if self.tts_token_id is None or self.asr_token_id is None:
            raise RuntimeError("ASR sampling requires configured TTS/ASR task tokens.")
        # With the shared vocabulary layout, characters begin immediately after
        # PAD and task tokens are allocated after the character vocabulary.
        text_start = self.pad_token_id + 1
        text_end = min(self.tts_token_id, self.asr_token_id)
        if text_end <= text_start:
            raise RuntimeError("Task token ids leave no valid character-token range for ASR.")
        allowed = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
        allowed[text_start:text_end] = True
        allowed[self.eot_token_id] = True
        masked_logits = logits.masked_fill(~allowed, float("-inf"))
        return MU.sample_discrete_ids(
            logits=masked_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )

    def _build_asr_generation_prefix(
        self,
        *,
        discrete_speech: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        if self.asr_token_id is None:
            raise RuntimeError("ASR generation requires an asr_token_id.")
        speech_len = int(discrete_speech.shape[0])
        device = discrete_speech.device
        total = 1 + speech_len + 1
        token_ids = torch.full((1, total), self.pad_token_id, dtype=torch.long, device=device)
        discrete_values = torch.full(
            (1, total, self.num_discrete_tokens),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        token_types = torch.full((1, total), MU.TEXT_TOKEN_TYPE, dtype=torch.long, device=device)
        token_ids[0, 0] = self.asr_token_id
        if speech_len:
            token_types[0, 1 : 1 + speech_len] = MU.SPEECH_FRAME_TOKEN_TYPE
            discrete_values[0, 1 : 1 + speech_len] = discrete_speech
        token_ids[0, -1] = self.eos_token_id
        return token_ids, discrete_values, token_types

    @torch.no_grad()
    def _transcribe_one(
        self,
        *,
        discrete_speech: torch.LongTensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
    ) -> torch.LongTensor:
        token_ids, discrete_values, token_types = self._build_asr_generation_prefix(
            discrete_speech=discrete_speech
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
        generated: list[torch.LongTensor] = []
        for step_idx in range(max_new_tokens):
            sampled = self._sample_text_id(
                self._text_logits(hidden)[0],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            if int(sampled.item()) == self.eot_token_id:
                break
            generated.append(sampled)
            outputs = self._run_backbone(
                token_ids=sampled.view(1, 1),
                discrete_values=torch.full(
                    (1, 1, self.num_discrete_tokens),
                    self.pad_token_id,
                    dtype=torch.long,
                    device=hidden.device,
                ),
                token_type_ids=torch.full(
                    (1, 1), MU.TEXT_TOKEN_TYPE, dtype=torch.long, device=hidden.device
                ),
                attention_mask=None,
                past_key_values=cache,
                use_cache=True,
                position_offset=prefix_len + step_idx,
            )
            hidden = outputs.last_hidden_state[:, -1]
            cache = outputs.past_key_values
        if not generated:
            return torch.empty((0,), dtype=torch.long, device=token_ids.device)
        return torch.stack(generated)

    @torch.no_grad()
    def transcribe(
        self,
        *,
        discrete_speech: torch.LongTensor,
        speech_lengths: Optional[torch.Tensor | int] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = False,
        return_dict: bool = True,
    ) -> MU.PrismTTSGenerationOutput | torch.LongTensor:
        """Autoregressively transcribe discrete speech frames into text token ids.

        ``discrete_speech`` contains real codec frames only; do not append the
        all-stream terminal EOS frame used internally by the training collator.
        """
        if self.text_lm_head is None or self.asr_token_id is None:
            raise RuntimeError("transcribe() requires the multitask Stage-1 configuration.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0.")
        discrete_speech = MU.normalize_discrete_tokens(
            discrete_speech,
            "discrete_speech",
            num_discrete_tokens=self.num_discrete_tokens,
        )
        batch_size = int(discrete_speech.shape[0])
        speech_lengths = MU.normalize_lengths(
            speech_lengths,
            batch_size,
            int(discrete_speech.shape[1]),
            "speech_lengths",
            discrete_speech.device,
            default_value=int(discrete_speech.shape[1]),
        )
        results = [
            self._transcribe_one(
                discrete_speech=discrete_speech[index, : int(speech_lengths[index])],
                max_new_tokens=int(max_new_tokens),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            for index in range(batch_size)
        ]
        max_length = max((int(tokens.shape[0]) for tokens in results), default=0)
        output = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long,
            device=discrete_speech.device,
        )
        for index, tokens in enumerate(results):
            if tokens.numel():
                output[index, : tokens.shape[0]] = tokens
        if not return_dict:
            return output
        return MU.PrismTTSGenerationOutput(text_ids=output)

    def _build_generation_prefix(
        self,
        *,
        text_prompt: torch.LongTensor,
        discrete_prompt: torch.LongTensor,
        text_target: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        prompt_speech_len = int(discrete_prompt.shape[0])
        task_prefix_len = 1 if self.tts_token_id is not None else 0
        total = (
            task_prefix_len
            + int(text_prompt.shape[0])
            + 1
            + prompt_speech_len
            + 1
            + int(text_target.shape[0])
            + 1
        )
        device = text_prompt.device
        token_ids = torch.full((1, total), self.pad_token_id, dtype=torch.long, device=device)
        discrete_values = torch.full(
            (1, total, self.num_discrete_tokens), self.pad_token_id, dtype=torch.long, device=device
        )
        token_types = torch.full((1, total), MU.TEXT_TOKEN_TYPE, dtype=torch.long, device=device)
        cursor = 0
        if self.tts_token_id is not None:
            token_ids[0, cursor] = self.tts_token_id
            cursor += 1
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
