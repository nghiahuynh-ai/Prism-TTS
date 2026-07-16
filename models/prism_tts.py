from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig

from models.flow_head import FlowHead
from models.llama_backbone import LlamaBackbone, WindowedCausalEncoder
from utils import model_utils as MU


class PrismTTS(nn.Module):
    """
    Prism-TTS autoregressive continuous-latent model (no vector quantization).

    Speech is modeled as one continuous latent per frame. A causal Llama backbone
    produces a contextual embedding for every position; a small per-frame head
    (``FlowHead``, an AdaLN MLP) generates the next continuous frame, trained with
    continuous-time consistency modeling (sCM / TrigFlow). A binary EOS head
    predicts end-of-sequence.

    Sequence layout (per sample), fully causal:
    text_prompt -> EOT -> [prompt frames] -> EOS -> text_target -> EOT -> [target frames]

    Teacher-forcing shift: the hidden state at the position preceding target frame
    ``i`` conditions the head that generates frame ``i``.
    """

    def __init__(
        self,
        llama_config: LlamaConfig,
        continuous_latent_size: int,
        flow_num_res_blocks: int = 4,
        flow_model_channels: Optional[int] = None,
        eos_loss_weight: float = 1.0,
        sigma_data: float = 1.0,
        p_mean: float = -1.0,
        p_std: float = 1.6,
        tangent_warmup_steps: int = 1000,
        tangent_norm_const: float = 0.1,
        sample_steps: int = 1,
        eos_threshold: float = 0.5,
        latent_stats_decay: float = 0.999,
        inject_backbone_noise: bool = True,
        use_short_context: bool = True,
        short_context_layers: int = 2,
        short_context_window: int = 10,
    ):
        """Initialize backbone, per-frame consistency head, EOS head, and latent stats."""
        super().__init__()
        if continuous_latent_size < 1:
            raise ValueError("continuous_latent_size must be at least 1.")
        if eos_loss_weight < 0.0:
            raise ValueError("eos_loss_weight must be >= 0.")
        if sigma_data <= 0.0:
            raise ValueError("sigma_data must be > 0.")
        if p_std <= 0.0:
            raise ValueError("p_std must be > 0.")
        if tangent_warmup_steps < 0:
            raise ValueError("tangent_warmup_steps must be >= 0.")
        if tangent_norm_const <= 0.0:
            raise ValueError("tangent_norm_const must be > 0.")
        if sample_steps < 1:
            raise ValueError("sample_steps must be at least 1.")
        if not (0.0 < latent_stats_decay < 1.0):
            raise ValueError("latent_stats_decay must be in (0, 1).")

        self.hidden_size = int(llama_config.hidden_size)
        self.continuous_latent_size = int(continuous_latent_size)

        self.eos_loss_weight = float(eos_loss_weight)
        self.sigma_data = float(sigma_data)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.tangent_warmup_steps = int(tangent_warmup_steps)
        self.tangent_norm_const = float(tangent_norm_const)
        self.sample_steps = int(sample_steps)
        self.eos_threshold = float(eos_threshold)
        self.latent_stats_decay = float(latent_stats_decay)
        self.inject_backbone_noise = bool(inject_backbone_noise)
        self.use_short_context = bool(use_short_context)
        self.short_context_window = int(short_context_window)

        self.backbone = LlamaBackbone(llama_config)
        self.continuous_proj = nn.Linear(self.continuous_latent_size, self.hidden_size)
        self.eos_head = nn.Linear(self.hidden_size, 1)

        # CALM-style short-context branch: a small sliding-window causal transformer
        # over clean local latents; its output is added to the long-context backbone
        # hidden state to form the head conditioning Z = z_long + z_short.
        self.short_encoder = (
            WindowedCausalEncoder(
                llama_config,
                num_layers=int(short_context_layers),
                window=int(short_context_window),
            )
            if self.use_short_context
            else None
        )

        # Consistency head F_theta(x_t / sigma_data, t; cond) conditioned on the
        # backbone hidden state directly.
        self.flow_head = FlowHead(
            in_channels=self.continuous_latent_size,
            model_channels=flow_model_channels or self.hidden_size,
            out_channels=self.continuous_latent_size,
            z_channels=self.hidden_size,
            num_res_blocks=flow_num_res_blocks,
        )

        # Learned text/speech type embeddings added to token/latent embeddings.
        self.type_embeddings = nn.Parameter(torch.empty(2, self.hidden_size))

        vocab_size = int(self.backbone.config.vocab_size)
        pad_candidate = self.backbone.config.pad_token_id
        self.pad_token_id = 0
        if pad_candidate is not None and 0 <= int(pad_candidate) < vocab_size:
            self.pad_token_id = int(pad_candidate)

        eos_candidate = self.backbone.config.eos_token_id
        self.eos_token_id = 0
        if eos_candidate is not None and 0 <= int(eos_candidate) < vocab_size:
            self.eos_token_id = int(eos_candidate)
        # Shared layout: EOT = EOS - 1, text tokens begin at PAD + 1.
        self.eot_token_id = max(0, self.eos_token_id - 1)
        self.text_token_offset = self.pad_token_id + 1

        # Running latent normalization statistics ("center and normalize").
        self.register_buffer("emb_mean", torch.zeros(self.continuous_latent_size))
        self.register_buffer("emb_std", torch.ones(self.continuous_latent_size))
        self.register_buffer("stats_initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("train_step", torch.zeros((), dtype=torch.long))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize trainable parameters using the configured initializer range."""
        std = getattr(self.backbone.config, "initializer_range", 0.02)
        nn.init.normal_(self.continuous_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.type_embeddings, mean=0.0, std=std)
        nn.init.normal_(self.eos_head.weight, mean=0.0, std=std)
        if self.continuous_proj.bias is not None:
            nn.init.zeros_(self.continuous_proj.bias)
        if self.eos_head.bias is not None:
            nn.init.zeros_(self.eos_head.bias)

    # ------------------------------------------------------------------
    # Latent normalization
    # ------------------------------------------------------------------
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.emb_mean.to(dtype=x.dtype)
        std = self.emb_std.to(dtype=x.dtype).clamp_min(1e-4)
        return (x - mean) / std

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.emb_mean.to(dtype=x.dtype)
        std = self.emb_std.to(dtype=x.dtype).clamp_min(1e-4)
        return x * std + mean

    @torch.no_grad()
    def _update_latent_stats(self, speech_latents: torch.Tensor) -> None:
        if speech_latents.numel() == 0:
            return
        sp = speech_latents.detach().float()
        batch_mean = sp.mean(dim=0)
        batch_std = (sp.var(dim=0, unbiased=False) + 1e-8).sqrt()
        if bool(self.stats_initialized.item()):
            d = self.latent_stats_decay
            self.emb_mean.mul_(d).add_(batch_mean, alpha=1.0 - d)
            self.emb_std.mul_(d).add_(batch_std, alpha=1.0 - d)
        else:
            self.emb_mean.copy_(batch_mean)
            self.emb_std.copy_(batch_std)
            self.stats_initialized.fill_(True)

    # ------------------------------------------------------------------
    # Backbone encoding
    # ------------------------------------------------------------------
    def _embed(
        self,
        token_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        continuous_norm: torch.FloatTensor,
    ) -> torch.FloatTensor:
        vocab_size = int(self.backbone.config.vocab_size)
        token_ids_safe = token_ids.clamp(min=0, max=vocab_size - 1)
        text_embeds = self.backbone.embed_tokens(token_ids_safe)
        speech_embeds = self.continuous_proj(continuous_norm.to(dtype=text_embeds.dtype))
        is_speech = (token_type_ids == MU.SPEECH_TOKEN_TYPE).unsqueeze(-1)
        embeds = torch.where(is_speech, speech_embeds, text_embeds)
        type_embeds = self.type_embeddings[token_type_ids.clamp(min=0, max=1)]
        return embeds + type_embeds

    def _encode_hidden(
        self,
        token_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        continuous_norm: torch.FloatTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        inputs_embeds = self._embed(token_ids, token_type_ids, continuous_norm)
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def _encode_conditioning(
        self,
        token_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        long_continuous: torch.FloatTensor,
        short_continuous: torch.FloatTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """Per-position head conditioning Z = z_long + z_short.

        The long-context backbone sees `long_continuous` (noise-injected during
        training); the short-context branch sees `short_continuous` (always clean).
        """
        z = self._encode_hidden(token_ids, token_type_ids, long_continuous, attention_mask)
        if self.short_encoder is not None:
            short_embeds = self._embed(token_ids, token_type_ids, short_continuous)
            z = z + self.short_encoder(short_embeds, attention_mask)
        return z

    # ------------------------------------------------------------------
    # Consistency (sCM / TrigFlow) head
    # ------------------------------------------------------------------
    def _consistency_loss(
        self,
        x0: torch.FloatTensor,
        cond: torch.FloatTensor,
    ) -> torch.Tensor:
        """Continuous-time consistency (sCM / TrigFlow) loss on normalized frames.

        Follows Lu & Song (2024) TrigFlow with the paper (``cos``) tangent
        parameterization and adaptive uncertainty weighting.
        """
        device = x0.device
        dtype = torch.float32
        x0 = x0.to(dtype)
        cond = cond.to(dtype)
        num = x0.shape[0]
        sigma_data = self.sigma_data

        autocast_device = device.type if device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=autocast_device, enabled=False):
            # Sample t from a logit-normal proposal over noise level sigma.
            tau = torch.randn(num, 1, device=device, dtype=dtype) * self.p_std + self.p_mean
            sigma = tau.exp()
            t = torch.arctan(sigma / sigma_data)
            cos_t = torch.cos(t)
            sin_t = torch.sin(t)

            z = torch.randn_like(x0) * sigma_data
            x_t = cos_t * x0 + sin_t * z
            # dx_t/dt along the (conditional) probability-flow trajectory.
            dxt_dt = cos_t * z - sin_t * x0

            # JVP tangent vectors for primals (x_t / sigma_data, t). The cos*sin
            # factors from the sCM rearrangement are folded into the tangents.
            v_x = (cos_t * sin_t * dxt_dt) / sigma_data
            v_t = cos_t * sin_t

            def fwd(scaled_x: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
                return self.flow_head(scaled_x, t_in.reshape(-1), cond)

            f_theta, f_theta_grad = torch.func.jvp(
                fwd,
                (x_t / sigma_data, t),
                (v_x, v_t),
            )
            f_theta_grad = f_theta_grad.detach()
            f_theta_minus = f_theta.detach()

            warmup = self.tangent_warmup_steps
            r = 1.0 if warmup <= 0 else min(1.0, float(self.train_step.item()) / float(warmup))
            g = -(cos_t * cos_t) * (sigma_data * f_theta_minus - dxt_dt)
            g = g - r * (cos_t * sin_t * x_t + sigma_data * f_theta_grad)

            # Tangent normalization for stability (vector norm over feature dim).
            g_norm = torch.norm(g, dim=-1, keepdim=True)
            g = g / (g_norm + self.tangent_norm_const)

            logvar = self.flow_head.logvar(t)
            weight = 1.0 / sigma
            squared = (f_theta - f_theta_minus - g) ** 2
            loss = (weight / logvar.exp()) * squared + logvar
            return loss.mean()

    @torch.no_grad()
    def sample_frame(
        self,
        cond: torch.FloatTensor,
        num_steps: Optional[int] = None,
    ) -> torch.FloatTensor:
        """Sample normalized continuous frames from the consistency head.

        1-step (default) uses the TrigFlow boundary at t=pi/2; multi-step uses the
        deterministic denoise/renoise consistency sampler.
        """
        num_steps = self.sample_steps if num_steps is None else int(num_steps)
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")
        sigma_data = self.sigma_data
        dtype = cond.dtype
        num = cond.shape[0]
        x = torch.randn(num, self.continuous_latent_size, device=cond.device, dtype=dtype) * sigma_data
        ts = torch.linspace(math.pi / 2.0, 0.0, num_steps + 1, device=cond.device, dtype=dtype)

        for step_idx in range(num_steps):
            t_cur = ts[step_idx]
            t_vec = torch.full((num,), float(t_cur), device=cond.device, dtype=dtype)
            f_theta = self.flow_head(x / sigma_data, t_vec, cond)
            x0_pred = torch.cos(t_cur) * x - torch.sin(t_cur) * sigma_data * f_theta
            if step_idx == num_steps - 1:
                x = x0_pred
            else:
                t_next = ts[step_idx + 1]
                noise = torch.randn_like(x)
                x = torch.cos(t_next) * x0_pred + torch.sin(t_next) * sigma_data * noise
        return x

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        flat_token_ids: torch.LongTensor,
        flat_continuous_values: torch.FloatTensor,
        flat_token_type_ids: torch.LongTensor,
        flat_target_block_ids: torch.LongTensor,
        flat_target_block_counts: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> MU.PrismTTSOutput | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run autoregressive next-frame consistency training from collate tensors."""
        flat = MU.build_flat_batch_from_collate(
            flat_token_ids=flat_token_ids,
            flat_continuous_values=flat_continuous_values,
            flat_token_type_ids=flat_token_type_ids,
            flat_target_block_ids=flat_target_block_ids,
            flat_target_block_counts=flat_target_block_counts,
            attention_mask=attention_mask,
            continuous_latent_size=self.continuous_latent_size,
        )

        speech_mask = (flat.token_type_ids == MU.SPEECH_TOKEN_TYPE) & flat.attention_mask
        if self.training:
            self._update_latent_stats(flat.continuous_values[speech_mask])

        # Clean normalized latents supply the per-frame head targets.
        continuous_norm = self._normalize(flat.continuous_values)

        # The backbone consumes a noise-injected copy of the context during training
        # (exposure-bias regularization); non-speech positions are left untouched.
        backbone_continuous = continuous_norm
        if self.training and self.inject_backbone_noise:
            noised = MU.inject_continuous_backbone_noise(continuous_norm)
            backbone_continuous = torch.where(
                speech_mask.unsqueeze(-1), noised, continuous_norm
            )

        hidden_states = self._encode_conditioning(
            token_ids=flat.token_ids,
            token_type_ids=flat.token_type_ids,
            long_continuous=backbone_continuous,
            short_continuous=continuous_norm,
            attention_mask=flat.attention_mask,
        )

        # Teacher-forcing shift: conditioning for target frame at position q is the
        # hidden state at position q-1.
        hidden_prev = torch.cat(
            [torch.zeros_like(hidden_states[:, :1]), hidden_states[:, :-1]],
            dim=1,
        )

        target_mask = (flat.target_block_ids >= 0) & flat.attention_mask
        counts = flat.target_block_counts.to(device=target_mask.device)
        last_frame_id = (counts[:, None] - 1).expand_as(flat.target_block_ids)
        eos_target_full = flat.target_block_ids == last_frame_id

        cond = hidden_prev[target_mask]
        target_norm = continuous_norm[target_mask]
        eos_target = eos_target_full[target_mask].to(dtype=hidden_states.dtype)

        if cond.shape[0] == 0:
            zero = hidden_states.sum() * 0.0
            consistency_loss = zero
            eos_loss = zero
        else:
            consistency_loss = self._consistency_loss(target_norm, cond)
            eos_logit = self.eos_head(cond).squeeze(-1)
            eos_loss = F.binary_cross_entropy_with_logits(eos_logit, eos_target)

        loss = consistency_loss + self.eos_loss_weight * eos_loss

        if self.training:
            self.train_step += 1

        if not return_dict:
            return loss, consistency_loss, eos_loss
        return MU.PrismTTSOutput(
            loss=loss,
            consistency_loss=consistency_loss,
            eos_loss=eos_loss,
        )

    # ------------------------------------------------------------------
    # Autoregressive generation (full-recompute)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        *,
        text_prompt: torch.LongTensor,
        continuous_prompt: torch.FloatTensor,
        text_target: Optional[torch.LongTensor] = None,
        text_prompt_lengths: Optional[torch.Tensor | int] = None,
        speech_prompt_lengths: Optional[torch.Tensor | int] = None,
        text_target_lengths: Optional[torch.Tensor | int] = None,
        max_new_frames: int = 1024,
        eos_threshold: Optional[float] = None,
        num_sample_steps: Optional[int] = None,
        return_dict: bool = True,
    ) -> MU.PrismTTSGenerationOutput | torch.FloatTensor:
        """Autoregressively generate target continuous frames (full-recompute AR)."""
        text_prompt = MU.normalize_text_tokens(text_prompt, "text_prompt")
        batch_size = int(text_prompt.shape[0])
        device = text_prompt.device
        dtype = self.continuous_proj.weight.dtype

        continuous_prompt = MU.normalize_continuous_latents(
            continuous_prompt,
            expected_len=int(continuous_prompt.shape[1]),
            name="continuous_prompt",
            continuous_latent_size=self.continuous_latent_size,
        )
        if text_target is None:
            text_target = text_prompt.new_zeros((batch_size, 0))
        else:
            text_target = MU.normalize_text_tokens(text_target, "text_target")

        eos_threshold = self.eos_threshold if eos_threshold is None else float(eos_threshold)
        num_sample_steps = self.sample_steps if num_sample_steps is None else int(num_sample_steps)
        max_new_frames = int(max_new_frames)
        if max_new_frames < 0:
            raise ValueError("max_new_frames must be >= 0.")

        text_prompt_lengths = MU.normalize_lengths(
            text_prompt_lengths, batch_size, int(text_prompt.shape[1]),
            "text_prompt_lengths", device, default_value=int(text_prompt.shape[1]),
        )
        speech_prompt_lengths = MU.normalize_lengths(
            speech_prompt_lengths, batch_size, int(continuous_prompt.shape[1]),
            "speech_prompt_lengths", device, default_value=int(continuous_prompt.shape[1]),
        )
        text_target_lengths = MU.normalize_lengths(
            text_target_lengths, batch_size, int(text_target.shape[1]),
            "text_target_lengths", device, default_value=int(text_target.shape[1]),
        )
        max_text_target = int(text_target_lengths.max().item()) if batch_size > 0 else 0
        text_ids_out = text_target[:, :max_text_target]

        continuous_prompt_norm = self._normalize(continuous_prompt.to(dtype=dtype))

        prefix_lengths = (
            text_prompt_lengths + 1 + speech_prompt_lengths + 1 + text_target_lengths + 1
        )
        max_prefix = int(prefix_lengths.max().item()) if batch_size > 0 else 0
        total = max_prefix + max_new_frames

        token_ids_buf = torch.full((batch_size, total), self.pad_token_id, dtype=torch.long, device=device)
        token_type_buf = torch.full((batch_size, total), MU.TEXT_TOKEN_TYPE, dtype=torch.long, device=device)
        continuous_buf = torch.zeros((batch_size, total, self.continuous_latent_size), dtype=dtype, device=device)
        attention_buf = torch.zeros((batch_size, total), dtype=torch.bool, device=device)

        for sample_idx in range(batch_size):
            l1 = int(text_prompt_lengths[sample_idx].item())
            l2 = int(speech_prompt_lengths[sample_idx].item())
            l3 = int(text_target_lengths[sample_idx].item())
            pos = 0
            if l1 > 0:
                token_ids_buf[sample_idx, pos:pos + l1] = text_prompt[sample_idx, :l1]
            pos += l1
            token_ids_buf[sample_idx, pos] = self.eot_token_id
            pos += 1
            if l2 > 0:
                token_type_buf[sample_idx, pos:pos + l2] = MU.SPEECH_TOKEN_TYPE
                continuous_buf[sample_idx, pos:pos + l2, :] = continuous_prompt_norm[sample_idx, :l2, :]
            pos += l2
            token_ids_buf[sample_idx, pos] = self.eos_token_id
            pos += 1
            if l3 > 0:
                token_ids_buf[sample_idx, pos:pos + l3] = text_target[sample_idx, :l3]
            pos += l3
            token_ids_buf[sample_idx, pos] = self.eot_token_id
            pos += 1
            attention_buf[sample_idx, :pos] = True

        gen_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        eos_scores_buf = torch.zeros((batch_size, max_new_frames), dtype=torch.float32, device=device)
        batch_arange = torch.arange(batch_size, device=device)

        if total > 0 and max_new_frames > 0 and batch_size > 0:
            for step_idx in range(max_new_frames):
                cur_len = max_prefix + step_idx
                if cur_len <= 0:
                    break
                hidden_states = self._encode_conditioning(
                    token_ids=token_ids_buf[:, :cur_len],
                    token_type_ids=token_type_buf[:, :cur_len],
                    long_continuous=continuous_buf[:, :cur_len, :],
                    short_continuous=continuous_buf[:, :cur_len, :],
                    attention_mask=attention_buf[:, :cur_len],
                )
                cond_idx = (prefix_lengths + step_idx - 1).clamp(min=0, max=cur_len - 1)
                cond = hidden_states[batch_arange, cond_idx]

                eos_logit = self.eos_head(cond).squeeze(-1)
                eos_prob = torch.sigmoid(eos_logit.float())
                next_frame = self.sample_frame(cond, num_steps=num_sample_steps)

                active_idx = torch.nonzero(~finished, as_tuple=False).squeeze(1)
                if active_idx.numel() > 0:
                    place_idx = prefix_lengths[active_idx] + step_idx
                    continuous_buf[active_idx, place_idx, :] = next_frame[active_idx].to(dtype=dtype)
                    token_type_buf[active_idx, place_idx] = MU.SPEECH_TOKEN_TYPE
                    token_ids_buf[active_idx, place_idx] = self.pad_token_id
                    attention_buf[active_idx, place_idx] = True
                    eos_scores_buf[active_idx, step_idx] = eos_prob[active_idx]
                    gen_count[active_idx] += 1
                    finished[active_idx] = eos_prob[active_idx] > eos_threshold

                if bool(finished.all().item()):
                    break

        max_gen = int(gen_count.max().item()) if batch_size > 0 else 0
        out_norm = torch.zeros((batch_size, max_gen, self.continuous_latent_size), dtype=dtype, device=device)
        eos_scores = torch.zeros((batch_size, max_gen), dtype=torch.float32, device=device)
        for sample_idx in range(batch_size):
            count = int(gen_count[sample_idx].item())
            if count <= 0:
                continue
            start = int(prefix_lengths[sample_idx].item())
            out_norm[sample_idx, :count, :] = continuous_buf[sample_idx, start:start + count, :]
            eos_scores[sample_idx, :count] = eos_scores_buf[sample_idx, :count]

        continuous_latents = self._denormalize(out_norm)

        if not return_dict:
            return continuous_latents
        return MU.PrismTTSGenerationOutput(
            text_ids=text_ids_out,
            continuous_latents=continuous_latents,
            eos_scores=eos_scores,
        )

    # ------------------------------------------------------------------
    # End-to-end generation from raw text/audio
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_e2e(
        self,
        raw_text_prompt: str | Sequence[str],
        raw_speech_prompt: Any | list[Any],
        raw_text_target: str | Sequence[str],
        *,
        text_tokenizer: Optional[Callable[[str], Sequence[int]]] = None,
        speech_encoder: Optional[Callable[[Any], torch.Tensor]] = None,
        speech_decoder: Optional[Callable[[torch.Tensor], Any]] = None,
        output_type: str = "tensor",
        return_dict: bool = True,
        mimi_model_name_or_path: str = "kyutai/mimi",
        mimi_revision: str = "main",
        mimi_token: str | bool | None = None,
        mimi_local_files_only: bool = False,
        mimi_num_quantizers: int = 32,
        **generate_kwargs: Any,
    ) -> MU.PrismTTSGenerationOutput | torch.FloatTensor | torch.Tensor:
        """End-to-end generation from raw text/audio-like inputs (continuous latents)."""
        output_type_normalized = str(output_type).strip().lower()
        if output_type_normalized not in ("tensor", "speech"):
            raise ValueError("output_type must be one of {'tensor', 'speech'}.")

        prompt_text_list = MU.normalize_raw_text_batch(raw_text_prompt, "raw_text_prompt")
        target_text_list = MU.normalize_raw_text_batch(raw_text_target, "raw_text_target")
        speech_prompt_list = (
            list(raw_speech_prompt) if isinstance(raw_speech_prompt, list) else [raw_speech_prompt]
        )

        batch_size = len(prompt_text_list)
        if len(target_text_list) != batch_size:
            raise ValueError("raw_text_target batch size must match raw_text_prompt.")
        if len(speech_prompt_list) != batch_size:
            raise ValueError("raw_speech_prompt batch size must match raw_text_prompt.")

        params = list(self.parameters())
        if len(params) == 0:
            raise RuntimeError("Model has no parameters.")
        device = params[0].device
        continuous_dtype = self.continuous_proj.weight.dtype

        if text_tokenizer is None:
            from dataset.dataset import SharedVocabTokenizer

            discrete_token_count = self.eot_token_id
            vocab_path = Path(__file__).resolve().parents[1] / "dataset" / "vocab.txt"
            text_tokenizer = SharedVocabTokenizer(
                vocab_path=vocab_path,
                text_token_offset=self.text_token_offset,
                eos_token_id=self.eos_token_id,
                append_eos=False,
            )
            del discrete_token_count

        if speech_encoder is None:
            speech_encoder = MU.build_default_mimi_speech_encoder(
                num_quantizers=int(mimi_num_quantizers),
                device=device,
                continuous_dtype=continuous_dtype,
                mimi_model_name_or_path=mimi_model_name_or_path,
                mimi_revision=mimi_revision,
                mimi_token=mimi_token,
                mimi_local_files_only=bool(mimi_local_files_only),
                raw_prompt_name="raw_speech_prompt",
            )
        if speech_decoder is None and output_type_normalized == "speech":
            speech_decoder = MU.build_lazy_mimi_speech_decoder(
                device=device,
                continuous_dtype=continuous_dtype,
                mimi_model_name_or_path=mimi_model_name_or_path,
                mimi_revision=mimi_revision,
                mimi_token=mimi_token,
                mimi_local_files_only=bool(mimi_local_files_only),
            )

        prompt_token_lists: list[list[int]] = []
        target_token_lists: list[list[int]] = []
        continuous_list: list[torch.FloatTensor] = []
        for sample_idx in range(batch_size):
            prompt_token_lists.append([int(tok) for tok in text_tokenizer(prompt_text_list[sample_idx])])
            target_token_lists.append([int(tok) for tok in text_tokenizer(target_text_list[sample_idx])])

            continuous = speech_encoder(speech_prompt_list[sample_idx])
            continuous = torch.as_tensor(continuous, device=device, dtype=continuous_dtype)
            if continuous.dim() == 3 and continuous.shape[0] == 1:
                continuous = continuous.squeeze(0)
            if continuous.dim() != 2:
                raise ValueError("speech_encoder must return continuous latents [L, D] or [D, L].")
            if continuous.shape[1] != self.continuous_latent_size and continuous.shape[0] == self.continuous_latent_size:
                continuous = continuous.transpose(0, 1).contiguous()
            if continuous.shape[1] != self.continuous_latent_size:
                raise ValueError(
                    "Encoded continuous prompt channel mismatch: expected "
                    f"{self.continuous_latent_size}, got {continuous.shape[1]}."
                )
            continuous_list.append(continuous)

        max_prompt_text = max((len(t) for t in prompt_token_lists), default=0)
        max_target_text = max((len(t) for t in target_token_lists), default=0)
        max_speech_prompt = max((int(c.shape[0]) for c in continuous_list), default=0)

        text_prompt = torch.full((batch_size, max_prompt_text), self.pad_token_id, dtype=torch.long, device=device)
        text_target = torch.full((batch_size, max_target_text), self.pad_token_id, dtype=torch.long, device=device)
        continuous_prompt = torch.zeros(
            (batch_size, max_speech_prompt, self.continuous_latent_size), dtype=continuous_dtype, device=device
        )
        text_prompt_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        speech_prompt_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        text_target_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

        for sample_idx in range(batch_size):
            prompt_tokens = prompt_token_lists[sample_idx]
            target_tokens = target_token_lists[sample_idx]
            continuous = continuous_list[sample_idx]
            text_prompt_lengths[sample_idx] = len(prompt_tokens)
            target_len = len(target_tokens)
            text_target_lengths[sample_idx] = target_len
            speech_prompt_lengths[sample_idx] = int(continuous.shape[0])
            if prompt_tokens:
                text_prompt[sample_idx, : len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
            if target_tokens:
                text_target[sample_idx, :target_len] = torch.tensor(target_tokens, dtype=torch.long, device=device)
            if continuous.shape[0] > 0:
                continuous_prompt[sample_idx, : continuous.shape[0], :] = continuous

        generation = self.generate(
            text_prompt=text_prompt,
            continuous_prompt=continuous_prompt,
            text_target=text_target,
            text_prompt_lengths=text_prompt_lengths,
            speech_prompt_lengths=speech_prompt_lengths,
            text_target_lengths=text_target_lengths,
            return_dict=True,
            **generate_kwargs,
        )

        if output_type_normalized == "tensor":
            if return_dict:
                return generation
            return generation.continuous_latents

        if speech_decoder is None:
            raise RuntimeError("speech_decoder is required for output_type='speech'.")
        if generation.continuous_latents is None:
            raise RuntimeError("Generation produced no continuous latents to decode.")

        decoded_speech = speech_decoder(generation.continuous_latents)
        speech_tensor = decoded_speech if torch.is_tensor(decoded_speech) else torch.as_tensor(decoded_speech)
        if speech_tensor.dim() == 3 and speech_tensor.shape[1] == 1:
            speech_tensor = speech_tensor[:, 0, :]
        if speech_tensor.dim() == 1:
            speech_tensor = speech_tensor.unsqueeze(0)
        return speech_tensor
