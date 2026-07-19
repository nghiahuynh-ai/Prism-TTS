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
from models.llama_backbone import (
    LlamaBackbone,
    WindowedCausalEncoder,
    build_bidirectional_4d_mask,
)
from utils import model_utils as MU


class PrismTTS(nn.Module):
    """
    Prism-TTS masked-generative continuous-latent model (no vector quantization).

    Speech is modeled as one continuous latent per frame. A (bidirectional) Llama
    backbone produces a contextual embedding for every position; a small per-frame
    head (``FlowHead``, an AdaLN MLP) generates each masked frame, trained with
    MeanFlow (average-velocity) or plain flow matching. A binary EOS head predicts
    end-of-sequence.

    Single-utterance sequence layout (per sample):
    text_target -> EOT -> [target frames]

    Masked-generative objective: the first ``protected_prefix_ratio`` fraction of
    frames is kept clean as an in-context condition; a random suffix of the
    remaining frames is masked (replaced by a learned mask token) and reconstructed
    from full bidirectional context. There is no teacher-forcing shift: the head
    for a masked frame is conditioned on that frame's own hidden state.
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
        short_context_chunk_size: int = 128,
        gradient_checkpointing: bool = False,
        # Masked-generative (MAR) + MeanFlow settings.
        attn_mode: str = "bidirectional",
        head_mode: str = "flowmatch",
        protected_prefix_ratio: float = 0.3,
        always_mask_terminal: bool = True,
        noise_mode: str = "ramp",
        ramp_max: float = 1.0,
        ramp_random_intensity: bool = True,
        meanflow_p: float = 1.0,
        meanflow_c: float = 1e-3,
        t_logit_mean: float = -0.4,
        t_logit_std: float = 1.0,
        r_eq_t_prob: float = 0.75,
        eos_pos_weight: Optional[float] = None,
        default_block_size: int = 1,
        frames_after_eos: int = 0,
    ):
        """Initialize backbone, per-frame flow head, EOS head, mask token, and stats."""
        super().__init__()
        if continuous_latent_size < 1:
            raise ValueError("continuous_latent_size must be at least 1.")
        if eos_loss_weight < 0.0:
            raise ValueError("eos_loss_weight must be >= 0.")
        if sigma_data <= 0.0:
            raise ValueError("sigma_data must be > 0.")
        if sample_steps < 1:
            raise ValueError("sample_steps must be at least 1.")
        if not (0.0 < latent_stats_decay < 1.0):
            raise ValueError("latent_stats_decay must be in (0, 1).")
        attn_mode = str(attn_mode).strip().lower()
        head_mode = str(head_mode).strip().lower()
        noise_mode = str(noise_mode).strip().lower()
        if attn_mode not in ("bidirectional", "causal"):
            raise ValueError("attn_mode must be one of {'bidirectional', 'causal'}.")
        if head_mode not in ("flowmatch", "meanflow"):
            raise ValueError("head_mode must be one of {'flowmatch', 'meanflow'}.")
        if not (0.0 <= protected_prefix_ratio < 1.0):
            raise ValueError("protected_prefix_ratio must be in [0, 1).")
        if noise_mode not in ("ramp", "iid", "none"):
            raise ValueError("noise_mode must be one of {'ramp', 'iid', 'none'}.")
        if meanflow_p < 0.0:
            raise ValueError("meanflow_p must be >= 0.")
        if meanflow_c <= 0.0:
            raise ValueError("meanflow_c must be > 0.")
        if t_logit_std <= 0.0:
            raise ValueError("t_logit_std must be > 0.")
        if not (0.0 <= r_eq_t_prob <= 1.0):
            raise ValueError("r_eq_t_prob must be in [0, 1].")
        if default_block_size < 1:
            raise ValueError("default_block_size must be at least 1.")
        if frames_after_eos < 0:
            raise ValueError("frames_after_eos must be >= 0.")

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
        self.short_context_chunk_size = int(short_context_chunk_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.attn_mode = str(attn_mode)
        self.head_mode = str(head_mode)
        self.protected_prefix_ratio = float(protected_prefix_ratio)
        self.always_mask_terminal = bool(always_mask_terminal)
        self.noise_mode = str(noise_mode)
        self.ramp_max = float(ramp_max)
        self.ramp_random_intensity = bool(ramp_random_intensity)
        self.meanflow_p = float(meanflow_p)
        self.meanflow_c = float(meanflow_c)
        self.t_logit_mean = float(t_logit_mean)
        self.t_logit_std = float(t_logit_std)
        self.r_eq_t_prob = float(r_eq_t_prob)
        self.eos_pos_weight = None if eos_pos_weight is None else float(eos_pos_weight)
        self.default_block_size = int(default_block_size)
        self.frames_after_eos = int(frames_after_eos)

        self.backbone = LlamaBackbone(llama_config)
        self.backbone.gradient_checkpointing = self.gradient_checkpointing
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
                chunk_size=self.short_context_chunk_size,
            )
            if self.use_short_context
            else None
        )
        if self.short_encoder is not None:
            self.short_encoder.gradient_checkpointing = self.gradient_checkpointing

        # Per-frame head F_theta(x_t, t[, r]; cond) conditioned on the backbone
        # hidden state directly.
        self.flow_head = FlowHead(
            in_channels=self.continuous_latent_size,
            model_channels=flow_model_channels or self.hidden_size,
            out_channels=self.continuous_latent_size,
            z_channels=self.hidden_size,
            num_res_blocks=flow_num_res_blocks,
            # torch.func.jvp (MeanFlow) is incompatible with checkpoint's custom
            # autograd function. Backbone checkpointing still covers the dominant
            # sequence activations in that mode.
            grad_checkpointing=self.gradient_checkpointing and self.head_mode != "meanflow",
        )

        # Learned text/speech type embeddings added to token/latent embeddings.
        self.type_embeddings = nn.Parameter(torch.empty(2, self.hidden_size))
        # Learned embedding substituted at masked speech positions (MAR mask token).
        self.masked_speech_embedding = nn.Parameter(torch.empty(self.hidden_size))

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
        nn.init.normal_(self.masked_speech_embedding, mean=0.0, std=std)
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
        masked_positions: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        vocab_size = int(self.backbone.config.vocab_size)
        token_ids_safe = token_ids.clamp(min=0, max=vocab_size - 1)
        text_embeds = self.backbone.embed_tokens(token_ids_safe)
        speech_embeds = self.continuous_proj(continuous_norm.to(dtype=text_embeds.dtype))
        is_speech = (token_type_ids == MU.SPEECH_TOKEN_TYPE).unsqueeze(-1)
        embeds = torch.where(is_speech, speech_embeds, text_embeds)
        if masked_positions is not None:
            mask_emb = self.masked_speech_embedding.to(dtype=embeds.dtype).view(1, 1, -1)
            embeds = torch.where(masked_positions.unsqueeze(-1), mask_emb, embeds)
        type_embeds = self.type_embeddings[token_type_ids.clamp(min=0, max=1)]
        return embeds + type_embeds

    def _backbone_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Backbone mask: bidirectional 4D (MAR) or a 2D pad mask (causal fallback)."""
        if self.attn_mode == "bidirectional":
            batch_size, seq_len, _ = inputs_embeds.shape
            return build_bidirectional_4d_mask(
                attention_mask,
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
        return attention_mask

    def _encode_hidden(
        self,
        token_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        continuous_norm: torch.FloatTensor,
        attention_mask: torch.BoolTensor,
        masked_positions: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        inputs_embeds = self._embed(token_ids, token_type_ids, continuous_norm, masked_positions)
        backbone_mask = self._backbone_attention_mask(attention_mask, inputs_embeds)
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=backbone_mask,
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
        masked_positions: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        """Per-position head conditioning Z = z_long + z_short.

        The long-context backbone sees `long_continuous` (noise-injected during
        training); the short-context branch sees `short_continuous` (always clean).
        Masked speech positions get a learned mask token in both branches.
        """
        z = self._encode_hidden(
            token_ids, token_type_ids, long_continuous, attention_mask, masked_positions
        )
        if self.short_encoder is not None:
            short_embeds = self._embed(
                token_ids, token_type_ids, short_continuous, masked_positions
            )
            z = z + self.short_encoder(short_embeds, attention_mask)
        return z

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------
    def _sample_masked_positions(self, flat: MU.FlatBatch) -> torch.BoolTensor:
        """Sample a contiguous-suffix mask over each sample's target frames.

        Protects the first `protected_prefix_ratio` fraction of frames (in-context
        condition), masks a random suffix of length K in [1, eligible], and (if
        `always_mask_terminal`) forces the terminal frame to be masked so the EOS
        head is trained on the true end.
        """
        batch_size, seq_len = flat.token_ids.shape
        device = flat.token_ids.device
        is_target = (flat.target_block_ids >= 0) & flat.attention_mask
        counts = flat.target_block_counts.to(device=device)
        bid = flat.target_block_ids
        masked = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            length = int(counts[i].item())
            if length <= 0:
                continue
            protected = int(math.ceil(length * self.protected_prefix_ratio))
            protected = min(length, max(0, protected))
            eligible = length - protected
            target_row = is_target[i]
            if eligible <= 0:
                if self.always_mask_terminal:
                    masked[i] = target_row & (bid[i] == (length - 1))
                continue
            k = int(torch.randint(1, eligible + 1, (1,), device=device).item())
            mask_start = length - k
            row_masked = target_row & (bid[i] >= mask_start)
            if self.always_mask_terminal:
                row_masked = row_masked | (target_row & (bid[i] == (length - 1)))
            masked[i] = row_masked
        return masked

    # ------------------------------------------------------------------
    # Per-frame flow head (MeanFlow / flow matching)
    # ------------------------------------------------------------------
    def _sample_logit_normal(
        self, num: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Sample t in (0,1) from a logit-normal proposal (MeanFlow recipe)."""
        raw = torch.randn(num, 1, device=device, dtype=dtype) * self.t_logit_std + self.t_logit_mean
        return torch.sigmoid(raw)

    def _flow_loss(self, x0: torch.FloatTensor, cond: torch.FloatTensor) -> torch.Tensor:
        if self.head_mode == "meanflow":
            return self._meanflow_loss(x0, cond)
        return self._flowmatch_loss(x0, cond)

    def _flowmatch_loss(self, x0: torch.FloatTensor, cond: torch.FloatTensor) -> torch.Tensor:
        """Rectified-flow control: MSE on instantaneous velocity v = e - x0."""
        device = x0.device
        dtype = torch.float32
        x0 = x0.to(dtype)
        cond = cond.to(dtype)
        num = x0.shape[0]
        autocast_device = device.type if device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=autocast_device, enabled=False):
            t = self._sample_logit_normal(num, device, dtype)
            e = torch.randn_like(x0)
            z_t = (1.0 - t) * x0 + t * e
            v = e - x0
            pred = self.flow_head(z_t, t.reshape(-1), cond)
            return F.mse_loss(pred, v)

    def _meanflow_loss(self, x0: torch.FloatTensor, cond: torch.FloatTensor) -> torch.Tensor:
        """MeanFlow average-velocity loss (arXiv 2505.13447).

        u_tgt = v - (t - r) * d/dt u, with d/dt u via a JVP over (z_t, r, t) with
        tangent (v, 0, 1); adaptive weighting w = 1 / (||delta||^2 + c)^p (stop-grad).
        Run in fp32 with autocast disabled (JVP is unstable under mixed precision).
        """
        device = x0.device
        dtype = torch.float32
        x0 = x0.to(dtype)
        cond = cond.to(dtype)
        num = x0.shape[0]
        autocast_device = device.type if device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=autocast_device, enabled=False):
            t = self._sample_logit_normal(num, device, dtype)
            r = self._sample_logit_normal(num, device, dtype)
            r = torch.minimum(r, t)
            eq = torch.rand(num, 1, device=device, dtype=dtype) < self.r_eq_t_prob
            r = torch.where(eq, t, r)

            e = torch.randn_like(x0)
            z_t = (1.0 - t) * x0 + t * e
            v = e - x0

            def fwd(z_in: torch.Tensor, r_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
                return self.flow_head(z_in, t_in.reshape(-1), cond, r_in.reshape(-1))

            u, dudt = torch.func.jvp(
                fwd,
                (z_t, r, t),
                (v, torch.zeros_like(r), torch.ones_like(t)),
            )
            u_tgt = (v - (t - r) * dudt).detach()
            delta = u - u_tgt
            squared = delta.pow(2)
            with torch.no_grad():
                weight = 1.0 / (squared.mean(dim=-1, keepdim=True) + self.meanflow_c).pow(
                    self.meanflow_p
                )
            return (weight * squared).mean()

    @torch.no_grad()
    def sample_frame(
        self,
        cond: torch.FloatTensor,
        num_steps: Optional[int] = None,
    ) -> torch.FloatTensor:
        """Sample normalized continuous frames from the head (1-step or few-step)."""
        num_steps = self.sample_steps if num_steps is None else int(num_steps)
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")
        device = cond.device
        dtype = cond.dtype
        num = cond.shape[0]
        x = torch.randn(num, self.continuous_latent_size, device=device, dtype=dtype)
        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)

        if self.head_mode == "meanflow":
            for step_idx in range(num_steps):
                t_cur = ts[step_idx]
                t_next = ts[step_idx + 1]
                t_vec = torch.full((num,), float(t_cur), device=device, dtype=dtype)
                r_vec = torch.full((num,), float(t_next), device=device, dtype=dtype)
                u = self.flow_head(x, t_vec, cond, r_vec)
                x = x - (t_cur - t_next) * u
            return x

        # Flow-matching Euler integration from t=1 (noise) to t=0 (data).
        for step_idx in range(num_steps):
            t_cur = ts[step_idx]
            dt = ts[step_idx] - ts[step_idx + 1]
            t_vec = torch.full((num,), float(t_cur), device=device, dtype=dtype)
            v = self.flow_head(x, t_vec, cond)
            x = x - dt * v
        return x

    def _eos_loss(
        self, eos_logit: torch.Tensor, eos_target: torch.Tensor
    ) -> torch.Tensor:
        """Binary EOS loss with a positive-class weight for the 1-per-utterance imbalance."""
        if self.eos_pos_weight is not None:
            pos_weight = torch.tensor(
                self.eos_pos_weight, device=eos_logit.device, dtype=eos_logit.dtype
            )
        else:
            pos = eos_target.sum()
            neg = float(eos_target.numel()) - pos
            pos_weight = torch.where(
                pos > 0,
                neg / pos.clamp(min=1.0),
                torch.ones((), device=eos_logit.device, dtype=eos_target.dtype),
            ).clamp(min=1.0, max=1000.0).to(dtype=eos_logit.dtype)
        return F.binary_cross_entropy_with_logits(
            eos_logit, eos_target, pos_weight=pos_weight
        )

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
        """Run masked-generative reconstruction training from collate tensors."""
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

        # Clean normalized latents supply the per-frame head targets + short context.
        continuous_norm = self._normalize(flat.continuous_values)

        # Sample the masked suffix (per sample) over target frames.
        masked_positions = self._sample_masked_positions(flat)

        # The long-context backbone consumes a noise-perturbed copy of the visible
        # context during training (exposure-bias regularization). Masked positions
        # are replaced by the mask token in `_embed`, so their values are irrelevant.
        long_continuous = continuous_norm
        if self.training and self.inject_backbone_noise and self.noise_mode != "none":
            if self.noise_mode == "ramp":
                long_continuous = MU.apply_context_noise_ramp(
                    continuous_norm,
                    flat.target_block_ids,
                    flat.target_block_counts,
                    masked_positions,
                    speech_mask,
                    protected_prefix_ratio=self.protected_prefix_ratio,
                    k_max=self.ramp_max,
                    random_intensity=self.ramp_random_intensity,
                )
            else:  # "iid" (CALM-style per-frame noise on the visible context)
                noised = MU.inject_continuous_backbone_noise(continuous_norm)
                visible_speech = (speech_mask & (~masked_positions)).unsqueeze(-1)
                long_continuous = torch.where(visible_speech, noised, continuous_norm)

        hidden_states = self._encode_conditioning(
            token_ids=flat.token_ids,
            token_type_ids=flat.token_type_ids,
            long_continuous=long_continuous,
            short_continuous=continuous_norm,
            attention_mask=flat.attention_mask,
            masked_positions=masked_positions,
        )

        counts = flat.target_block_counts.to(device=hidden_states.device)
        last_frame_id = counts[:, None] - 1
        eos_target_full = (flat.target_block_ids == last_frame_id) & (flat.target_block_ids >= 0)

        cond = hidden_states[masked_positions]
        target_norm = continuous_norm[masked_positions]
        eos_target = eos_target_full[masked_positions].to(dtype=hidden_states.dtype)

        if cond.shape[0] == 0:
            zero = hidden_states.sum() * 0.0
            flow_loss = zero
            eos_loss = zero
        else:
            flow_loss = self._flow_loss(target_norm, cond)
            eos_logit = self.eos_head(cond).squeeze(-1)
            eos_loss = self._eos_loss(eos_logit, eos_target)

        loss = flow_loss + self.eos_loss_weight * eos_loss

        if self.training:
            self.train_step += 1

        if not return_dict:
            return loss, flow_loss, eos_loss
        return MU.PrismTTSOutput(
            loss=loss,
            flow_loss=flow_loss,
            eos_loss=eos_loss,
        )

    # ------------------------------------------------------------------
    # Masked-generative generation (full-recompute)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        *,
        text_target: torch.LongTensor,
        continuous_condition: Optional[torch.FloatTensor] = None,
        text_target_lengths: Optional[torch.Tensor | int] = None,
        condition_lengths: Optional[torch.Tensor | int] = None,
        max_new_frames: int = 1024,
        block_size: Optional[int] = None,
        eos_threshold: Optional[float] = None,
        num_sample_steps: Optional[int] = None,
        frames_after_eos: Optional[int] = None,
        return_dict: bool = True,
    ) -> MU.PrismTTSGenerationOutput | torch.FloatTensor:
        """Generate continuous frames left-to-right (latent- or block-by-block).

        Layout: text_target -> EOT -> [optional condition frames] -> [generated].
        `block_size` = 1 is latent-by-latent (highest quality); larger blocks decode
        `block_size` masked frames per backbone pass (faster, but the frames in a
        block are sampled independently from the same context).
        """
        text_target = MU.normalize_text_tokens(text_target, "text_target")
        batch_size = int(text_target.shape[0])
        device = text_target.device
        dtype = self.continuous_proj.weight.dtype
        latent_size = self.continuous_latent_size

        block_size = self.default_block_size if block_size is None else int(block_size)
        if block_size < 1:
            raise ValueError("block_size must be at least 1.")
        eos_threshold = self.eos_threshold if eos_threshold is None else float(eos_threshold)
        num_sample_steps = self.sample_steps if num_sample_steps is None else int(num_sample_steps)
        frames_after_eos = (
            self.frames_after_eos if frames_after_eos is None else int(frames_after_eos)
        )
        if frames_after_eos < 0:
            raise ValueError("frames_after_eos must be >= 0.")
        max_new_frames = int(max_new_frames)
        if max_new_frames < 0:
            raise ValueError("max_new_frames must be >= 0.")

        text_target_lengths = MU.normalize_lengths(
            text_target_lengths, batch_size, int(text_target.shape[1]),
            "text_target_lengths", device, default_value=int(text_target.shape[1]),
        )

        if continuous_condition is None:
            cond_frames_norm = torch.zeros((batch_size, 0, latent_size), dtype=dtype, device=device)
            condition_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            continuous_condition = MU.normalize_continuous_latents(
                continuous_condition,
                expected_len=int(continuous_condition.shape[1]),
                name="continuous_condition",
                continuous_latent_size=latent_size,
            )
            condition_lengths = MU.normalize_lengths(
                condition_lengths, batch_size, int(continuous_condition.shape[1]),
                "condition_lengths", device, default_value=int(continuous_condition.shape[1]),
            )
            cond_frames_norm = self._normalize(continuous_condition.to(dtype=dtype))

        max_text_target = int(text_target_lengths.max().item()) if batch_size > 0 else 0
        text_ids_out = text_target[:, :max_text_target]

        prefix_lengths = text_target_lengths + 1 + condition_lengths
        max_prefix = int(prefix_lengths.max().item()) if batch_size > 0 else 0
        total = max_prefix + max_new_frames

        token_ids_buf = torch.full((batch_size, total), self.pad_token_id, dtype=torch.long, device=device)
        token_type_buf = torch.full((batch_size, total), MU.TEXT_TOKEN_TYPE, dtype=torch.long, device=device)
        continuous_buf = torch.zeros((batch_size, total, latent_size), dtype=dtype, device=device)
        attention_buf = torch.zeros((batch_size, total), dtype=torch.bool, device=device)
        masked_buf = torch.zeros((batch_size, total), dtype=torch.bool, device=device)

        for sample_idx in range(batch_size):
            lt = int(text_target_lengths[sample_idx].item())
            lc = int(condition_lengths[sample_idx].item())
            pos = 0
            if lt > 0:
                token_ids_buf[sample_idx, pos:pos + lt] = text_target[sample_idx, :lt]
            pos += lt
            token_ids_buf[sample_idx, pos] = self.eot_token_id
            pos += 1
            if lc > 0:
                token_type_buf[sample_idx, pos:pos + lc] = MU.SPEECH_TOKEN_TYPE
                continuous_buf[sample_idx, pos:pos + lc, :] = cond_frames_norm[sample_idx, :lc, :]
            pos += lc
            attention_buf[sample_idx, :pos] = True

        gen_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        eos_countdown = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        eos_scores_buf = torch.zeros((batch_size, max_new_frames), dtype=torch.float32, device=device)
        batch_arange = torch.arange(batch_size, device=device)

        step = 0
        while step < max_new_frames and batch_size > 0:
            cur_block = min(block_size, max_new_frames - step)
            for offset in range(cur_block):
                place = prefix_lengths + (step + offset)
                token_type_buf[batch_arange, place] = MU.SPEECH_TOKEN_TYPE
                attention_buf[batch_arange, place] = True
                masked_buf[batch_arange, place] = True

            cur_len = max_prefix + step + cur_block
            hidden_states = self._encode_conditioning(
                token_ids=token_ids_buf[:, :cur_len],
                token_type_ids=token_type_buf[:, :cur_len],
                long_continuous=continuous_buf[:, :cur_len, :],
                short_continuous=continuous_buf[:, :cur_len, :],
                attention_mask=attention_buf[:, :cur_len],
                masked_positions=masked_buf[:, :cur_len],
            )

            for offset in range(cur_block):
                idx = step + offset
                place = prefix_lengths + idx
                cond_vec = hidden_states[batch_arange, place]
                eos_logit = self.eos_head(cond_vec).squeeze(-1)
                eos_prob = torch.sigmoid(eos_logit.float())
                frame = self.sample_frame(cond_vec, num_steps=num_sample_steps).to(dtype=dtype)

                active = ~finished
                # Commit the frame and unmask (finished samples ramble; ignored later).
                continuous_buf[batch_arange, place, :] = frame
                token_ids_buf[batch_arange, place] = self.pad_token_id
                masked_buf[batch_arange, place] = False

                eos_scores_buf[batch_arange, idx] = torch.where(
                    active, eos_prob, eos_scores_buf[batch_arange, idx]
                )
                gen_count = gen_count + active.long()

                just_fired = active & (eos_countdown < 0) & (eos_prob > eos_threshold)
                eos_countdown = torch.where(
                    just_fired,
                    torch.full_like(eos_countdown, frames_after_eos),
                    eos_countdown,
                )
                has_fired = active & (eos_countdown >= 0)
                finish_now = has_fired & (eos_countdown == 0)
                finished = finished | finish_now
                decrement = has_fired & (eos_countdown > 0)
                eos_countdown = torch.where(decrement, eos_countdown - 1, eos_countdown)

            step += cur_block
            if bool(finished.all().item()):
                break

        max_gen = int(gen_count.max().item()) if batch_size > 0 else 0
        out_norm = torch.zeros((batch_size, max_gen, latent_size), dtype=dtype, device=device)
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
        raw_text_target: str | Sequence[str],
        *,
        raw_speech_condition: Any | list[Any] | None = None,
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
        """End-to-end generation: target text (+ optional reference audio) -> latents/speech.

        The optional reference clip (`raw_speech_condition`) seeds the in-context
        condition frames for zeroshot voice cloning; without it this is plain TTS.
        """
        output_type_normalized = str(output_type).strip().lower()
        if output_type_normalized not in ("tensor", "speech"):
            raise ValueError("output_type must be one of {'tensor', 'speech'}.")

        target_text_list = MU.normalize_raw_text_batch(raw_text_target, "raw_text_target")
        batch_size = len(target_text_list)

        if raw_speech_condition is None:
            speech_condition_list: list[Any] = [None] * batch_size
        elif isinstance(raw_speech_condition, list):
            speech_condition_list = list(raw_speech_condition)
        else:
            speech_condition_list = [raw_speech_condition]
        if len(speech_condition_list) != batch_size:
            raise ValueError("raw_speech_condition batch size must match raw_text_target.")

        params = list(self.parameters())
        if len(params) == 0:
            raise RuntimeError("Model has no parameters.")
        device = params[0].device
        continuous_dtype = self.continuous_proj.weight.dtype

        if text_tokenizer is None:
            from dataset.dataset import SharedVocabTokenizer

            vocab_path = Path(__file__).resolve().parents[1] / "dataset" / "vocab.txt"
            text_tokenizer = SharedVocabTokenizer(
                vocab_path=vocab_path,
                text_token_offset=self.text_token_offset,
                eos_token_id=self.eos_token_id,
                append_eos=False,
            )

        needs_encoder = any(item is not None for item in speech_condition_list)
        if speech_encoder is None and needs_encoder:
            speech_encoder = MU.build_default_mimi_speech_encoder(
                num_quantizers=int(mimi_num_quantizers),
                device=device,
                continuous_dtype=continuous_dtype,
                mimi_model_name_or_path=mimi_model_name_or_path,
                mimi_revision=mimi_revision,
                mimi_token=mimi_token,
                mimi_local_files_only=bool(mimi_local_files_only),
                raw_prompt_name="raw_speech_condition",
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

        target_token_lists: list[list[int]] = []
        continuous_list: list[Optional[torch.FloatTensor]] = []
        for sample_idx in range(batch_size):
            target_token_lists.append(
                [int(tok) for tok in text_tokenizer(target_text_list[sample_idx])]
            )
            raw_condition = speech_condition_list[sample_idx]
            if raw_condition is None:
                continuous_list.append(None)
                continue
            continuous = speech_encoder(raw_condition)
            continuous = torch.as_tensor(continuous, device=device, dtype=continuous_dtype)
            if continuous.dim() == 3 and continuous.shape[0] == 1:
                continuous = continuous.squeeze(0)
            if continuous.dim() != 2:
                raise ValueError("speech_encoder must return continuous latents [L, D] or [D, L].")
            if continuous.shape[1] != self.continuous_latent_size and continuous.shape[0] == self.continuous_latent_size:
                continuous = continuous.transpose(0, 1).contiguous()
            if continuous.shape[1] != self.continuous_latent_size:
                raise ValueError(
                    "Encoded continuous condition channel mismatch: expected "
                    f"{self.continuous_latent_size}, got {continuous.shape[1]}."
                )
            continuous_list.append(continuous)

        max_target_text = max((len(t) for t in target_token_lists), default=0)
        max_condition = max(
            (int(c.shape[0]) for c in continuous_list if c is not None), default=0
        )

        text_target = torch.full((batch_size, max_target_text), self.pad_token_id, dtype=torch.long, device=device)
        continuous_condition = torch.zeros(
            (batch_size, max_condition, self.continuous_latent_size), dtype=continuous_dtype, device=device
        )
        text_target_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        condition_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

        for sample_idx in range(batch_size):
            target_tokens = target_token_lists[sample_idx]
            text_target_lengths[sample_idx] = len(target_tokens)
            if target_tokens:
                text_target[sample_idx, : len(target_tokens)] = torch.tensor(
                    target_tokens, dtype=torch.long, device=device
                )
            continuous = continuous_list[sample_idx]
            if continuous is not None and continuous.shape[0] > 0:
                condition_lengths[sample_idx] = int(continuous.shape[0])
                continuous_condition[sample_idx, : continuous.shape[0], :] = continuous

        generation = self.generate(
            text_target=text_target,
            continuous_condition=continuous_condition if max_condition > 0 else None,
            text_target_lengths=text_target_lengths,
            condition_lengths=condition_lengths if max_condition > 0 else None,
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
