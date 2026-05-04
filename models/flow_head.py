import math
from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def _resolve_attention_heads(model_channels: int, requested_heads: int) -> int:
    requested = max(1, int(requested_heads))
    if model_channels % requested == 0:
        return requested
    for head_count in range(requested - 1, 0, -1):
        if model_channels % head_count == 0:
            return head_count
    return 1


def _resolve_conv_groups(dim: int, max_groups: int = 16) -> int:
    groups = max(1, min(int(max_groups), int(dim)))
    for candidate in range(groups, 0, -1):
        if dim % candidate == 0:
            return candidate
    return 1


def _apply_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if gate.dim() == 2:
        return x * gate.unsqueeze(1)
    if gate.dim() == 3:
        return x * gate
    raise ValueError(f"gate must be 2D or 3D, got shape {tuple(gate.shape)}.")


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into conditioning vectors."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: int = 10000,
    ) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
            return self.mlp(t_freq)
        if t.dim() == 2:
            flat_t = t.reshape(-1)
            flat_freq = self.timestep_embedding(flat_t, self.frequency_embedding_size)
            flat_emb = self.mlp(flat_freq)
            return flat_emb.view(t.shape[0], t.shape[1], -1)
        raise ValueError("t must be scalar, [batch], or [batch, seq].")


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        conv_groups = _resolve_conv_groups(dim, max_groups=groups)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=conv_groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=conv_groups, padding=kernel_size // 2),
            nn.Mish(),
        )
        self._mask_after_layers = [
            idx for idx, layer in enumerate(self.conv) if isinstance(layer, nn.Conv1d)
        ]

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x must have shape [batch, seq, channels].")
        y = x.transpose(1, 2)
        mask_expanded = None
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)
            y = y.masked_fill(~mask_expanded, 0.0)

        for idx, layer in enumerate(self.conv):
            y = layer(y)
            if mask_expanded is not None and idx in self._mask_after_layers:
                y = y.masked_fill(~mask_expanded, 0.0)
        return y.transpose(1, 2)


class InputEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        *,
        use_conv_pos_embedding: bool = True,
    ):
        super().__init__()
        self.proj = nn.Linear(in_channels, model_channels)
        self.conv_pos_embed = (
            ConvPositionEmbedding(model_channels)
            if use_conv_pos_embedding
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        h = self.proj(x)
        if self.conv_pos_embed is not None:
            h = h + self.conv_pos_embed(h, mask=mask)
        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), 0.0)
        return h


class AdaLayerNorm(nn.Module):
    """DiT-style adaptive LayerNorm for transformer blocks."""

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6, bias=True)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        modulation = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        if shift_msa.dim() == 2:
            normed = modulate(self.norm(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        elif shift_msa.dim() == 3:
            normed = modulate(self.norm(x), shift_msa, scale_msa)
        else:
            raise ValueError("emb must be rank-2 or rank-3.")
        return normed, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormFinal(nn.Module):
    """Final DiT adaptive LayerNorm."""

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2, bias=True)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(emb)).chunk(2, dim=-1)
        if scale.dim() == 2:
            return modulate(self.norm(x), shift.unsqueeze(1), scale.unsqueeze(1))
        if scale.dim() == 3:
            return modulate(self.norm(x), shift, scale)
        raise ValueError("emb must be rank-2 or rank-3.")


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner_dim = max(dim, int(dim * ff_mult))
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim, ff_mult=ff_mult, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        *,
        mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t_emb)
        key_padding_mask = None if mask is None else ~mask
        attn_out, _ = self.attn(
            norm_x,
            norm_x,
            norm_x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + _apply_gate(attn_out, gate_msa)

        ff_in = self.ff_norm(x)
        if shift_mlp.dim() == 2:
            ff_in = modulate(ff_in, shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        elif shift_mlp.dim() == 3:
            ff_in = modulate(ff_in, shift_mlp, scale_mlp)
        else:
            raise ValueError("timestep embedding must be rank-2 or rank-3.")

        ff_out = self.ff(ff_in)
        x = x + _apply_gate(ff_out, gate_mlp)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x


class FlowHead(nn.Module):
    """DiT-style flow head for continuous latent velocity prediction."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        *,
        num_heads: int = 8,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
        use_conv_pos_embedding: bool = True,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        if in_channels < 1 or model_channels < 1 or out_channels < 1:
            raise ValueError("FlowHead channel sizes must all be >= 1.")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be >= 1.")

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.num_heads = _resolve_attention_heads(model_channels, num_heads)

        self.time_embed = TimestepEmbedder(model_channels)
        self.input_embed = InputEmbedding(
            in_channels=in_channels,
            model_channels=model_channels,
            use_conv_pos_embedding=use_conv_pos_embedding,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=model_channels,
                    num_heads=self.num_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(num_res_blocks)
            ]
        )
        self.norm_out = AdaLayerNormFinal(model_channels)
        self.proj_out = nn.Linear(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out AdaLN modulation layers in DiT blocks.
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers.
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    @staticmethod
    def _normalize_inputs(
        x: torch.Tensor,
        mask: torch.BoolTensor | None,
    ) -> tuple[torch.Tensor, torch.BoolTensor, bool]:
        squeeze_seq_dim = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_seq_dim = True
        elif x.dim() != 3:
            raise ValueError("x must have shape [batch, channels] or [batch, seq, channels].")

        batch_size, seq_len, _ = x.shape
        if mask is None:
            mask = torch.ones(
                batch_size,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            )
        else:
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)
            if mask.dim() != 2 or mask.shape[0] != batch_size or mask.shape[1] != seq_len:
                raise ValueError("mask must have shape [batch, seq].")
            mask = mask.to(device=x.device, dtype=torch.bool)

        return x, mask, squeeze_seq_dim

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """
        Predict flow velocity.

        Args:
            x: [batch, channels] or [batch, seq, channels]
            t: scalar, [batch], or [batch, seq]
            mask: optional valid-position mask [batch, seq]
        """
        x, resolved_mask, squeeze_seq_dim = self._normalize_inputs(x, mask)
        h = self.input_embed(x, mask=resolved_mask)
        t_emb = self.time_embed(t).to(dtype=h.dtype, device=h.device)

        if t_emb.dim() == 2 and t_emb.shape[0] != h.shape[0]:
            raise ValueError("t batch size must match inputs.")
        if t_emb.dim() == 3 and t_emb.shape[:2] != h.shape[:2]:
            raise ValueError("t shape must match [batch, seq] when rank-3.")
        if t_emb.dim() not in (2, 3):
            raise ValueError("t embedding must be rank-2 or rank-3.")

        if self.grad_checkpointing and self.training and not torch.jit.is_scripting():
            for block in self.transformer_blocks:
                h = checkpoint(
                    partial(block, mask=resolved_mask),
                    h,
                    t_emb,
                    use_reentrant=False,
                )
        else:
            for block in self.transformer_blocks:
                h = block(h, t_emb, mask=resolved_mask)

        h = self.norm_out(h, t_emb)
        out = self.proj_out(h)
        out = out.masked_fill(~resolved_mask.unsqueeze(-1), 0.0)
        if squeeze_seq_dim:
            out = out.squeeze(1)
        return out

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cfg_scale: float,
        mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        combined_mask = None
        if mask is not None:
            half_mask = mask[: len(mask) // 2]
            combined_mask = torch.cat([half_mask, half_mask], dim=0)

        model_out = self.forward(combined, t, mask=combined_mask)
        eps = model_out[..., : self.in_channels]
        rest = model_out[..., self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=-1)
