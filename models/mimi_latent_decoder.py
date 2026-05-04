from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _resolve_torch_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if not isinstance(dtype, str):
        raise TypeError("dtype must be a torch.dtype, string alias, or None.")

    normalized = dtype.strip().lower()
    aliases: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        raise ValueError(f"Unsupported dtype alias: {dtype!r}.")
    return resolved


class MimiPreUpsampleLatentDecoder:
    """
    Decode Mimi latents produced right before Mimi's upsampling block.

    This keeps only the decoder-side Mimi modules in memory:
    `upsample -> decoder_transformer -> decoder`.

    The decoder also accepts discrete Mimi codec codes and internally runs
    `quantizer.decode(codes)` before waveform synthesis.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "kyutai/mimi",
        *,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype | None = None,
        local_files_only: bool = False,
        revision: str = "main",
        token: str | bool | None = None,
    ) -> None:
        try:
            from transformers import MimiModel
        except ModuleNotFoundError as exc:
            raise ImportError(
                "MimiPreUpsampleLatentDecoder requires `transformers` with MimiModel support."
            ) from exc

        torch_dtype = _resolve_torch_dtype(dtype)

        load_kwargs: dict[str, Any] = {
            "local_files_only": bool(local_files_only),
            "revision": revision,
        }
        if token is not None:
            load_kwargs["token"] = token
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        mimi = MimiModel.from_pretrained(
            pretrained_model_name_or_path,
            **load_kwargs,
        )
        self.hidden_size = int(mimi.config.hidden_size)
        self.sample_rate = int(getattr(mimi.config, "sampling_rate", 24_000))
        self.device = torch.device(device)
        self.num_codebooks = int(
            getattr(mimi.config, "num_quantizers", 0)
            or getattr(mimi.config, "num_codebooks", 0)
            or 0
        )

        self.quantizer = mimi.quantizer
        self.upsample = mimi.upsample
        self.decoder_transformer = mimi.decoder_transformer
        self.decoder = mimi.decoder
        del mimi

        self._modules: list[nn.Module] = []
        if self.quantizer is not None:
            self._modules.append(self.quantizer)
        if self.upsample is not None:
            self._modules.append(self.upsample)
        self._modules.extend([self.decoder_transformer, self.decoder])

        for module in self._modules:
            module.eval()
            module.requires_grad_(False)
            module.to(device=self.device)
        if torch_dtype is not None:
            for module in self._modules:
                module.to(dtype=torch_dtype)

    def _module_dtype(self) -> torch.dtype:
        for module in self._modules:
            parameter = next(module.parameters(), None)
            if parameter is not None:
                return parameter.dtype
            buffer = next(module.buffers(), None)
            if buffer is not None:
                return buffer.dtype
        return torch.float32

    def _prepare_embeddings(self, latents: torch.Tensor | np.ndarray) -> torch.Tensor:
        if not torch.is_tensor(latents):
            latents = torch.as_tensor(latents)
        latents = latents.detach()
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)
        if latents.dim() != 3:
            raise ValueError(
                "Expected latents with shape [T, C], [B, T, C], or [B, C, T]."
            )

        if latents.shape[-1] == self.hidden_size:
            embeddings = latents.transpose(1, 2).contiguous()
        elif latents.shape[1] == self.hidden_size:
            embeddings = latents.contiguous()
        else:
            raise ValueError(
                f"Latent channels must match Mimi hidden_size={self.hidden_size}, "
                f"got shape={tuple(latents.shape)}."
            )

        return embeddings.to(device=self.device, dtype=self._module_dtype())

    def _prepare_audio_codes(self, codes: torch.Tensor | np.ndarray) -> torch.Tensor:
        if not torch.is_tensor(codes):
            codes = torch.as_tensor(codes)
        codes = codes.detach()

        if codes.dim() == 2:
            if self.num_codebooks > 0:
                if int(codes.shape[0]) == self.num_codebooks:
                    normalized = codes.unsqueeze(0)
                elif int(codes.shape[1]) == self.num_codebooks:
                    normalized = codes.transpose(0, 1).unsqueeze(0)
                elif int(codes.shape[0]) <= int(codes.shape[1]):
                    normalized = codes.unsqueeze(0)
                else:
                    normalized = codes.transpose(0, 1).unsqueeze(0)
            elif int(codes.shape[0]) <= int(codes.shape[1]):
                normalized = codes.unsqueeze(0)
            else:
                normalized = codes.transpose(0, 1).unsqueeze(0)
        elif codes.dim() == 3:
            if self.num_codebooks > 0:
                if int(codes.shape[1]) == self.num_codebooks:
                    normalized = codes
                elif int(codes.shape[2]) == self.num_codebooks:
                    normalized = codes.transpose(1, 2)
                elif int(codes.shape[1]) <= int(codes.shape[2]):
                    normalized = codes
                else:
                    normalized = codes.transpose(1, 2)
            elif int(codes.shape[1]) <= int(codes.shape[2]):
                normalized = codes
            else:
                normalized = codes.transpose(1, 2)
        else:
            raise ValueError(
                "Expected codec codes with shape [N, T], [T, N], [B, N, T], or [B, T, N]."
            )

        return normalized.to(device=self.device, dtype=torch.long)

    def _decode_codes_to_embeddings(self, codes: torch.Tensor | np.ndarray) -> torch.Tensor:
        if self.quantizer is None:
            raise RuntimeError("Mimi quantizer is not available for discrete-code decoding.")
        audio_codes = self._prepare_audio_codes(codes)
        embeddings = self.quantizer.decode(audio_codes)
        if not torch.is_tensor(embeddings):
            embeddings = torch.as_tensor(embeddings)
        if embeddings.dim() != 3:
            raise ValueError(
                "Decoded Mimi quantizer output must have shape [B, C, T], "
                f"got {tuple(embeddings.shape)}."
            )
        return embeddings.to(device=self.device, dtype=self._module_dtype())

    def _looks_like_discrete_codes(self, value: torch.Tensor | np.ndarray) -> bool:
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        if not tensor.dtype.is_floating_point:
            return True
        if self.num_codebooks <= 0:
            return False
        if tensor.dim() == 2:
            return int(tensor.shape[0]) == self.num_codebooks or int(tensor.shape[1]) == self.num_codebooks
        if tensor.dim() == 3:
            return int(tensor.shape[1]) == self.num_codebooks or int(tensor.shape[2]) == self.num_codebooks
        return False

    @torch.inference_mode()
    def __call__(self, representation: torch.Tensor | np.ndarray) -> torch.Tensor:
        if self._looks_like_discrete_codes(representation):
            embeddings = self._decode_codes_to_embeddings(representation)
        else:
            embeddings = self._prepare_embeddings(representation)
        if self.upsample is not None:
            embeddings = self.upsample(embeddings)

        decoder_outputs = self.decoder_transformer(
            embeddings.transpose(1, 2),
            return_dict=True,
        )
        hidden_states = decoder_outputs.last_hidden_state.transpose(1, 2)
        audio = self.decoder(hidden_states)
        if audio.dim() == 3 and audio.shape[1] == 1:
            audio = audio[:, 0, :]
        return audio
