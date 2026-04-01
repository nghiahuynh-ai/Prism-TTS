from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


_REQUIRED_KEYS = (
    "text_target",
    "discrete_target",
    "continuous_target",
    "text_prompt",
    "discrete_prompt",
    "continuous_prompt",
)


def _to_long_1d(value: Any, name: str) -> torch.LongTensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(tensor.shape)}.")
    return tensor.contiguous()


def _to_float_1d(value: Any, name: str) -> torch.FloatTensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(tensor.shape)}.")
    return tensor.contiguous()


def _to_bool_1d(value: Any, name: str) -> torch.BoolTensor:
    tensor = torch.as_tensor(value)
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(tensor.shape)}.")
    return tensor.to(dtype=torch.bool).contiguous()


def _to_long_2d(value: Any, name: str) -> torch.LongTensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(tensor.shape)}.")
    return tensor.contiguous()


def _to_float_2d(value: Any, name: str) -> torch.FloatTensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(tensor.shape)}.")
    return tensor.contiguous()


def _normalize_split_sample(sample: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    missing = [key for key in _REQUIRED_KEYS if key not in sample]
    if missing:
        raise KeyError(f"Missing required keys: {missing}")

    text_target = _to_long_1d(sample["text_target"], "text_target")
    text_prompt = _to_long_1d(sample["text_prompt"], "text_prompt")

    discrete_target = _to_long_2d(sample["discrete_target"], "discrete_target")
    discrete_prompt = _to_long_2d(sample["discrete_prompt"], "discrete_prompt")
    continuous_target = _to_float_2d(sample["continuous_target"], "continuous_target")
    continuous_prompt = _to_float_2d(sample["continuous_prompt"], "continuous_prompt")

    if discrete_target.shape[1] != discrete_prompt.shape[1]:
        raise ValueError(
            "discrete_target and discrete_prompt must have the same number of discrete streams."
        )
    if continuous_target.shape[1] != continuous_prompt.shape[1]:
        raise ValueError(
            "continuous_target and continuous_prompt must have the same channel size."
        )

    normalized: dict[str, torch.Tensor] = {
        "text_target": text_target,
        "discrete_target": discrete_target,
        "continuous_target": continuous_target,
        "text_prompt": text_prompt,
        "discrete_prompt": discrete_prompt,
        "continuous_prompt": continuous_prompt,
    }

    if "attention_mask" in sample and sample["attention_mask"] is not None:
        attention_mask = _to_bool_1d(sample["attention_mask"], "attention_mask")
        normalized["attention_mask"] = attention_mask

    if "flow_timesteps" in sample and sample["flow_timesteps"] is not None:
        flow_timesteps = _to_float_1d(sample["flow_timesteps"], "flow_timesteps")
        normalized["flow_timesteps"] = flow_timesteps

    if "noise" in sample and sample["noise"] is not None:
        noise = _to_float_2d(sample["noise"], "noise")
        if noise.shape[1] != continuous_target.shape[1]:
            raise ValueError("noise channel size must match continuous_target channel size.")
        normalized["noise"] = noise

    return normalized


def _pad_1d(tensors: Sequence[torch.Tensor], pad_value: int | float | bool) -> torch.Tensor:
    if not tensors:
        raise ValueError("Cannot pad an empty tensor list.")
    dtype = tensors[0].dtype
    device = tensors[0].device
    max_length = max(int(t.shape[0]) for t in tensors)
    padded = torch.full((len(tensors), max_length), pad_value, dtype=dtype, device=device)
    for idx, tensor in enumerate(tensors):
        if tensor.dim() != 1:
            raise ValueError(f"Expected 1D tensor, got shape {tuple(tensor.shape)}.")
        padded[idx, : tensor.shape[0]] = tensor.to(device=device, dtype=dtype)
    return padded


def _pad_2d(tensors: Sequence[torch.Tensor], pad_value: int | float) -> torch.Tensor:
    if not tensors:
        raise ValueError("Cannot pad an empty tensor list.")
    dtype = tensors[0].dtype
    device = tensors[0].device
    channels = int(tensors[0].shape[1])
    max_length = max(int(t.shape[0]) for t in tensors)
    padded = torch.full((len(tensors), max_length, channels), pad_value, dtype=dtype, device=device)
    for idx, tensor in enumerate(tensors):
        if tensor.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got shape {tuple(tensor.shape)}.")
        if tensor.shape[1] != channels:
            raise ValueError("All tensors in batch must share the same channel size.")
        padded[idx, : tensor.shape[0], :] = tensor.to(device=device, dtype=dtype)
    return padded
