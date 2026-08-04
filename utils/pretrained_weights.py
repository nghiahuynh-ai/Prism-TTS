"""Load model weights for adaptation without restoring trainer state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class PretrainedWeightLoadResult:
    """Details of a pretrained-weight initialization."""

    checkpoint_path: Path
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    mismatched_keys: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]
    used_ema: bool

    def summary(self) -> str:
        return (
            f"loaded {len(self.loaded_keys)} tensors from {self.checkpoint_path} "
            f"(missing={len(self.missing_keys)}, unexpected={len(self.unexpected_keys)}, "
            f"shape_mismatched={len(self.mismatched_keys)}, ema={self.used_ema})"
        )


def _resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    resolved = Path(checkpoint_path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved


def _as_tensor_state_dict(payload: Any, *, source: str) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(
            f"{source} must be a non-empty mapping of tensor names to tensors."
        )

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError(f"{source} must be a mapping of string names to tensors.")
        state_dict[key] = value
    return state_dict


def extract_model_state_dict(
    checkpoint_payload: Mapping[str, Any],
    *,
    use_ema: bool = False,
) -> dict[str, torch.Tensor]:
    """Extract PrismTTS model weights from raw or Lightning checkpoint payloads."""
    if use_ema:
        ema_state = checkpoint_payload.get("ema_state")
        if isinstance(ema_state, Mapping) and ema_state:
            return _as_tensor_state_dict(ema_state, source="checkpoint ema_state")

    state_dict = checkpoint_payload.get("state_dict")
    if isinstance(state_dict, Mapping) and state_dict:
        lightning_state = _as_tensor_state_dict(
            state_dict, source="checkpoint state_dict"
        )
        return {
            key[len("model.") :] if key.startswith("model.") else key: value
            for key, value in lightning_state.items()
        }

    return _as_tensor_state_dict(checkpoint_payload, source="checkpoint payload")


def _load_checkpoint_payload(checkpoint_path: Path) -> Mapping[str, Any]:
    # Explicitly retain the historical behavior for Lightning checkpoints,
    # which include metadata in addition to tensors. ``weights_only`` was
    # introduced after the oldest PyTorch version supported by this project.
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Unsupported checkpoint payload type: {type(payload).__name__}."
        )
    return payload


def _format_key_preview(keys: tuple[str, ...]) -> str:
    preview = ", ".join(keys[:10])
    return preview + (", ..." if len(keys) > 10 else "")


def load_pretrained_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    use_ema: bool = False,
    strict: bool = False,
) -> PretrainedWeightLoadResult:
    """Initialize compatible model weights without loading optimizer/trainer state.

    With ``strict=False`` (the default), tensors absent from the target model,
    absent from the source checkpoint, or with incompatible shapes are skipped.
    This makes architecture-changing fine-tuning possible while preserving a
    clear report of what was and was not initialized.
    """
    resolved = _resolve_checkpoint_path(checkpoint_path)
    source_state = extract_model_state_dict(
        _load_checkpoint_payload(resolved),
        use_ema=use_ema,
    )
    target_state = model.state_dict()

    compatible_state: dict[str, torch.Tensor] = {}
    unexpected_keys: list[str] = []
    mismatched_keys: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key, value in source_state.items():
        target_value = target_state.get(key)
        if target_value is None:
            unexpected_keys.append(key)
        elif tuple(value.shape) != tuple(target_value.shape):
            mismatched_keys.append((key, tuple(value.shape), tuple(target_value.shape)))
        else:
            compatible_state[key] = value

    missing_keys = tuple(key for key in target_state if key not in compatible_state)
    result = PretrainedWeightLoadResult(
        checkpoint_path=resolved,
        loaded_keys=tuple(compatible_state),
        missing_keys=missing_keys,
        unexpected_keys=tuple(unexpected_keys),
        mismatched_keys=tuple(mismatched_keys),
        used_ema=use_ema,
    )

    if strict and (
        result.missing_keys or result.unexpected_keys or result.mismatched_keys
    ):
        details: list[str] = []
        if result.missing_keys:
            details.append(
                f"missing ({len(result.missing_keys)}): {_format_key_preview(result.missing_keys)}"
            )
        if result.unexpected_keys:
            details.append(
                f"unexpected ({len(result.unexpected_keys)}): "
                f"{_format_key_preview(result.unexpected_keys)}"
            )
        if result.mismatched_keys:
            keys = tuple(item[0] for item in result.mismatched_keys)
            details.append(
                f"shape mismatched ({len(result.mismatched_keys)}): {_format_key_preview(keys)}"
            )
        raise RuntimeError(
            "Pretrained weights are not an exact model match; " + "; ".join(details)
        )

    model.load_state_dict(compatible_state, strict=False)
    return result
