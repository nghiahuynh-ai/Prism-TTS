from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.prism_tts import PrismTTS


def resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    resolved = Path(checkpoint_path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved


def load_checkpoint_payload(checkpoint_path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = resolve_checkpoint_path(checkpoint_path)
    payload = torch.load(resolved, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload).__name__}.")
    return resolved, payload


def extract_model_state_dict(
    checkpoint_payload: dict[str, Any],
    *,
    use_ema: bool,
) -> dict[str, torch.Tensor]:
    if use_ema:
        ema_state = checkpoint_payload.get("ema_state")
        if isinstance(ema_state, dict) and ema_state:
            return dict(ema_state)

    state_dict = checkpoint_payload.get("state_dict")
    if isinstance(state_dict, dict) and state_dict:
        stripped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                stripped[key[len("model.") :]] = value
            else:
                stripped[key] = value
        return stripped

    if checkpoint_payload and all(isinstance(key, str) for key in checkpoint_payload):
        if all(torch.is_tensor(value) for value in checkpoint_payload.values()):
            return dict(checkpoint_payload)

    raise ValueError("Unable to find a model state dict in checkpoint payload.")


def load_model_weights(
    model: PrismTTS,
    checkpoint_path: str | Path,
    *,
    use_ema: bool,
    strict: bool = True,
) -> tuple[Path, list[str], list[str]]:
    resolved, payload = load_checkpoint_payload(checkpoint_path)
    state_dict = extract_model_state_dict(payload, use_ema=use_ema)
    try:
        incompatible = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load model weights from {resolved}: {exc}") from exc
    return resolved, list(incompatible.missing_keys), list(incompatible.unexpected_keys)
