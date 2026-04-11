from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPY_PATH = Path("samples/EN_B00002_S09994_W000018.npy")


def _extract_modal_arrays(payload: Any, npy_path: Path) -> tuple[Any, Any]:
    if isinstance(payload, np.ndarray) and payload.dtype.names is not None:
        names = set(payload.dtype.names)
        if "discrete" in names and "continuous" in names:
            discrete = payload["discrete"]
            continuous = payload["continuous"]
            if isinstance(discrete, np.ndarray) and discrete.dtype == object and discrete.shape == ():
                discrete = discrete.item()
            if (
                isinstance(continuous, np.ndarray)
                and continuous.dtype == object
                and continuous.shape == ()
            ):
                continuous = continuous.item()
            return discrete, continuous

    if isinstance(payload, np.ndarray) and payload.dtype == object and payload.shape == ():
        value = payload.item()
        if isinstance(value, dict) and "discrete" in value and "continuous" in value:
            return value["discrete"], value["continuous"]
        if isinstance(value, tuple) and len(value) == 2:
            return value[0], value[1]

    raise ValueError(
        f"{npy_path} must contain 'discrete' and 'continuous' arrays. "
        "Supported formats: structured arrays or pickled dict/tuple payloads."
    )


def load_discrete_array(npy_path: Path) -> np.ndarray:
    try:
        try:
            payload = np.load(npy_path, allow_pickle=False)
        except ValueError as exc:
            if "allow_pickle=False" not in str(exc):
                raise
            payload = np.load(npy_path, allow_pickle=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {npy_path}: {exc}") from exc

    discrete_raw, _ = _extract_modal_arrays(payload, npy_path)
    discrete = np.asarray(discrete_raw, dtype=np.int64)
    if discrete.ndim != 2:
        raise ValueError(f"Expected 2D discrete array [L, N] in {npy_path}, got {discrete.shape}.")
    return discrete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print one discrete stream from a Prism-TTS .npy file.")
    parser.add_argument(
        "--npy",
        type=Path,
        default=DEFAULT_NPY_PATH,
        help=f"Path to .npy file (default: {DEFAULT_NPY_PATH}).",
    )
    parser.add_argument(
        "--stream-index",
        type=int,
        default=0,
        help="0-based discrete stream index to print (default: 0).",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    cwd_resolved = (Path.cwd() / expanded).resolve()
    if cwd_resolved.is_file():
        return cwd_resolved
    return (PROJECT_ROOT / expanded).resolve()


def main() -> None:
    args = parse_args()
    npy_path = _resolve_path(args.npy)
    discrete = load_discrete_array(npy_path)

    stream_index = int(args.stream_index)
    if stream_index < 0 or stream_index >= int(discrete.shape[1]):
        raise IndexError(
            f"stream-index out of range: {stream_index} (available: 0..{int(discrete.shape[1]) - 1})"
        )

    stream = discrete[:, stream_index]
    print(stream.tolist())


if __name__ == "__main__":
    main()
