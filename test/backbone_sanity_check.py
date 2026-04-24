from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Avoid Matplotlib cache warnings triggered indirectly by optional Lightning imports.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import build_shared_token_layout
from utils import generate_utils
from utils import tokenizer_utils as TU


@dataclass(frozen=True)
class MetadataSample:
    target_audio_relpath: str
    target_duration: float
    target_transcript: str
    target_npy_path: Path
    prompt_audio_relpath: str
    prompt_duration: float
    prompt_transcript: str
    prompt_npy_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backbone sanity check: generate discrete tokens from prompt + target text "
            "and compare them with metadata target discrete tokens."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a PrismTTS checkpoint (.ckpt or state_dict .pt).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("samples/metadata.txt"),
        help="Manifest-style metadata file with prompt/target pair entries.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="0-based metadata sample index.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("config/model.yaml"),
        help="Path to model YAML config.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("config/data.yaml"),
        help="Path to data YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("synth/backbone_sanity"),
        help="Directory to write summary and token dumps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device (e.g. 'cpu', 'cuda:0', or 'auto').",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=("float32", "float16", "bfloat16"),
        help="PrismTTS model dtype.",
    )
    parser.add_argument(
        "--max-new-blocks",
        type=int,
        default=None,
        help="Override generated block length. Defaults to aligned GT target length.",
    )
    parser.add_argument(
        "--flow-num-steps",
        type=int,
        default=1,
        help="Flow sampling steps during generation (1 is fastest for backbone-only checks).",
    )
    parser.add_argument(
        "--preview-blocks",
        type=int,
        default=32,
        help="How many leading blocks to print into summary preview.",
    )
    parser.add_argument(
        "--mismatch-limit",
        type=int,
        default=32,
        help="Max mismatch blocks to include in summary JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed.",
    )
    parser.add_argument(
        "--use-ema",
        dest="use_ema",
        action="store_true",
        default=True,
        help="Use checkpoint EMA weights when available.",
    )
    parser.add_argument(
        "--no-ema",
        dest="use_ema",
        action="store_false",
        help="Disable EMA and load regular checkpoint weights.",
    )
    return parser.parse_args()


def _resolve_output_dir(path: Path) -> Path:
    output_dir = path.expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_metadata_feature_path(metadata_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    candidate_paths: list[Path] = []

    if path.is_absolute():
        candidate_paths.append(path)
    else:
        candidate_paths.append(metadata_path.parent / path)
        candidate_paths.append(Path.cwd() / path)
        candidate_paths.append(metadata_path.parent / path.name)
        candidate_paths.append(Path.cwd() / path.name)
        candidate_paths.append(PROJECT_ROOT / "samples" / path.name)

    for candidate in candidate_paths:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved

    joined = "\n".join(f"- {candidate.resolve()}" for candidate in candidate_paths)
    raise FileNotFoundError(
        "Unable to resolve metadata feature path "
        f"{raw_path!r} from {metadata_path}. Tried:\n{joined}"
    )


def _parse_metadata_line(
    metadata_path: Path,
    line: str,
    line_number: int,
) -> MetadataSample:
    parts = [part.strip() for part in line.split("|")]
    while parts and parts[-1] == "":
        parts.pop()
    if len(parts) != 8:
        raise ValueError(
            "Each metadata line must have 8 fields separated by '|'. "
            f"line={line_number}, fields={len(parts)}"
        )

    (
        target_audio_relpath,
        target_duration_raw,
        target_transcript,
        target_npy_raw,
        prompt_audio_relpath,
        prompt_duration_raw,
        prompt_transcript,
        prompt_npy_raw,
    ) = parts

    try:
        target_duration = float(target_duration_raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid target duration at line {line_number}: {target_duration_raw!r}."
        ) from exc

    try:
        prompt_duration = float(prompt_duration_raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid prompt duration at line {line_number}: {prompt_duration_raw!r}."
        ) from exc

    return MetadataSample(
        target_audio_relpath=target_audio_relpath,
        target_duration=target_duration,
        target_transcript=target_transcript,
        target_npy_path=_resolve_metadata_feature_path(metadata_path, target_npy_raw),
        prompt_audio_relpath=prompt_audio_relpath,
        prompt_duration=prompt_duration,
        prompt_transcript=prompt_transcript,
        prompt_npy_path=_resolve_metadata_feature_path(metadata_path, prompt_npy_raw),
    )


def _load_metadata_sample(metadata_path: Path, sample_index: int) -> MetadataSample:
    resolved_metadata = metadata_path.expanduser()
    if not resolved_metadata.is_absolute():
        resolved_metadata = (Path.cwd() / resolved_metadata).resolve()
    else:
        resolved_metadata = resolved_metadata.resolve()
    if not resolved_metadata.is_file():
        raise FileNotFoundError(f"metadata file not found: {resolved_metadata}")

    lines: list[tuple[int, str]] = []
    with resolved_metadata.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append((line_number, line))
    if not lines:
        raise ValueError(f"No valid metadata rows in {resolved_metadata}.")
    if sample_index < 0 or sample_index >= len(lines):
        raise IndexError(
            f"sample-index {sample_index} out of range for {resolved_metadata} "
            f"(num_rows={len(lines)})."
        )

    line_number, line = lines[sample_index]
    return _parse_metadata_line(
        metadata_path=resolved_metadata,
        line=line,
        line_number=line_number,
    )


def _extract_modal_arrays_from_npy(
    payload: Any,
    npy_path: Path,
) -> tuple[Any, Any]:
    if isinstance(payload, np.ndarray) and payload.dtype.names is not None:
        field_names = set(payload.dtype.names)
        if "discrete" in field_names and "continuous" in field_names:
            discrete = payload["discrete"]
            continuous = payload["continuous"]
            if (
                isinstance(discrete, np.ndarray)
                and discrete.dtype == object
                and discrete.shape == ()
            ):
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
        if isinstance(value, dict):
            if "discrete" in value and "continuous" in value:
                return value["discrete"], value["continuous"]
        if isinstance(value, tuple) and len(value) == 2:
            return value[0], value[1]

    raise ValueError(
        f"{npy_path} must contain 'discrete' and 'continuous' arrays. "
        "Supported formats: structured arrays or pickled dict/tuple payloads."
    )


def _load_npy_features(npy_path: Path) -> tuple[torch.LongTensor, torch.FloatTensor]:
    try:
        try:
            payload = np.load(npy_path, allow_pickle=False)
        except ValueError as exc:
            if "allow_pickle=False" not in str(exc):
                raise
            payload = np.load(npy_path, allow_pickle=True)
        discrete_raw, continuous_raw = _extract_modal_arrays_from_npy(payload, npy_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load npy {npy_path}: {exc}") from exc

    discrete = torch.as_tensor(np.asarray(discrete_raw), dtype=torch.long)
    continuous = torch.as_tensor(np.asarray(continuous_raw), dtype=torch.float32)
    if discrete.dim() != 2:
        raise ValueError(f"Expected 2D discrete [L, N] in {npy_path}, got {tuple(discrete.shape)}.")
    if continuous.dim() != 2:
        raise ValueError(
            f"Expected 2D continuous [L, C] in {npy_path}, got {tuple(continuous.shape)}."
        )
    if discrete.shape[0] != continuous.shape[0]:
        raise ValueError(
            f"Length mismatch in {npy_path}: discrete L={discrete.shape[0]}, "
            f"continuous L={continuous.shape[0]}."
        )
    return discrete.contiguous(), continuous.contiguous()


def _fit_modalities_to_model(
    *,
    discrete: torch.LongTensor,
    continuous: torch.FloatTensor,
    num_discrete_tokens: int,
    continuous_latent_size: int,
    source_label: str,
) -> tuple[torch.LongTensor, torch.FloatTensor]:
    if discrete.shape[1] < num_discrete_tokens:
        raise ValueError(
            f"{source_label} has {discrete.shape[1]} discrete streams, "
            f"but model expects {num_discrete_tokens}."
        )
    if continuous.shape[1] < continuous_latent_size:
        raise ValueError(
            f"{source_label} has latent dim={continuous.shape[1]}, "
            f"but model expects {continuous_latent_size}."
        )
    return (
        discrete[:, :num_discrete_tokens].contiguous(),
        continuous[:, :continuous_latent_size].contiguous(),
    )


def _pad_or_trim_discrete_to_length(
    discrete: torch.LongTensor,
    target_length: int,
    pad_value: int,
) -> torch.LongTensor:
    if target_length < 1:
        raise ValueError("target_length must be >= 1.")
    current_length = int(discrete.shape[0])
    if current_length == target_length:
        return discrete
    if current_length > target_length:
        return discrete[:target_length].contiguous()

    pad = torch.full(
        (target_length - current_length, discrete.shape[1]),
        fill_value=int(pad_value),
        dtype=discrete.dtype,
    )
    return torch.cat([discrete, pad], dim=0).contiguous()


def _compute_discrete_metrics(
    *,
    predicted: torch.LongTensor,
    groundtruth: torch.LongTensor,
    special_token_ids: tuple[int, ...],
    mismatch_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if predicted.dim() != 2 or groundtruth.dim() != 2:
        raise ValueError("predicted and groundtruth must be 2D [L, N].")
    if predicted.shape[1] != groundtruth.shape[1]:
        raise ValueError(
            f"Discrete stream count mismatch: pred={predicted.shape[1]} vs gt={groundtruth.shape[1]}."
        )

    pred_length = int(predicted.shape[0])
    gt_length = int(groundtruth.shape[0])
    common_length = min(pred_length, gt_length)
    if common_length < 1:
        return (
            {
                "pred_length": pred_length,
                "gt_length": gt_length,
                "common_length": common_length,
                "length_delta_pred_minus_gt": pred_length - gt_length,
                "token_accuracy": None,
                "block_accuracy": None,
                "token_accuracy_non_special_gt": None,
                "block_accuracy_non_special_gt": None,
                "exact_match": bool(pred_length == gt_length),
                "prefix_match_blocks": 0,
                "first_mismatch_block": None,
                "pred_special_ratio": None,
                "gt_special_ratio": None,
            },
            [],
        )

    pred_common = predicted[:common_length]
    gt_common = groundtruth[:common_length]
    token_equal = pred_common.eq(gt_common)
    block_equal = token_equal.all(dim=-1)

    prefix_match_blocks = 0
    for is_equal in block_equal.tolist():
        if bool(is_equal):
            prefix_match_blocks += 1
        else:
            break

    mismatch_indices = torch.nonzero(~block_equal, as_tuple=False).squeeze(-1)
    mismatch_examples: list[dict[str, Any]] = []
    for mismatch_idx in mismatch_indices[: max(0, int(mismatch_limit))]:
        block_idx = int(mismatch_idx.item())
        mismatch_examples.append(
            {
                "block_index": block_idx,
                "predicted": pred_common[block_idx].tolist(),
                "groundtruth": gt_common[block_idx].tolist(),
            }
        )

    special_ids_tensor = torch.tensor(
        list(special_token_ids),
        dtype=gt_common.dtype,
        device=gt_common.device,
    )
    gt_is_special = torch.isin(gt_common, special_ids_tensor)
    pred_is_special = torch.isin(pred_common, special_ids_tensor)
    gt_non_special = ~gt_is_special

    token_accuracy_non_special = None
    if bool(gt_non_special.any().item()):
        token_accuracy_non_special = float(
            token_equal[gt_non_special].float().mean().item()
        )

    gt_non_special_blocks = gt_non_special.any(dim=-1)
    block_accuracy_non_special = None
    if bool(gt_non_special_blocks.any().item()):
        block_accuracy_non_special = float(
            block_equal[gt_non_special_blocks].float().mean().item()
        )

    exact_match = bool(pred_length == gt_length) and bool(token_equal.all().item())

    return (
        {
            "pred_length": pred_length,
            "gt_length": gt_length,
            "common_length": common_length,
            "length_delta_pred_minus_gt": pred_length - gt_length,
            "token_accuracy": float(token_equal.float().mean().item()),
            "block_accuracy": float(block_equal.float().mean().item()),
            "token_accuracy_non_special_gt": token_accuracy_non_special,
            "block_accuracy_non_special_gt": block_accuracy_non_special,
            "exact_match": exact_match,
            "prefix_match_blocks": int(prefix_match_blocks),
            "first_mismatch_block": (
                None if mismatch_indices.numel() == 0 else int(mismatch_indices[0].item())
            ),
            "pred_special_ratio": float(pred_is_special.float().mean().item()),
            "gt_special_ratio": float(gt_is_special.float().mean().item()),
        },
        mismatch_examples,
    )


def _discrete_tokens_to_printable(discrete_tokens: torch.LongTensor) -> list[int] | list[list[int]]:
    if discrete_tokens.dim() != 2:
        raise ValueError(f"Expected 2D [L, N], got {tuple(discrete_tokens.shape)}.")
    if discrete_tokens.shape[1] < 1:
        raise ValueError("Expected at least one discrete stream.")
    discrete_cpu = discrete_tokens.to(dtype=torch.long).cpu()
    if discrete_cpu.shape[1] == 1:
        return discrete_cpu[:, 0].tolist()
    return discrete_cpu.tolist()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))

    device = generate_utils.resolve_device(args.device)
    model_dtype = generate_utils.resolve_torch_dtype(args.dtype)
    output_dir = _resolve_output_dir(args.output_dir)

    model_config = generate_utils.read_yaml(args.model_config)
    data_config = generate_utils.read_yaml(args.data_config)

    model = generate_utils.build_model(model_config)
    generate_utils.load_checkpoint(model, args.checkpoint, use_ema=bool(args.use_ema))
    model.to(device=device, dtype=model_dtype)
    model.eval()

    sample = _load_metadata_sample(args.metadata, int(args.sample_index))
    prompt_discrete_raw, prompt_continuous_raw = _load_npy_features(sample.prompt_npy_path)
    target_discrete_raw, target_continuous_raw = _load_npy_features(sample.target_npy_path)

    prompt_discrete, prompt_continuous = _fit_modalities_to_model(
        discrete=prompt_discrete_raw,
        continuous=prompt_continuous_raw,
        num_discrete_tokens=int(model.num_discrete_tokens),
        continuous_latent_size=int(model.continuous_latent_size),
        source_label="prompt_npy",
    )
    target_discrete, target_continuous = _fit_modalities_to_model(
        discrete=target_discrete_raw,
        continuous=target_continuous_raw,
        num_discrete_tokens=int(model.num_discrete_tokens),
        continuous_latent_size=int(model.continuous_latent_size),
        source_label="target_npy",
    )

    data_cfg = generate_utils.require_mapping(data_config, "data")
    shared_layout_cfg = generate_utils.require_mapping(data_cfg, "shared_layout")
    dataset_cfg = generate_utils.require_mapping(data_cfg, "dataset")

    discrete_token_count = int(shared_layout_cfg["discrete_token_count"])
    _, eos_token_id, pad_token_id, _ = build_shared_token_layout(
        discrete_token_count
    )
    text_pad_value = int(pad_token_id)
    discrete_pad_value = int(pad_token_id)

    tokenizer, resolved_default_tokenizer = TU.build_generation_text_tokenizer(
        cached_default_text_tokenizer=getattr(model, "_default_text_tokenizer", None),
        use_separate_codec_embedding=bool(getattr(model, "use_separate_codec_embedding", False)),
        backbone_name=getattr(model, "backbone_name"),
        backbone_hf_checkpoint=getattr(model, "backbone_hf_checkpoint"),
        backbone_hf_kwargs=getattr(model, "backbone_hf_kwargs"),
        discrete_vocab_size=int(getattr(model, "discrete_vocab_size")),
        eot_token_id=int(getattr(model, "eot_token_id")),
        eos_token_id=int(getattr(model, "eos_token_id")),
        pad_token_id=int(getattr(model, "pad_token_id")),
        append_eos_to_text=bool(dataset_cfg.get("append_eos_to_text", False)),
        vocab_path=data_cfg.get("vocab_path", "dataset/vocab.txt"),
    )
    if hasattr(model, "_default_text_tokenizer"):
        model._default_text_tokenizer = resolved_default_tokenizer

    prompt_text_tokens = generate_utils.safe_tokenize(
        tokenizer,
        sample.prompt_transcript,
        "prompt_transcript",
    )
    target_text_tokens = generate_utils.safe_tokenize(
        tokenizer,
        sample.target_transcript,
        "target_transcript",
    )
    if target_text_tokens.numel() == 0:
        raise ValueError("Target transcript became empty after tokenization.")

    aligned_text_prompt = prompt_text_tokens
    aligned_discrete_prompt = prompt_discrete
    aligned_continuous_prompt = prompt_continuous
    aligned_discrete_target = target_discrete

    max_new_blocks = args.max_new_blocks
    if max_new_blocks is None:
        max_new_blocks = int(aligned_discrete_target.shape[0])
    if int(max_new_blocks) < 1:
        raise ValueError("--max-new-blocks must be >= 1.")
    max_new_blocks = int(max_new_blocks)

    text_target = target_text_tokens.unsqueeze(0)

    text_prompt_tensor = aligned_text_prompt.unsqueeze(0).to(device=device, dtype=torch.long)
    discrete_prompt_tensor = aligned_discrete_prompt.transpose(0, 1).unsqueeze(0).to(
        device=device,
        dtype=torch.long,
    )
    continuous_prompt_tensor = aligned_continuous_prompt.unsqueeze(0).to(
        device=device,
        dtype=model_dtype,
    )
    target_lengths = torch.tensor([text_target.shape[1]], dtype=torch.long, device=device)

    with torch.no_grad():
        generation = model.generate(
            text_prompt=text_prompt_tensor,
            discrete_prompt=discrete_prompt_tensor,
            continuous_prompt=continuous_prompt_tensor,
            text_target=text_target.to(device=device, dtype=torch.long),
            text_prompt_lengths=torch.tensor(
                [text_prompt_tensor.shape[1]],
                dtype=torch.long,
                device=device,
            ),
            speech_prompt_lengths=torch.tensor(
                [aligned_discrete_prompt.shape[0]],
                dtype=torch.long,
                device=device,
            ),
            text_target_lengths=torch.tensor(
                [text_target.shape[1]],
                dtype=torch.long,
                device=device,
            ),
            speech_target_lengths=target_lengths,
            max_new_blocks=max_new_blocks,
            discrete_eos_token_id=eos_token_id,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            do_sample=False,
            flow_num_steps=int(args.flow_num_steps),
            force_silent_special_tokens=True,
            return_dict=True,
        )

    if generation.discrete_ids is None or generation.discrete_ids.shape[0] == 0:
        raise RuntimeError("Model generation returned no discrete ids.")

    predicted_discrete = (
        generation.discrete_ids[0].detach().to(dtype=torch.long).cpu().transpose(0, 1).contiguous()
    )
    groundtruth_discrete = _pad_or_trim_discrete_to_length(
        aligned_discrete_target.to(dtype=torch.long).cpu(),
        target_length=max_new_blocks,
        pad_value=discrete_pad_value,
    )
    special_token_ids = (eos_token_id, pad_token_id)
    metrics, mismatch_examples = _compute_discrete_metrics(
        predicted=predicted_discrete,
        groundtruth=groundtruth_discrete,
        special_token_ids=special_token_ids,
        mismatch_limit=int(args.mismatch_limit),
    )

    generated_path = output_dir / "generated_discrete.npy"
    groundtruth_path = output_dir / "groundtruth_discrete.npy"
    np.save(generated_path, predicted_discrete.numpy().astype(np.int32))
    np.save(groundtruth_path, groundtruth_discrete.numpy().astype(np.int32))

    preview_blocks = max(1, int(args.preview_blocks))
    preview_length = min(
        preview_blocks,
        int(predicted_discrete.shape[0]),
        int(groundtruth_discrete.shape[0]),
    )

    summary = {
        "input": {
            "checkpoint": str(args.checkpoint),
            "metadata": str(args.metadata),
            "sample_index": int(args.sample_index),
            "target_audio_relpath": sample.target_audio_relpath,
            "target_duration": float(sample.target_duration),
            "target_transcript": sample.target_transcript,
            "target_npy_path": str(sample.target_npy_path),
            "prompt_audio_relpath": sample.prompt_audio_relpath,
            "prompt_duration": float(sample.prompt_duration),
            "prompt_transcript": sample.prompt_transcript,
            "prompt_npy_path": str(sample.prompt_npy_path),
        },
        "runtime": {
            "device": str(device),
            "dtype": args.dtype,
            "seed": args.seed,
            "use_ema": bool(args.use_ema),
            "max_new_blocks": int(max_new_blocks),
            "flow_num_steps": int(args.flow_num_steps),
            "num_discrete_tokens": int(model.num_discrete_tokens),
            "evaluated_discrete_streams": int(predicted_discrete.shape[1]),
            "continuous_latent_size": int(model.continuous_latent_size),
        },
        "lengths": {
            "prompt_raw_blocks": int(prompt_discrete.shape[0]),
            "prompt_aligned_blocks": int(aligned_discrete_prompt.shape[0]),
            "target_raw_blocks": int(target_discrete.shape[0]),
            "target_aligned_blocks": int(aligned_discrete_target.shape[0]),
            "generated_blocks": int(predicted_discrete.shape[0]),
            "evaluation_target_blocks": int(groundtruth_discrete.shape[0]),
        },
        "special_token_ids": {
            "eos": int(eos_token_id),
            "pad": int(pad_token_id),
        },
        "metrics": metrics,
        "preview": {
            "num_blocks": int(preview_length),
            "generated_discrete": predicted_discrete[:preview_length].tolist(),
            "groundtruth_discrete": groundtruth_discrete[:preview_length].tolist(),
        },
        "generated_discrete": _discrete_tokens_to_printable(predicted_discrete),
        "groundtruth_discrete": _discrete_tokens_to_printable(groundtruth_discrete),
        "mismatch_examples": mismatch_examples,
        "paths": {
            "output_dir": str(output_dir),
            "generated_discrete_npy": str(generated_path),
            "groundtruth_discrete_npy": str(groundtruth_path),
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    token_acc = metrics["token_accuracy"]
    block_acc = metrics["block_accuracy"]
    token_acc_display = "n/a" if token_acc is None else f"{float(token_acc):.6f}"
    block_acc_display = "n/a" if block_acc is None else f"{float(block_acc):.6f}"
    print("[backbone_sanity_check] completed")
    print(f"[backbone_sanity_check] output_dir={output_dir}")
    print(
        "[backbone_sanity_check] match metrics: "
        f"token_acc={token_acc_display}, block_acc={block_acc_display}, "
        f"exact_match={metrics['exact_match']}"
    )
    print(
        "[backbone_sanity_check] generated discrete tokens: "
        f"{_discrete_tokens_to_printable(predicted_discrete)}"
    )
    print(
        "[backbone_sanity_check] groundtruth discrete tokens: "
        f"{_discrete_tokens_to_printable(groundtruth_discrete)}"
    )
    print(f"[backbone_sanity_check] wrote summary: {summary_path}")
    print(f"[backbone_sanity_check] wrote generated discrete: {generated_path}")
    print(f"[backbone_sanity_check] wrote groundtruth discrete: {groundtruth_path}")


if __name__ == "__main__":
    main()
