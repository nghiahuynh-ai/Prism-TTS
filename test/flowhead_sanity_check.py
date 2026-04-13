from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, MimiModel

# Avoid Matplotlib cache warnings triggered indirectly by optional Lightning imports.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mimi_latent_decoder import MimiPreUpsampleLatentDecoder
from utils import generate_utils
from utils.model_utils import normalize_discrete_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FlowHead-only sanity check: sample continuous latents from GT discrete tokens "
            "and compare against GT continuous latents from target audio."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a PrismTTS checkpoint (.ckpt or state_dict .pt).",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("config/model.yaml"),
        help="Path to model YAML config.",
    )
    parser.add_argument(
        "--target-audio",
        type=Path,
        required=True,
        help="Target audio used to extract GT discrete/continuous Mimi latents.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("synth/flowhead_sanity"),
        help="Directory to write output wavs and summary.",
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
        "--flow-num-steps",
        type=int,
        default=64,
        help="Override flow sampling steps.",
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
    parser.add_argument(
        "--mimi-model",
        type=str,
        default="kyutai/mimi",
        help="Hugging Face Mimi model id/path used for encode/decode.",
    )
    parser.add_argument(
        "--mimi-revision",
        type=str,
        default="main",
        help="Mimi model revision.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token for private models.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load Mimi assets from local cache/files.",
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


def _encode_audio_with_mimi(
    *,
    audio_path: Path,
    feature_extractor: Any,
    mimi_model: MimiModel,
    mimi_sample_rate: int,
    device: torch.device,
    num_quantizers: int,
) -> tuple[torch.LongTensor, torch.FloatTensor]:
    audio, sample_rate = generate_utils.read_wav(audio_path)
    audio = generate_utils.resample_if_needed(audio, sample_rate, mimi_sample_rate)

    features = feature_extractor(
        raw_audio=audio,
        sampling_rate=mimi_sample_rate,
        return_tensors="pt",
    )
    input_values = features["input_values"]
    if input_values.dim() == 2:
        input_values = input_values.unsqueeze(1)
    if input_values.dim() != 3:
        raise ValueError(f"Unexpected Mimi input shape: {tuple(input_values.shape)}")
    input_values = input_values.to(device=device)

    padding_mask = features.get("padding_mask")
    if padding_mask is not None:
        if padding_mask.dim() == 2:
            padding_mask = padding_mask.unsqueeze(1)
        if padding_mask.dim() != 3:
            raise ValueError(f"Unexpected Mimi padding_mask shape: {tuple(padding_mask.shape)}")
        padding_mask = padding_mask.to(device=device)

    with torch.no_grad():
        encoded = mimi_model.encode(
            input_values=input_values,
            padding_mask=padding_mask,
            num_quantizers=num_quantizers,
            return_dict=True,
        )
        codes = encoded.audio_codes
        if codes is None:
            raise RuntimeError("Mimi encode did not return audio_codes.")
        latents = mimi_model.quantizer.decode(codes)

    discrete = codes[0].transpose(0, 1).to(dtype=torch.long).cpu()  # [L, N]
    continuous = latents[0].transpose(0, 1).to(dtype=torch.float32).cpu()  # [L, C]
    return discrete, continuous


def _decode_latents_to_wav(
    *,
    decoder: MimiPreUpsampleLatentDecoder,
    latents: torch.FloatTensor,
    output_audio_path: Path,
) -> dict[str, Any]:
    with torch.no_grad():
        decoded = decoder(latents.detach())

    if isinstance(decoded, torch.Tensor):
        waveform = decoded.detach().cpu().float().numpy()
    else:
        waveform = np.asarray(decoded, dtype=np.float32)
    waveform = np.squeeze(waveform)
    if waveform.ndim > 1:
        waveform = waveform.reshape(-1)
    if waveform.size == 0:
        raise RuntimeError(f"Decoded waveform is empty for {output_audio_path.name}.")

    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform = waveform / peak

    sample_rate = int(decoder.sample_rate)
    generate_utils.write_wav(output_audio_path, waveform, sample_rate=sample_rate)
    mel_path = generate_utils.save_mel_spectrogram_plot(
        waveform=waveform,
        sample_rate=sample_rate,
        output_audio_path=output_audio_path,
    )
    return {
        "audio_path": str(output_audio_path),
        "mel_path": None if mel_path is None else str(mel_path),
        "sample_rate": sample_rate,
        "num_blocks": int(latents.shape[0]),
        "waveform_peak_before_norm": peak,
    }


def _compute_latent_metrics(
    *,
    sampled: torch.FloatTensor,
    target: torch.FloatTensor,
) -> dict[str, float]:
    if sampled.shape != target.shape:
        raise ValueError(
            f"Latent shape mismatch sampled={tuple(sampled.shape)} vs target={tuple(target.shape)}."
        )
    sampled_flat = sampled.reshape(-1)
    target_flat = target.reshape(-1)
    cosine = F.cosine_similarity(
        sampled_flat.unsqueeze(0),
        target_flat.unsqueeze(0),
        dim=-1,
    )
    return {
        "mse": float(F.mse_loss(sampled, target).item()),
        "mae": float(F.l1_loss(sampled, target).item()),
        "cosine_similarity": float(cosine.item()),
        "sampled_std": float(sampled.std(unbiased=False).item()),
        "target_std": float(target.std(unbiased=False).item()),
    }


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))

    device = generate_utils.resolve_device(args.device)
    model_dtype = generate_utils.resolve_torch_dtype(args.dtype)
    output_dir = _resolve_output_dir(args.output_dir)

    model_config = generate_utils.read_yaml(args.model_config)
    model = generate_utils.build_model(model_config)
    generate_utils.load_checkpoint(model, args.checkpoint, use_ema=bool(args.use_ema))
    model.to(device=device, dtype=model_dtype)
    model.eval()

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.mimi_model,
        revision=args.mimi_revision,
        token=args.hf_token,
        local_files_only=bool(args.local_files_only),
    )
    mimi_sample_rate = int(getattr(feature_extractor, "sampling_rate", 24_000))
    mimi_model = MimiModel.from_pretrained(
        args.mimi_model,
        revision=args.mimi_revision,
        token=args.hf_token,
        local_files_only=bool(args.local_files_only),
    )
    mimi_model.to(device=device)
    mimi_model.eval()

    gt_discrete_raw, gt_continuous_raw = _encode_audio_with_mimi(
        audio_path=args.target_audio,
        feature_extractor=feature_extractor,
        mimi_model=mimi_model,
        mimi_sample_rate=mimi_sample_rate,
        device=device,
        num_quantizers=int(model.num_discrete_tokens),
    )

    gt_discrete = gt_discrete_raw.unsqueeze(0).to(device=device, dtype=torch.long)
    with torch.no_grad():
        normalized_gt_discrete = normalize_discrete_tokens(
            gt_discrete,
            "groundtruth_discrete",
            num_discrete_tokens=model.num_discrete_tokens,
        )
        cond = model._discrete_condition(normalized_gt_discrete)
        sampled_latents = model.sample_continuous_latent(
            cond=cond,
            num_steps=args.flow_num_steps,
        )

    sampled_latents_cpu = sampled_latents[0].detach().to(dtype=torch.float32).cpu()
    gt_continuous_cpu = gt_continuous_raw.to(dtype=torch.float32)

    if sampled_latents_cpu.shape != gt_continuous_cpu.shape:
        min_len = min(int(sampled_latents_cpu.shape[0]), int(gt_continuous_cpu.shape[0]))
        sampled_latents_cpu = sampled_latents_cpu[:min_len]
        gt_continuous_cpu = gt_continuous_cpu[:min_len]

    metrics = _compute_latent_metrics(
        sampled=sampled_latents_cpu,
        target=gt_continuous_cpu,
    )

    decoder = MimiPreUpsampleLatentDecoder(
        pretrained_model_name_or_path=args.mimi_model,
        device=str(device),
        dtype=args.dtype,
        local_files_only=bool(args.local_files_only),
        revision=args.mimi_revision,
        token=args.hf_token,
    )
    sampled_path = output_dir / "flow_on_gt_discrete.wav"
    gt_path = output_dir / "groundtruth_continuous.wav"

    sampled_decode = _decode_latents_to_wav(
        decoder=decoder,
        latents=sampled_latents_cpu,
        output_audio_path=sampled_path,
    )
    gt_decode = _decode_latents_to_wav(
        decoder=decoder,
        latents=gt_continuous_cpu,
        output_audio_path=gt_path,
    )

    summary = {
        "input": {
            "target_audio": str(args.target_audio),
        },
        "runtime": {
            "device": str(device),
            "dtype": args.dtype,
            "seed": args.seed,
            "flow_num_steps": args.flow_num_steps,
            "num_discrete_streams": int(model.num_discrete_tokens),
            "continuous_latent_size": int(model.continuous_latent_size),
        },
        "lengths": {
            "target_raw_frames": int(gt_discrete_raw.shape[0]),
            "evaluation_frames": int(sampled_latents_cpu.shape[0]),
        },
        "latent_metrics": metrics,
        "paths": {
            "output_dir": str(output_dir),
            "flow_on_gt_discrete": sampled_decode,
            "groundtruth_continuous": gt_decode,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print("[flowhead_sanity_check] completed")
    print(f"[flowhead_sanity_check] output_dir={output_dir}")
    print(
        "[flowhead_sanity_check] latent metrics: "
        f"mse={metrics['mse']:.6f}, mae={metrics['mae']:.6f}, "
        f"cosine={metrics['cosine_similarity']:.6f}"
    )
    print(f"[flowhead_sanity_check] wrote sampled wav: {sampled_decode['audio_path']}")
    print(f"[flowhead_sanity_check] wrote gt wav: {gt_decode['audio_path']}")
    print(f"[flowhead_sanity_check] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
