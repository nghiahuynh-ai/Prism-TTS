from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "prism_tts_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from dataset.dataset import BatchCollate, build_shared_token_layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic batch, run BatchCollate, and save visualizations "
            "of collated text/discrete/continuous/attention streams."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "batch_collate",
        help="Directory where images will be written.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="batch_collate",
        help="Prefix for generated filenames.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Number of synthetic samples to collate.",
    )
    parser.add_argument(
        "--num-discrete-streams",
        type=int,
        default=2,
        help="Number of discrete streams in each synthetic sample.",
    )
    parser.add_argument(
        "--continuous-dim",
        type=int,
        default=4,
        help="Channel size for continuous latents.",
    )
    parser.add_argument(
        "--discrete-token-count",
        type=int,
        default=128,
        help="Number of non-special discrete ids.",
    )
    parser.add_argument(
        "--discrete-stream-delay-ms",
        type=float,
        default=80.0,
        help="Shared delay in ms for all discrete streams and the continuous stream.",
    )
    parser.add_argument(
        "--codec-frame-rate-hz",
        type=float,
        default=12.5,
        help="Codec frame rate used to convert delay ms to frame delays.",
    )
    parser.add_argument(
        "--min-text-len",
        type=int,
        default=2,
        help="Minimum text token length for prompt/target.",
    )
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=8,
        help="Maximum text token length for prompt/target.",
    )
    parser.add_argument(
        "--min-frame-len",
        type=int,
        default=2,
        help="Minimum frame length for discrete/continuous prompt/target.",
    )
    parser.add_argument(
        "--max-frame-len",
        type=int,
        default=8,
        help="Maximum frame length for discrete/continuous prompt/target.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base seed for reproducible synthetic samples.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Image DPI.",
    )
    return parser


def _random_length(
    min_len: int,
    max_len: int,
    generator: torch.Generator,
) -> int:
    if min_len < 1:
        raise ValueError("Minimum length must be >= 1.")
    if max_len < min_len:
        raise ValueError("Maximum length must be >= minimum length.")
    if min_len == max_len:
        return min_len
    return int(torch.randint(min_len, max_len + 1, size=(1,), generator=generator).item())


def _make_sample(
    sample_index: int,
    *,
    discrete_token_count: int,
    num_discrete_streams: int,
    continuous_dim: int,
    min_text_len: int,
    max_text_len: int,
    min_frame_len: int,
    max_frame_len: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    _, _, _, text_token_offset = build_shared_token_layout(discrete_token_count)
    generator = torch.Generator().manual_seed(seed + sample_index)

    text_prompt_len = _random_length(min_text_len, max_text_len, generator)
    text_target_len = _random_length(min_text_len, max_text_len, generator)
    frame_prompt_len = _random_length(min_frame_len, max_frame_len, generator)
    frame_target_len = _random_length(min_frame_len, max_frame_len, generator)

    text_span = 256
    text_prompt = torch.randint(
        low=text_token_offset,
        high=text_token_offset + text_span,
        size=(text_prompt_len,),
        dtype=torch.long,
        generator=generator,
    )
    text_target = torch.randint(
        low=text_token_offset,
        high=text_token_offset + text_span,
        size=(text_target_len,),
        dtype=torch.long,
        generator=generator,
    )

    discrete_prompt = torch.randint(
        low=0,
        high=discrete_token_count,
        size=(frame_prompt_len, num_discrete_streams),
        dtype=torch.long,
        generator=generator,
    )
    discrete_target = torch.randint(
        low=0,
        high=discrete_token_count,
        size=(frame_target_len, num_discrete_streams),
        dtype=torch.long,
        generator=generator,
    )

    continuous_prompt = torch.randn(
        frame_prompt_len,
        continuous_dim,
        generator=generator,
        dtype=torch.float32,
    ) + float(sample_index) * 0.25
    continuous_target = torch.randn(
        frame_target_len,
        continuous_dim,
        generator=generator,
        dtype=torch.float32,
    ) + float(sample_index) * 0.25

    return {
        "text_prompt": text_prompt,
        "discrete_prompt": discrete_prompt,
        "continuous_prompt": continuous_prompt,
        "text_target": text_target,
        "discrete_target": discrete_target,
        "continuous_target": continuous_target,
    }


def _draw_boundaries(ax: Any, prompt_len: int, valid_len: int) -> None:
    prompt_boundary = prompt_len - 0.5
    valid_boundary = valid_len - 0.5
    ax.axvline(prompt_boundary, color="#0b3954", linewidth=1.5, linestyle="--")
    ax.axvline(valid_boundary, color="#d7263d", linewidth=1.5, linestyle="--")


def _save_sample_figure(
    batch: dict[str, torch.Tensor],
    sample_idx: int,
    output_path: Path,
    *,
    delay_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    dpi: int,
) -> None:
    text = batch["text"][sample_idx].cpu()
    discrete = batch["discrete"][sample_idx].transpose(0, 1).cpu()
    continuous = batch["continuous"][sample_idx].transpose(0, 1).cpu()
    attention = batch["attention_mask"][sample_idx].to(dtype=torch.int32).unsqueeze(0).cpu()

    prompt_len = int(batch["prompt_lengths"][sample_idx].item())
    target_len = int(batch["target_lengths"][sample_idx].item())
    valid_len = prompt_len + target_len

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [1.0, 2.0, 2.0, 1.0]},
    )

    im_text = axes[0].imshow(
        text.unsqueeze(0).numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    axes[0].set_ylabel("text")
    axes[0].set_yticks([0])
    axes[0].set_title("Text IDs")
    fig.colorbar(im_text, ax=axes[0], fraction=0.03, pad=0.01)

    im_discrete = axes[1].imshow(
        discrete.numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
    )
    axes[1].set_ylabel("streams")
    axes[1].set_yticks(list(range(discrete.shape[0])))
    axes[1].set_yticklabels([f"D{i}" for i in range(discrete.shape[0])])
    axes[1].set_title(
        f"Discrete IDs (delay={delay_token_id}, eos={eos_token_id}, pad={pad_token_id})"
    )
    fig.colorbar(im_discrete, ax=axes[1], fraction=0.03, pad=0.01)

    im_cont = axes[2].imshow(
        continuous.numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
    )
    axes[2].set_ylabel("channels")
    axes[2].set_yticks(list(range(continuous.shape[0])))
    axes[2].set_yticklabels([f"C{i}" for i in range(continuous.shape[0])])
    axes[2].set_title("Continuous Latents")
    fig.colorbar(im_cont, ax=axes[2], fraction=0.03, pad=0.01)

    im_attn = axes[3].imshow(
        attention.numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#f1efe9", "#0b3954"]),
        vmin=0,
        vmax=1,
    )
    axes[3].set_ylabel("mask")
    axes[3].set_yticks([0])
    axes[3].set_yticklabels(["attn"])
    axes[3].set_title("Attention Mask (1=valid, 0=pad)")
    fig.colorbar(im_attn, ax=axes[3], fraction=0.03, pad=0.01)

    for ax in axes:
        _draw_boundaries(ax, prompt_len=prompt_len, valid_len=valid_len)
        ax.set_xlim(-0.5, float(text.shape[0]) - 0.5)
    axes[-1].set_xlabel("time steps")

    fig.suptitle(
        f"Sample {sample_idx} | prompt_len={prompt_len}, target_len={target_len}, total={text.shape[0]}",
        fontsize=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_overview_figure(
    batch: dict[str, torch.Tensor],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    text = batch["text"].cpu()
    attention_mask = batch["attention_mask"].to(dtype=torch.int32).cpu()
    prompt_lengths = batch["prompt_lengths"].cpu()
    target_lengths = batch["target_lengths"].cpu()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1, 1]})

    im_text = axes[0].imshow(text.numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
    axes[0].set_title("Batch Text IDs [B, L]")
    axes[0].set_ylabel("sample")
    fig.colorbar(im_text, ax=axes[0], fraction=0.03, pad=0.01)

    im_mask = axes[1].imshow(
        attention_mask.numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#f1efe9", "#0b3954"]),
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Batch Attention Mask [B, L]")
    axes[1].set_ylabel("sample")
    fig.colorbar(im_mask, ax=axes[1], fraction=0.03, pad=0.01)

    sample_idx = torch.arange(text.shape[0], dtype=torch.long)
    axes[2].bar(sample_idx.numpy() - 0.15, prompt_lengths.numpy(), width=0.3, label="prompt")
    axes[2].bar(sample_idx.numpy() + 0.15, target_lengths.numpy(), width=0.3, label="target")
    axes[2].set_title("Prompt/Target Lengths")
    axes[2].set_xlabel("sample")
    axes[2].set_ylabel("length")
    axes[2].set_xticks(sample_idx.numpy())
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _validate_collated_batch(batch: dict[str, torch.Tensor]) -> None:
    batch_size, text_len = batch["text"].shape
    if batch["discrete"].shape[:2] != (batch_size, text_len):
        raise RuntimeError("discrete shape does not align with text shape.")
    if batch["continuous"].shape[:2] != (batch_size, text_len):
        raise RuntimeError("continuous shape does not align with text shape.")
    if batch["attention_mask"].shape != (batch_size, text_len):
        raise RuntimeError("attention_mask shape does not align with text shape.")

    for idx in range(batch_size):
        prompt_len = int(batch["prompt_lengths"][idx].item())
        target_len = int(batch["target_lengths"][idx].item())
        valid_len = prompt_len + target_len
        if valid_len > text_len:
            raise RuntimeError(
                f"sample {idx}: prompt_len + target_len ({valid_len}) exceeds padded length {text_len}."
            )
        actual_valid = int(batch["attention_mask"][idx].sum().item())
        if actual_valid != valid_len:
            raise RuntimeError(
                f"sample {idx}: attention_mask valid count ({actual_valid}) != expected {valid_len}."
            )


def _print_batch_summary(batch: dict[str, torch.Tensor]) -> None:
    print("Collated keys and shapes:")
    for key in sorted(batch.keys()):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
    print("Per-sample lengths:")
    for idx in range(int(batch["text"].shape[0])):
        prompt_len = int(batch["prompt_lengths"][idx].item())
        target_len = int(batch["target_lengths"][idx].item())
        print(f"  - sample {idx}: prompt_len={prompt_len}, target_len={target_len}")


def main() -> None:
    args = build_parser().parse_args()
    delay_ms = float(args.discrete_stream_delay_ms)

    delay_token_id, eos_token_id, pad_token_id, _ = build_shared_token_layout(
        args.discrete_token_count
    )

    samples = [
        _make_sample(
            sample_index=sample_idx,
            discrete_token_count=args.discrete_token_count,
            num_discrete_streams=args.num_discrete_streams,
            continuous_dim=args.continuous_dim,
            min_text_len=args.min_text_len,
            max_text_len=args.max_text_len,
            min_frame_len=args.min_frame_len,
            max_frame_len=args.max_frame_len,
            seed=args.seed,
        )
        for sample_idx in range(args.batch_size)
    ]

    collate = BatchCollate(
        discrete_token_count=args.discrete_token_count,
        discrete_stream_delay_ms=delay_ms,
        codec_frame_rate_hz=args.codec_frame_rate_hz,
    )
    batch = collate(samples)
    _validate_collated_batch(batch)
    _print_batch_summary(batch)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = args.output_dir / f"{args.output_prefix}_overview.png"
    _save_overview_figure(batch, overview_path, dpi=args.dpi)
    print(f"Saved overview image: {overview_path}")

    for sample_idx in range(args.batch_size):
        output_path = args.output_dir / f"{args.output_prefix}_sample_{sample_idx}.png"
        _save_sample_figure(
            batch=batch,
            sample_idx=sample_idx,
            output_path=output_path,
            delay_token_id=delay_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            dpi=args.dpi,
        )
        print(f"Saved sample image:   {output_path}")


if __name__ == "__main__":
    main()
