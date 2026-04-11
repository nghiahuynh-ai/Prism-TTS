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
            "for split text/speech prompt/target tensors."
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


def _plot_row(
    ax: Any,
    values: torch.Tensor,
    *,
    title: str,
    ylabel: str,
    cmap: str,
) -> None:
    image = ax.imshow(
        values.numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.colorbar(image, ax=ax, fraction=0.03, pad=0.01)


def _save_sample_figure(
    batch: dict[str, torch.Tensor],
    sample_idx: int,
    output_path: Path,
    *,
    eos_token_id: int,
    pad_token_id: int,
    dpi: int,
) -> None:
    text_prompt_len = int(batch["text_prompt_lengths"][sample_idx].item())
    speech_prompt_len = int(batch["speech_prompt_lengths"][sample_idx].item())
    text_target_len = int(batch["text_target_lengths"][sample_idx].item())
    speech_target_len = int(batch["speech_target_lengths"][sample_idx].item())

    text_prompt = batch["text_prompt"][sample_idx, :text_prompt_len].cpu().unsqueeze(0)
    text_target = batch["text_target"][sample_idx, :text_target_len].cpu().unsqueeze(0)

    discrete_prompt = (
        batch["discrete_prompt"][sample_idx, :speech_prompt_len, :].transpose(0, 1).cpu()
    )
    discrete_target = (
        batch["discrete_target"][sample_idx, :speech_target_len, :].transpose(0, 1).cpu()
    )

    continuous_prompt = (
        batch["continuous_prompt"][sample_idx, :speech_prompt_len, :].transpose(0, 1).cpu()
    )
    continuous_target = (
        batch["continuous_target"][sample_idx, :speech_target_len, :].transpose(0, 1).cpu()
    )

    attention = batch["attention_mask"][sample_idx].to(dtype=torch.int32).unsqueeze(0).cpu()

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(14, 14),
        gridspec_kw={"height_ratios": [1, 1, 2, 2, 2, 2, 1]},
    )

    _plot_row(
        axes[0],
        text_prompt,
        title="Text Prompt",
        ylabel="text",
        cmap="viridis",
    )
    _plot_row(
        axes[1],
        text_target,
        title=f"Text Target (special: eos={eos_token_id}, pad={pad_token_id})",
        ylabel="text",
        cmap="viridis",
    )
    _plot_row(
        axes[2],
        discrete_prompt,
        title="Discrete Prompt [N, L2]",
        ylabel="stream",
        cmap="magma",
    )
    _plot_row(
        axes[3],
        discrete_target,
        title="Discrete Target [N, L3]",
        ylabel="stream",
        cmap="magma",
    )
    _plot_row(
        axes[4],
        continuous_prompt,
        title="Continuous Prompt [C, L2]",
        ylabel="chan",
        cmap="coolwarm",
    )
    _plot_row(
        axes[5],
        continuous_target,
        title="Continuous Target [C, L3]",
        ylabel="chan",
        cmap="coolwarm",
    )

    attn_img = axes[6].imshow(
        attention.numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#f1efe9", "#0b3954"]),
        vmin=0,
        vmax=1,
    )
    axes[6].set_title("Flattened Attention Mask (1=valid, 0=pad)")
    axes[6].set_ylabel("mask")
    axes[6].set_xlabel("flattened sequence index")
    plt.colorbar(attn_img, ax=axes[6], fraction=0.03, pad=0.01)

    fig.suptitle(
        (
            f"Sample {sample_idx} | "
            f"text_prompt={text_prompt_len}, speech_prompt={speech_prompt_len}, "
            f"text_target={text_target_len}, speech_target={speech_target_len}"
        ),
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
    attention_mask = batch["attention_mask"].to(dtype=torch.int32).cpu()
    text_prompt_lengths = batch["text_prompt_lengths"].cpu()
    speech_prompt_lengths = batch["speech_prompt_lengths"].cpu()
    text_target_lengths = batch["text_target_lengths"].cpu()
    speech_target_lengths = batch["speech_target_lengths"].cpu()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})

    mask_img = axes[0].imshow(
        attention_mask.numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#f1efe9", "#0b3954"]),
        vmin=0,
        vmax=1,
    )
    axes[0].set_title("Batch Flattened Attention Mask [B, L]")
    axes[0].set_ylabel("sample")
    plt.colorbar(mask_img, ax=axes[0], fraction=0.03, pad=0.01)

    sample_idx = torch.arange(attention_mask.shape[0], dtype=torch.long)
    width = 0.2
    axes[1].bar(sample_idx.numpy() - 1.5 * width, text_prompt_lengths.numpy(), width=width, label="text_prompt")
    axes[1].bar(sample_idx.numpy() - 0.5 * width, speech_prompt_lengths.numpy(), width=width, label="speech_prompt")
    axes[1].bar(sample_idx.numpy() + 0.5 * width, text_target_lengths.numpy(), width=width, label="text_target")
    axes[1].bar(sample_idx.numpy() + 1.5 * width, speech_target_lengths.numpy(), width=width, label="speech_target")
    axes[1].set_title("Per-sample Split Lengths")
    axes[1].set_xlabel("sample")
    axes[1].set_ylabel("length")
    axes[1].set_xticks(sample_idx.numpy())
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _expected_flat_length(sample: dict[str, torch.Tensor], num_discrete_streams: int) -> int:
    text_prompt_len = int(sample["text_prompt"].shape[0])
    speech_prompt_len = int(sample["discrete_prompt"].shape[0])
    text_target_len = int(sample["text_target"].shape[0])
    speech_target_len = int(sample["discrete_target"].shape[0])

    speech_block_size = num_discrete_streams + 1
    return (
        text_prompt_len
        + 1
        + speech_prompt_len * speech_block_size
        + 1
        + text_target_len
        + 1
        + speech_target_len * speech_block_size
        + 1
    )


def _validate_collated_batch(batch: dict[str, torch.Tensor]) -> None:
    batch_size = int(batch["text_prompt"].shape[0])
    if batch["discrete_prompt"].shape[0] != batch_size or batch["continuous_prompt"].shape[0] != batch_size:
        raise RuntimeError("Prompt tensors do not agree on batch dimension.")
    if batch["discrete_target"].shape[0] != batch_size or batch["continuous_target"].shape[0] != batch_size:
        raise RuntimeError("Target tensors do not agree on batch dimension.")

    num_discrete_streams = int(batch["discrete_prompt"].shape[2])

    for idx in range(batch_size):
        sample = {
            "text_prompt": batch["text_prompt"][idx, : int(batch["text_prompt_lengths"][idx].item())],
            "discrete_prompt": batch["discrete_prompt"][
                idx, : int(batch["speech_prompt_lengths"][idx].item()), :
            ],
            "text_target": batch["text_target"][idx, : int(batch["text_target_lengths"][idx].item())],
            "discrete_target": batch["discrete_target"][
                idx, : int(batch["speech_target_lengths"][idx].item()), :
            ],
        }
        expected_valid = _expected_flat_length(sample, num_discrete_streams=num_discrete_streams)
        actual_valid = int(batch["attention_mask"][idx].sum().item())
        if actual_valid != expected_valid:
            raise RuntimeError(
                f"sample {idx}: attention_mask valid count ({actual_valid}) != expected ({expected_valid})."
            )


def _print_batch_summary(batch: dict[str, torch.Tensor]) -> None:
    print("Collated keys and shapes:")
    for key in sorted(batch.keys()):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
    print("Per-sample lengths:")
    for idx in range(int(batch["text_prompt"].shape[0])):
        print(
            "  - sample "
            f"{idx}: text_prompt={int(batch['text_prompt_lengths'][idx].item())}, "
            f"speech_prompt={int(batch['speech_prompt_lengths'][idx].item())}, "
            f"text_target={int(batch['text_target_lengths'][idx].item())}, "
            f"speech_target={int(batch['speech_target_lengths'][idx].item())}"
        )


def main() -> None:
    args = build_parser().parse_args()

    _, eos_token_id, pad_token_id, _ = build_shared_token_layout(args.discrete_token_count)

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

    collate = BatchCollate(discrete_token_count=args.discrete_token_count)
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
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            dpi=args.dpi,
        )
        print(f"Saved sample image:   {output_path}")


if __name__ == "__main__":
    main()
