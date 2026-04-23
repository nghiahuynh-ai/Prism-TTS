from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch
from transformers import LlamaConfig


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

from models.prism_tts import PrismTTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render PrismTTS stream-wise and block-wise attention masks and save them as images."
        )
    )
    parser.add_argument(
        "--num-discrete-tokens",
        type=int,
        required=True,
        help="Number of discrete streams per block.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        required=True,
        help="Number of block positions to visualize.",
    )
    parser.add_argument(
        "--attention-mask",
        type=str,
        default=None,
        help=(
            "Optional comma/space-separated 0/1 block-level or token-level mask. "
            "Example: '1 1 1 0' or '1,1,1,0,0,0,0,0'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "attention_masks",
        help="Directory to write image files into.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="prism_tts_attention_masks",
        help="Filename prefix for the generated images.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output image DPI.",
    )
    return parser


def make_model(num_discrete_tokens: int) -> PrismTTS:
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return PrismTTS(
        backbone_config=config,
        num_discrete_tokens=num_discrete_tokens,
        discrete_vocab_size=32,
        continuous_latent_size=8,
        flow_num_res_blocks=1,
        flow_sample_steps=2,
    ).eval()


def parse_attention_mask(
    attention_mask_text: Optional[str],
    num_blocks: int,
    block_size: int,
) -> Optional[torch.Tensor]:
    if attention_mask_text is None:
        return None

    parts = [part for part in re.split(r"[\s,]+", attention_mask_text.strip()) if part]
    if not parts:
        raise ValueError("attention mask text is empty.")

    values = []
    for part in parts:
        if part not in {"0", "1"}:
            raise ValueError(
                f"attention mask entries must be 0 or 1, got {part!r}."
            )
        values.append(int(part))

    expected_lengths = {num_blocks, num_blocks * block_size}
    if len(values) not in expected_lengths:
        raise ValueError(
            "attention mask must contain either "
            f"{num_blocks} block values or {num_blocks * block_size} token values, "
            f"got {len(values)}."
        )

    return torch.tensor([values], dtype=torch.long)


def token_labels(num_blocks: int, num_discrete_tokens: int) -> list[str]:
    labels = []
    for block_idx in range(num_blocks):
        labels.append(f"B{block_idx}:T")
        for token_idx in range(num_discrete_tokens):
            labels.append(f"B{block_idx}:D{token_idx}")
        labels.append(f"B{block_idx}:C")
    return labels


def apply_key_padding_mask(
    structural_mask: torch.Tensor,
    flat_attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if flat_attention_mask is None:
        return structural_mask
    key_mask = flat_attention_mask.to(dtype=torch.bool)[0]
    return structural_mask & key_mask.unsqueeze(0)


def render_mask_panels(
    masks: list[tuple[str, torch.Tensor]],
    num_discrete_tokens: int,
    block_size: int,
    num_blocks: int,
    output_path: Path,
    dpi: int,
) -> None:
    total_tokens = num_blocks * block_size
    labels = token_labels(num_blocks, num_discrete_tokens)
    cmap = matplotlib.colors.ListedColormap(["#f6f4ee", "#0b3954"])

    fig, axes = plt.subplots(1, len(masks), figsize=(7 * len(masks), 7), squeeze=False)
    for ax, (title, mask) in zip(axes[0], masks):
        ax.imshow(mask.to(dtype=torch.int).cpu().numpy(), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("Key positions")
        ax.set_ylabel("Query positions")

        if total_tokens <= 24:
            ticks = list(range(total_tokens))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(labels, rotation=90, fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
        else:
            block_centers = [
                block_idx * block_size + (block_size - 1) / 2
                for block_idx in range(num_blocks)
            ]
            block_labels = [f"B{block_idx}" for block_idx in range(num_blocks)]
            ax.set_xticks(block_centers)
            ax.set_yticks(block_centers)
            ax.set_xticklabels(block_labels, rotation=0)
            ax.set_yticklabels(block_labels)

        boundary_positions = [
            block_idx * block_size - 0.5 for block_idx in range(1, num_blocks)
        ]
        for boundary in boundary_positions:
            ax.axhline(boundary, color="#d95d39", linewidth=1.2)
            ax.axvline(boundary, color="#d95d39", linewidth=1.2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()

    model = make_model(args.num_discrete_tokens)
    total_tokens = args.num_blocks * model.block_size

    stream_mask, block_mask = model._build_dual_attention_masks(
        total_tokens=total_tokens,
        batch_size=1,
        device=torch.device("cpu"),
    )

    parsed_attention_mask = parse_attention_mask(
        args.attention_mask,
        num_blocks=args.num_blocks,
        block_size=model.block_size,
    )
    flat_attention_mask = model._flatten_attention_mask(
        parsed_attention_mask,
        num_blocks=args.num_blocks,
    )

    raw_masks = [
        ("Stream-wise", stream_mask[0, 0]),
        ("Block-wise", block_mask[0, 0]),
    ]
    raw_output_path = args.output_dir / f"{args.output_prefix}_raw.png"
    render_mask_panels(
        raw_masks,
        num_discrete_tokens=args.num_discrete_tokens,
        block_size=model.block_size,
        num_blocks=args.num_blocks,
        output_path=raw_output_path,
        dpi=args.dpi,
    )

    print(f"Saved raw masks to {raw_output_path}")

    if flat_attention_mask is not None:
        padded_masks = [
            ("Stream-wise + key padding", apply_key_padding_mask(stream_mask[0, 0], flat_attention_mask)),
            ("Block-wise + key padding", apply_key_padding_mask(block_mask[0, 0], flat_attention_mask)),
        ]
        padded_output_path = args.output_dir / f"{args.output_prefix}_with_padding.png"
        render_mask_panels(
            padded_masks,
            num_discrete_tokens=args.num_discrete_tokens,
            block_size=model.block_size,
            num_blocks=args.num_blocks,
            output_path=padded_output_path,
            dpi=args.dpi,
        )
        print(f"Saved padding-aware masks to {padded_output_path}")


if __name__ == "__main__":
    main()
