#!/usr/bin/env python3
"""Batch Prism-TTS synthesis from LibriSpeech-style pair metadata.

Expected metadata rows are:

    <target_file_id>|<target_transcript>|<prompt_file_id>|<prompt_transcript>

For LibriSpeech IDs, prompt audio is resolved as:

    <audio-root>/<split>/<speaker_id>/<chapter_id>/<prompt_file_id>.flac

Only prompt audio is read.  The target ID is used as the generated WAV filename
and the target transcript is synthesized.
"""

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
from transformers import AutoFeatureExtractor, MimiModel

# Avoid Matplotlib cache warnings triggered indirectly by optional Lightning imports.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import SharedVocabTokenizer, build_shared_token_layout
from models.mimi_latent_decoder import MimiPreUpsampleLatentDecoder
from utils import generate_utils
from utils.metadata_utils import (
    MetadataEntry,
    infer_split_from_metadata_path,
    load_metadata,
    resolve_librispeech_audio_path,
)


def _resolve_path(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def _select_entries(
    entries: list[MetadataEntry], *, start_index: int, limit: int | None
) -> list[MetadataEntry]:
    if start_index < 0:
        raise ValueError("--start-index must be >= 0.")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be >= 0.")
    if start_index >= len(entries):
        raise ValueError(
            f"--start-index {start_index} is outside the metadata range [0, {len(entries) - 1}]."
        )
    end_index = None if limit is None else start_index + limit
    selected = entries[start_index:end_index]
    if not selected:
        raise ValueError("No rows selected; set --limit to a positive number.")
    return selected


class PrismTTSBatchGenerator:
    """Load Prism-TTS and Mimi once, then synthesize metadata entries sequentially."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = generate_utils.resolve_device(args.device)
        self.model_dtype = generate_utils.resolve_torch_dtype(args.dtype)

        model_config = generate_utils.read_yaml(args.model_config)
        data_config = generate_utils.read_yaml(args.data_config)
        self.model = generate_utils.build_model(model_config)
        generate_utils.load_checkpoint(self.model, args.checkpoint, use_ema=bool(args.use_ema))
        self.model.to(device=self.device, dtype=self.model_dtype)
        self.model.eval()

        data_cfg = generate_utils.require_mapping(data_config, "data")
        shared_layout_cfg = generate_utils.require_mapping(data_cfg, "shared_layout")
        dataset_cfg = generate_utils.require_mapping(data_cfg, "dataset")
        discrete_token_count = int(shared_layout_cfg["discrete_token_count"])
        _, self.eos_token_id, self.pad_token_id, text_token_offset = build_shared_token_layout(
            discrete_token_count
        )

        vocab_path = Path(data_cfg.get("vocab_path", "dataset/vocab.txt"))
        if not vocab_path.is_absolute():
            vocab_path = (Path.cwd() / vocab_path).resolve()
        self.tokenizer = SharedVocabTokenizer(
            vocab_path=vocab_path,
            text_token_offset=text_token_offset,
            eos_token_id=self.eos_token_id,
            append_eos=bool(dataset_cfg.get("append_eos_to_text", False)),
        )

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            args.mimi_model,
            revision=args.mimi_revision,
            token=args.hf_token,
            local_files_only=bool(args.local_files_only),
        )
        self.mimi_sample_rate = int(getattr(self.feature_extractor, "sampling_rate", 24_000))
        self.mimi_model = MimiModel.from_pretrained(
            args.mimi_model,
            revision=args.mimi_revision,
            token=args.hf_token,
            local_files_only=bool(args.local_files_only),
        )
        self.mimi_model.to(device=self.device)
        self.mimi_model.eval()
        self.decoder = MimiPreUpsampleLatentDecoder(
            pretrained_model_name_or_path=args.mimi_model,
            device=str(self.device),
            dtype=args.dtype,
            local_files_only=bool(args.local_files_only),
            revision=args.mimi_revision,
            token=args.hf_token,
        )

    def _encode_prompt(self, prompt_audio_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_audio, prompt_sample_rate = generate_utils.read_audio(prompt_audio_path)
        prompt_audio = generate_utils.resample_if_needed(
            prompt_audio,
            prompt_sample_rate,
            self.mimi_sample_rate,
        )
        features = self.feature_extractor(
            raw_audio=prompt_audio,
            sampling_rate=self.mimi_sample_rate,
            return_tensors="pt",
        )
        input_values = features["input_values"]
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)
        if input_values.dim() != 3:
            raise ValueError(f"Unexpected Mimi input shape: {tuple(input_values.shape)}")
        input_values = input_values.to(device=self.device)

        padding_mask = features.get("padding_mask")
        if padding_mask is not None:
            if padding_mask.dim() == 2:
                padding_mask = padding_mask.unsqueeze(1)
            if padding_mask.dim() != 3:
                raise ValueError(f"Unexpected Mimi padding_mask shape: {tuple(padding_mask.shape)}")
            padding_mask = padding_mask.to(device=self.device)

        # Training uses the full Mimi codec reconstruction as the continuous
        # prefix even when the model predicts only a subset of codebooks.
        with torch.inference_mode():
            encoded = self.mimi_model.encode(
                input_values=input_values,
                padding_mask=padding_mask,
                num_quantizers=int(self.mimi_model.config.num_quantizers),
                return_dict=True,
            )
            prompt_codes = encoded.audio_codes
            if prompt_codes is None:
                raise RuntimeError("Mimi encode did not return audio_codes.")
            prompt_latents = self.mimi_model.quantizer.decode(prompt_codes)

        raw_discrete = prompt_codes[0, : self.model.num_discrete_tokens].transpose(0, 1).to(
            dtype=torch.long
        ).cpu()
        raw_continuous = prompt_latents[0].transpose(0, 1).to(dtype=torch.float32).cpu()
        return raw_discrete, raw_continuous

    def _run_generation(
        self,
        *,
        text_prompt: torch.Tensor,
        discrete_prompt: torch.Tensor,
        continuous_prompt: torch.Tensor,
        text_target: torch.Tensor,
        prompt_frame_count: int,
        max_new_blocks: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Any:
        with torch.inference_mode():
            return self.model.generate(
                text_prompt=text_prompt,
                discrete_prompt=discrete_prompt,
                continuous_prompt=continuous_prompt,
                text_target=text_target,
                text_prompt_lengths=torch.tensor(
                    [text_prompt.shape[1]], device=self.device, dtype=torch.long
                ),
                speech_prompt_lengths=torch.tensor(
                    [prompt_frame_count], device=self.device, dtype=torch.long
                ),
                text_target_lengths=torch.tensor(
                    [text_target.shape[1]], device=self.device, dtype=torch.long
                ),
                speech_target_lengths=torch.tensor(
                    [max_new_blocks], device=self.device, dtype=torch.long
                ),
                max_new_blocks=max_new_blocks,
                discrete_eos_token_id=self.eos_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                flow_num_steps=self.args.flow_num_steps,
                generation_method=str(self.args.generation_method),
                force_silent_special_tokens=bool(self.args.force_silent_special_tokens),
                return_dict=True,
            )

    def generate(
        self,
        *,
        entry: MetadataEntry,
        prompt_audio_path: Path,
        output_audio_path: Path,
    ) -> dict[str, int | float | str]:
        """Synthesize one entry and return JSON-serializable generation metrics."""
        if self.args.seed is not None:
            # Per-row seeds make output reproducible even when a previous run
            # is resumed with --skip-existing.
            row_seed = int(self.args.seed) + entry.index
            torch.manual_seed(row_seed)
            np.random.seed(row_seed)

        raw_prompt_discrete, raw_prompt_continuous = self._encode_prompt(prompt_audio_path)
        prompt_text_tokens = generate_utils.safe_tokenize(
            self.tokenizer,
            entry.prompt_transcript,
            "prompt_transcript",
        )
        target_text_tokens = generate_utils.safe_tokenize(
            self.tokenizer,
            entry.target_transcript,
            "target_transcript",
        )
        if target_text_tokens.numel() == 0:
            raise ValueError("Target transcript is empty after tokenization.")

        max_new_blocks = self.args.max_new_blocks
        if max_new_blocks is None:
            max_new_blocks = generate_utils.estimate_max_new_blocks(
                target_text_token_count=int(target_text_tokens.numel()),
                prompt_text_token_count=int(prompt_text_tokens.numel()),
                prompt_frame_count=int(raw_prompt_discrete.shape[0]),
                duration_scale=float(self.args.duration_scale),
                trailing_pad_blocks=int(self.args.trailing_pad_blocks),
            )
        if max_new_blocks < 1:
            raise ValueError("--max-new-blocks must be >= 1.")

        text_prompt = prompt_text_tokens.unsqueeze(0).to(device=self.device, dtype=torch.long)
        discrete_prompt = raw_prompt_discrete.transpose(0, 1).unsqueeze(0).to(
            device=self.device,
            dtype=torch.long,
        )
        continuous_prompt = raw_prompt_continuous.unsqueeze(0).to(
            device=self.device,
            dtype=self.model_dtype,
        )
        text_target = target_text_tokens.unsqueeze(0).to(device=self.device, dtype=torch.long)
        special_token_ids = (self.eos_token_id, self.pad_token_id)

        generation = self._run_generation(
            text_prompt=text_prompt,
            discrete_prompt=discrete_prompt,
            continuous_prompt=continuous_prompt,
            text_target=text_target,
            prompt_frame_count=int(raw_prompt_discrete.shape[0]),
            max_new_blocks=int(max_new_blocks),
            do_sample=bool(self.args.do_sample),
            temperature=float(self.args.temperature),
            top_k=int(self.args.top_k),
            top_p=float(self.args.top_p),
        )

        attempts: list[tuple[str, Any, dict[str, float], float]] = []

        def summarize_attempt(label: str, candidate: Any) -> dict[str, float] | None:
            if candidate.discrete_ids is None:
                return None
            stats = generate_utils.summarize_discrete_generation(
                discrete_ids=candidate.discrete_ids[0],
                num_discrete_tokens=int(self.model.num_discrete_tokens),
                special_token_ids=special_token_ids,
            )
            attempts.append((label, candidate, stats, generate_utils.discrete_quality_score(stats)))
            return stats

        discrete_stats = summarize_attempt("initial", generation)
        selected_attempt = "initial"
        if discrete_stats is not None and generate_utils.is_collapsed_discrete_stats(discrete_stats):
            retry_profiles = (
                ("retry-1", 0.9, 100, 0.95),
                ("retry-2", 1.1, 200, 0.97),
                ("retry-3", 1.25, 400, 0.99),
            )
            for label, temperature, top_k, top_p in retry_profiles:
                retry = self._run_generation(
                    text_prompt=text_prompt,
                    discrete_prompt=discrete_prompt,
                    continuous_prompt=continuous_prompt,
                    text_target=text_target,
                    prompt_frame_count=int(raw_prompt_discrete.shape[0]),
                    max_new_blocks=int(max_new_blocks),
                    do_sample=True,
                    temperature=max(temperature, float(self.args.temperature)),
                    top_k=max(top_k, int(self.args.top_k)),
                    top_p=max(top_p, float(self.args.top_p)),
                )
                summarize_attempt(label, retry)
            if attempts:
                selected_attempt, generation, discrete_stats, _ = max(
                    attempts,
                    key=lambda item: item[3],
                )

        predicted_latents = generation.continuous_latents
        if predicted_latents is None or predicted_latents.shape[0] == 0:
            raise RuntimeError("Generation produced no continuous latents.")
        sample_latents = predicted_latents[0]
        if generation.discrete_ids is not None:
            sample_latents = generate_utils.trim_latent_special_blocks(
                latents=sample_latents,
                discrete_ids=generation.discrete_ids[0],
                num_discrete_tokens=int(self.model.num_discrete_tokens),
                special_token_ids=special_token_ids,
                trim_head=bool(self.args.trim_leading_special_blocks),
                trim_tail=bool(self.args.trim_tail_special_blocks),
            )

        with torch.inference_mode():
            decoded = self.decoder(sample_latents.detach())
        waveform = decoded.detach().cpu().float().numpy() if torch.is_tensor(decoded) else np.asarray(decoded)
        waveform = np.squeeze(waveform)
        if waveform.ndim > 1:
            waveform = waveform.reshape(-1)
        if waveform.size == 0:
            raise RuntimeError("Decoded waveform is empty.")
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(waveform)))
        if peak > 0:
            waveform = waveform / peak

        generate_utils.write_wav(output_audio_path, waveform, sample_rate=int(self.decoder.sample_rate))
        mel_path = None
        if self.args.save_mel:
            mel_path = generate_utils.save_mel_spectrogram_plot(
                waveform=waveform,
                sample_rate=int(self.decoder.sample_rate),
                output_audio_path=output_audio_path,
            )

        metrics: dict[str, int | float | str] = {
            "selected_attempt": selected_attempt,
            "prompt_blocks": int(raw_prompt_discrete.shape[0]),
            "generated_blocks": int(sample_latents.shape[0]),
            "sample_rate": int(self.decoder.sample_rate),
            "waveform_samples": int(waveform.size),
        }
        if mel_path is not None:
            metrics["mel_path"] = str(mel_path)
        if discrete_stats is not None:
            metrics.update(
                {
                    "discrete_length": int(discrete_stats["length"]),
                    "discrete_unique_tokens": int(discrete_stats["unique_tokens"]),
                    "discrete_special_ratio": float(discrete_stats["special_ratio"]),
                    "discrete_longest_run_ratio": float(discrete_stats["longest_run_ratio"]),
                }
            )
        return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate Prism-TTS WAV files from four-column LibriSpeech metadata."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Rows: target_id|target_transcript|prompt_id|prompt_transcript.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="LibriSpeech root containing split directories (default: metadata parent).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Audio split below --audio-root (default: infer from metadata-<split>.txt).",
    )
    parser.add_argument(
        "--audio-extension",
        type=str,
        default=".flac",
        help="Prompt audio extension (default: .flac).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("synth/librispeech"),
        help="Directory for <target_file-id>.wav outputs.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSONL report containing a record for every selected row.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="0-based metadata row to start from.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of metadata rows.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not regenerate WAV files that already exist and are non-empty.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed row and continue; exits non-zero if any row fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate metadata and print resolved prompt/output paths without loading models.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="PrismTTS checkpoint (.ckpt/.pt).")
    parser.add_argument("--model-config", type=Path, default=Path("config/model.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("config/data.yaml"))
    parser.add_argument("--device", type=str, default="auto", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=("float32", "float16", "bfloat16"),
        help="PrismTTS model dtype.",
    )
    parser.add_argument("--mimi-model", type=str, default="kyutai/mimi")
    parser.add_argument("--mimi-revision", type=str, default="main")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token for private models.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed; each row uses base seed + metadata index.",
    )
    parser.add_argument("--max-new-blocks", type=int, default=None)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--trailing-pad-blocks", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--flow-num-steps", type=int, default=64)
    parser.add_argument("--generation-method", choices=("ar", "causal"), default="ar")
    parser.add_argument(
        "--force-silent-special-tokens",
        dest="force_silent_special_tokens",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-force-silent-special-tokens",
        dest="force_silent_special_tokens",
        action="store_false",
    )
    parser.add_argument(
        "--trim-leading-special-blocks",
        dest="trim_leading_special_blocks",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-trim-leading-special-blocks",
        dest="trim_leading_special_blocks",
        action="store_false",
    )
    parser.add_argument(
        "--trim-tail-special-blocks",
        dest="trim_tail_special_blocks",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-trim-tail-special-blocks",
        dest="trim_tail_special_blocks",
        action="store_false",
    )
    parser.add_argument(
        "--save-mel",
        action="store_true",
        help="Also save a mel-spectrogram PNG for each generated WAV.",
    )
    return parser.parse_args()


def _build_report_record(
    *,
    entry: MetadataEntry,
    prompt_audio_path: Path,
    output_audio_path: Path,
    status: str,
    metrics: dict[str, int | float | str] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "status": status,
        "metadata_index": entry.index,
        "metadata_line": entry.line_number,
        "target_file_id": entry.target_file_id,
        "target_transcript": entry.target_transcript,
        "prompt_file_id": entry.prompt_file_id,
        "prompt_transcript": entry.prompt_transcript,
        "prompt_audio": str(prompt_audio_path),
        "output_audio": str(output_audio_path),
    }
    if metrics is not None:
        record.update(metrics)
    if error is not None:
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
    return record


def main() -> None:
    args = parse_args()
    metadata_path = _resolve_path(args.metadata)
    entries = _select_entries(
        load_metadata(metadata_path),
        start_index=int(args.start_index),
        limit=args.limit,
    )
    audio_root = _resolve_path(args.audio_root) if args.audio_root is not None else metadata_path.parent
    split = args.split if args.split is not None else infer_split_from_metadata_path(metadata_path)
    if args.split is None and split is None:
        raise ValueError(
            "Could not infer the audio split from the metadata filename; pass --split explicitly."
        )
    output_dir = _resolve_path(args.output_dir)

    print(
        "[generate_from_metadata] "
        f"selected={len(entries)} metadata={metadata_path} audio_root={audio_root} split={split!r}"
    )
    if args.dry_run:
        for entry in entries:
            prompt_path = resolve_librispeech_audio_path(
                audio_root=audio_root,
                split=split,
                file_id=entry.prompt_file_id,
                extension=args.audio_extension,
                line_number=entry.line_number,
            )
            print(
                f"[dry-run] index={entry.index} prompt={prompt_path} "
                f"output={output_dir / f'{entry.target_file_id}.wav'}"
            )
        return

    report_handle = None
    if args.report is not None:
        report_path = _resolve_path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_handle = report_path.open("w", encoding="utf-8")

    generator = PrismTTSBatchGenerator(args)
    succeeded = 0
    skipped = 0
    failures = 0
    try:
        for position, entry in enumerate(entries, start=1):
            prompt_path = resolve_librispeech_audio_path(
                audio_root=audio_root,
                split=split,
                file_id=entry.prompt_file_id,
                extension=args.audio_extension,
                line_number=entry.line_number,
            )
            output_path = output_dir / f"{entry.target_file_id}.wav"
            if args.skip_existing and output_path.is_file() and output_path.stat().st_size > 0:
                skipped += 1
                record = _build_report_record(
                    entry=entry,
                    prompt_audio_path=prompt_path,
                    output_audio_path=output_path,
                    status="skipped",
                )
                print(
                    f"[generate_from_metadata] [{position}/{len(entries)}] "
                    f"skipped {entry.target_file_id}"
                )
            else:
                try:
                    metrics = generator.generate(
                        entry=entry,
                        prompt_audio_path=prompt_path,
                        output_audio_path=output_path,
                    )
                except Exception as exc:
                    failures += 1
                    record = _build_report_record(
                        entry=entry,
                        prompt_audio_path=prompt_path,
                        output_audio_path=output_path,
                        status="failed",
                        error=exc,
                    )
                    print(
                        f"[generate_from_metadata] [{position}/{len(entries)}] failed "
                        f"{entry.target_file_id}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    if not args.continue_on_error:
                        if report_handle is not None:
                            report_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            report_handle.flush()
                        raise
                else:
                    succeeded += 1
                    record = _build_report_record(
                        entry=entry,
                        prompt_audio_path=prompt_path,
                        output_audio_path=output_path,
                        status="generated",
                        metrics=metrics,
                    )
                    print(
                        f"[generate_from_metadata] [{position}/{len(entries)}] generated "
                        f"{entry.target_file_id} ({metrics['generated_blocks']} blocks)"
                    )
            if report_handle is not None:
                report_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                report_handle.flush()
    finally:
        if report_handle is not None:
            report_handle.close()

    print(
        "[generate_from_metadata] complete: "
        f"generated={succeeded} skipped={skipped} failed={failures}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
