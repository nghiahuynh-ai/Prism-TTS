from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoFeatureExtractor, MimiModel

# Avoid Matplotlib cache warnings triggered indirectly by optional Lightning imports.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from dataset.dataset import SharedVocabTokenizer, build_shared_token_layout
from models.mimi_latent_decoder import MimiPreUpsampleLatentDecoder
from utils import generate_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate speech with PrismTTS from raw text + prompt audio (Mimi codec)."
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
        "--data-config",
        type=Path,
        default=Path("config/data.yaml"),
        help="Path to data YAML config.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Target text to synthesize.",
    )
    parser.add_argument(
        "--prompt-audio",
        type=Path,
        required=True,
        help="Prompt/reference audio path (.wav).",
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="",
        help="Transcript of prompt audio. Optional; empty string is allowed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synth/output.wav"),
        help="Output WAV path.",
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
        "--mimi-model",
        type=str,
        default="kyutai/mimi",
        help="Hugging Face Mimi model id/path used for prompt encode/decode.",
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
        "--seed",
        type=int,
        default=None,
        help="Optional random seed.",
    )
    parser.add_argument(
        "--max-new-blocks",
        type=int,
        default=None,
        help="Override max generation blocks. If omitted, estimated from text/prompt ratio.",
    )
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to estimated duration when --max-new-blocks is not set.",
    )
    parser.add_argument(
        "--trailing-pad-blocks",
        type=int,
        default=0,
        help="Extra safety blocks added to --max-new-blocks estimate.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable stochastic sampling for discrete IDs.",
    )
    parser.add_argument(
        "--flow-num-steps",
        type=int,
        default=64,
        help="Override flow sampling steps for continuous latents.",
    )
    parser.add_argument(
        "--parallel-num-steps",
        type=int,
        default=64,
        help="Number of iterative refinement steps for parallel generation.",
    )
    parser.add_argument(
        "--generation-method",
        type=str,
        default="causal",
        choices=("causal", "parallel", "parallel_stable"),
        help="Speech block generation strategy.",
    )
    parser.add_argument(
        "--force-silent-special-tokens",
        dest="force_silent_special_tokens",
        action="store_true",
        default=True,
        help="Force predicted special discrete blocks (EOS/PAD) to zero continuous latents.",
    )
    parser.add_argument(
        "--no-force-silent-special-tokens",
        dest="force_silent_special_tokens",
        action="store_false",
        help="Disable zeroing of continuous latents for special discrete blocks.",
    )
    parser.add_argument(
        "--trim-leading-special-blocks",
        dest="trim_leading_special_blocks",
        action="store_true",
        default=True,
        help="Trim generated leading blocks where all discrete streams are EOS/PAD.",
    )
    parser.add_argument(
        "--no-trim-leading-special-blocks",
        dest="trim_leading_special_blocks",
        action="store_false",
        help="Disable special-leading trimming before audio decoding.",
    )
    parser.add_argument(
        "--trim-tail-special-blocks",
        dest="trim_tail_special_blocks",
        action="store_true",
        default=True,
        help="Trim generated tail blocks where all discrete streams are EOS/PAD.",
    )
    parser.add_argument(
        "--no-trim-tail-special-blocks",
        dest="trim_tail_special_blocks",
        action="store_false",
        help="Disable special-tail trimming before audio decoding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))

    device = generate_utils.resolve_device(args.device)
    model_dtype = generate_utils.resolve_torch_dtype(args.dtype)

    model_config = generate_utils.read_yaml(args.model_config)
    data_config = generate_utils.read_yaml(args.data_config)
    model = generate_utils.build_model(model_config)
    generate_utils.load_checkpoint(model, args.checkpoint, use_ema=bool(args.use_ema))
    model.to(device=device, dtype=model_dtype)
    model.eval()

    data_cfg = generate_utils.require_mapping(data_config, "data")
    shared_layout_cfg = generate_utils.require_mapping(data_cfg, "shared_layout")
    dataset_cfg = generate_utils.require_mapping(data_cfg, "dataset")

    discrete_token_count = int(shared_layout_cfg["discrete_token_count"])
    _, eos_token_id, pad_token_id, text_token_offset = build_shared_token_layout(
        discrete_token_count
    )

    vocab_path_raw = data_cfg.get("vocab_path", "dataset/vocab.txt")
    vocab_path = Path(vocab_path_raw)
    if not vocab_path.is_absolute():
        vocab_path = (Path.cwd() / vocab_path).resolve()
    tokenizer = SharedVocabTokenizer(
        vocab_path=vocab_path,
        text_token_offset=text_token_offset,
        eos_token_id=eos_token_id,
        append_eos=bool(dataset_cfg.get("append_eos_to_text", False)),
    )

    prompt_audio, prompt_sample_rate = generate_utils.read_wav(args.prompt_audio)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.mimi_model,
        revision=args.mimi_revision,
        token=args.hf_token,
        local_files_only=bool(args.local_files_only),
    )
    mimi_sample_rate = int(getattr(feature_extractor, "sampling_rate", 24_000))
    prompt_audio = generate_utils.resample_if_needed(prompt_audio, prompt_sample_rate, mimi_sample_rate)

    mimi_model = MimiModel.from_pretrained(
        args.mimi_model,
        revision=args.mimi_revision,
        token=args.hf_token,
        local_files_only=bool(args.local_files_only),
    )
    mimi_model.to(device=device)
    mimi_model.eval()

    features = feature_extractor(
        raw_audio=prompt_audio,
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
            num_quantizers=int(model.num_discrete_tokens),
            return_dict=True,
        )
        prompt_codes = encoded.audio_codes
        if prompt_codes is None:
            raise RuntimeError("Mimi encode did not return audio_codes.")
        prompt_latents = mimi_model.quantizer.decode(prompt_codes)
        # prompt_codes: [B, N, T], prompt_latents: [B, C, T]

    raw_prompt_discrete = prompt_codes[0].transpose(0, 1).to(dtype=torch.long).cpu()
    raw_prompt_continuous = prompt_latents[0].transpose(0, 1).to(dtype=torch.float32).cpu()

    prompt_text_tokens = generate_utils.safe_tokenize(tokenizer, args.prompt_text, "prompt_text")
    target_text_tokens = generate_utils.safe_tokenize(tokenizer, args.text, "text")
    if target_text_tokens.numel() == 0:
        raise ValueError("Target text is empty after tokenization.")

    max_new_blocks = args.max_new_blocks
    if max_new_blocks is None:
        max_new_blocks = generate_utils.estimate_max_new_blocks(
            target_text_token_count=int(target_text_tokens.numel()),
            prompt_text_token_count=int(prompt_text_tokens.numel()),
            prompt_frame_count=int(raw_prompt_discrete.shape[0]),
            duration_scale=float(args.duration_scale),
            trailing_pad_blocks=int(args.trailing_pad_blocks),
        )
    if max_new_blocks < 1:
        raise ValueError("--max-new-blocks must be >= 1.")

    text_prompt = prompt_text_tokens.unsqueeze(0).to(device=device, dtype=torch.long)
    # Pass [B, N, L] for compatibility with training/inference call-sites.
    discrete_prompt = raw_prompt_discrete.transpose(0, 1).unsqueeze(0).to(
        device=device,
        dtype=torch.long,
    )
    continuous_prompt = raw_prompt_continuous.unsqueeze(0).to(
        device=device,
        dtype=model_dtype,
    )
    text_target = target_text_tokens.unsqueeze(0).to(device=device, dtype=torch.long)
    special_token_ids = (eos_token_id, pad_token_id)

    def _run_generation(
        *,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Any:
        with torch.no_grad():
            return model.generate(
                text_prompt=text_prompt,
                discrete_prompt=discrete_prompt,
                continuous_prompt=continuous_prompt,
                text_target=text_target,
                text_prompt_lengths=torch.tensor([text_prompt.shape[1]], device=device, dtype=torch.long),
                speech_prompt_lengths=torch.tensor(
                    [raw_prompt_discrete.shape[0]],
                    device=device,
                    dtype=torch.long,
                ),
                text_target_lengths=torch.tensor([text_target.shape[1]], device=device, dtype=torch.long),
                speech_target_lengths=torch.tensor([int(max_new_blocks)], device=device, dtype=torch.long),
                max_new_blocks=int(max_new_blocks),
                discrete_eos_token_id=eos_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                flow_num_steps=args.flow_num_steps,
                parallel_num_steps=args.parallel_num_steps,
                generation_method=str(args.generation_method),
                force_silent_special_tokens=bool(args.force_silent_special_tokens),
                return_dict=True,
            )

    generation = _run_generation(
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
    )

    discrete_stats: dict[str, float] | None = None
    attempt_records: list[tuple[str, Any, dict[str, float], float]] = []

    def _summarize_attempt(label: str, candidate_generation: Any) -> dict[str, float] | None:
        if candidate_generation.discrete_ids is None:
            return None
        candidate_stats = generate_utils.summarize_discrete_generation(
            discrete_ids=candidate_generation.discrete_ids[0],
            num_discrete_tokens=int(model.num_discrete_tokens),
            special_token_ids=special_token_ids,
        )
        attempt_records.append(
            (
                label,
                candidate_generation,
                candidate_stats,
                generate_utils.discrete_quality_score(candidate_stats),
            )
        )
        return candidate_stats

    discrete_stats = _summarize_attempt("initial", generation)

    if discrete_stats is not None and generate_utils.is_collapsed_discrete_stats(discrete_stats):
        print("[generate.py] initial decode appears collapsed; running sampling retries.")
        retry_profiles = (
            (
                "retry-1",
                {
                    "do_sample": True,
                    "temperature": max(0.9, float(args.temperature)),
                    "top_k": max(100, int(args.top_k)),
                    "top_p": max(0.95, float(args.top_p)),
                },
            ),
            (
                "retry-2",
                {
                    "do_sample": True,
                    "temperature": max(1.1, float(args.temperature)),
                    "top_k": max(200, int(args.top_k)),
                    "top_p": max(0.97, float(args.top_p)),
                },
            ),
            (
                "retry-3",
                {
                    "do_sample": True,
                    "temperature": max(1.25, float(args.temperature)),
                    "top_k": max(400, int(args.top_k)),
                    "top_p": max(0.99, float(args.top_p)),
                },
            ),
        )
        for label, profile in retry_profiles:
            retry_generation = _run_generation(
                do_sample=bool(profile["do_sample"]),
                temperature=float(profile["temperature"]),
                top_k=int(profile["top_k"]),
                top_p=float(profile["top_p"]),
            )
            retry_stats = _summarize_attempt(label, retry_generation)
            if retry_stats is None:
                continue
            print(
                f"[generate.py] {label}: len={int(retry_stats['length'])}, "
                f"unique={int(retry_stats['unique_tokens'])}, "
                f"special_ratio={retry_stats['special_ratio']:.3f}, "
                f"longest_run_ratio={retry_stats['longest_run_ratio']:.3f}"
            )

        if attempt_records:
            best_label, best_generation, best_stats, _ = max(
                attempt_records,
                key=lambda item: item[3],
            )
            generation = best_generation
            discrete_stats = best_stats
            print(f"[generate.py] selected attempt: {best_label}")

    if discrete_stats is not None:
        print(
            "[generate.py] discrete summary: "
            f"len={int(discrete_stats['length'])}, unique={int(discrete_stats['unique_tokens'])}, "
            f"special_ratio={discrete_stats['special_ratio']:.3f}, "
            f"longest_run_ratio={discrete_stats['longest_run_ratio']:.3f}"
        )
    if generation.discrete_ids is not None and generation.discrete_ids.shape[0] > 0:
        sample_discrete_codes = generation.discrete_ids[0].detach().to(dtype=torch.long).cpu()
        print(
            "[generate.py] generated discrete codes "
            f"(shape={tuple(sample_discrete_codes.shape)}, format=[num_streams, num_blocks]):"
        )
        print(sample_discrete_codes.tolist())

    predicted_latents = generation.continuous_latents
    if predicted_latents is None or predicted_latents.shape[0] == 0:
        raise RuntimeError("Generation produced no continuous latents.")
    sample_latents = predicted_latents[0]

    latent_std = float(sample_latents.std(unbiased=False).item())
    if sample_latents.shape[0] > 1:
        latent_delta_std = float((sample_latents[1:] - sample_latents[:-1]).std(unbiased=False).item())
    else:
        latent_delta_std = 0.0
    print(
        "[generate.py] latent summary: "
        f"std={latent_std:.6f}, delta_std={latent_delta_std:.6f}"
    )

    if generation.discrete_ids is not None:
        sample_discrete = generation.discrete_ids[0]
        sample_latents = generate_utils.trim_latent_special_blocks(
            latents=sample_latents,
            discrete_ids=sample_discrete,
            num_discrete_tokens=int(model.num_discrete_tokens),
            special_token_ids=special_token_ids,
            trim_head=bool(args.trim_leading_special_blocks),
            trim_tail=bool(args.trim_tail_special_blocks),
        )

    decoder = MimiPreUpsampleLatentDecoder(
        pretrained_model_name_or_path=args.mimi_model,
        device=str(device),
        dtype=args.dtype,
        local_files_only=bool(args.local_files_only),
        revision=args.mimi_revision,
        token=args.hf_token,
    )
    with torch.no_grad():
        decoded = decoder(sample_latents.detach())

    if isinstance(decoded, torch.Tensor):
        waveform = decoded.detach().cpu().float().numpy()
    else:
        waveform = np.asarray(decoded, dtype=np.float32)
    waveform = np.squeeze(waveform)
    if waveform.ndim > 1:
        waveform = waveform.reshape(-1)
    if waveform.size == 0:
        raise RuntimeError("Decoded waveform is empty.")
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform = waveform / peak

    output_audio_path = args.output.expanduser()
    if not output_audio_path.is_absolute():
        output_audio_path = (Path.cwd() / output_audio_path).resolve()
    else:
        output_audio_path = output_audio_path.resolve()

    output_sample_rate = int(decoder.sample_rate)
    generate_utils.write_wav(output_audio_path, waveform, sample_rate=output_sample_rate)
    mel_path = generate_utils.save_mel_spectrogram_plot(
        waveform=waveform,
        sample_rate=output_sample_rate,
        output_audio_path=output_audio_path,
    )

    print(f"[generate.py] wrote audio: {output_audio_path}")
    if mel_path is not None:
        print(f"[generate.py] wrote mel spectrogram: {mel_path}")
    print(
        "[generate.py] summary: "
        f"prompt_blocks={text_prompt.shape[1]}, generated_blocks={int(sample_latents.shape[0])}, "
        f"sample_rate={output_sample_rate}"
    )


if __name__ == "__main__":
    main()
