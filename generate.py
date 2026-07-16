from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

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
        description="Generate speech with autoregressive PrismTTS from raw text + prompt audio."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="PrismTTS checkpoint path.")
    parser.add_argument("--model-config", type=Path, default=Path("config/model.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("config/data.yaml"))
    parser.add_argument("--text", type=str, required=True, help="Target text to synthesize.")
    parser.add_argument("--prompt-audio", type=Path, required=True, help="Prompt/reference audio (.wav).")
    parser.add_argument("--prompt-text", type=str, default="", help="Transcript of prompt audio.")
    parser.add_argument("--output", type=Path, default=Path("synth/output.wav"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=("float32", "float16", "bfloat16"),
    )
    parser.add_argument("--mimi-model", type=str, default="kyutai/mimi")
    parser.add_argument("--mimi-revision", type=str, default="main")
    parser.add_argument(
        "--mimi-num-quantizers",
        type=int,
        default=32,
        help="Number of Mimi RVQ codebooks used to reconstruct the continuous prompt latent.",
    )
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-new-frames",
        type=int,
        default=None,
        help="Override max generated frames. If omitted, estimated from text/prompt ratio.",
    )
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--trailing-pad-frames", type=int, default=0)
    parser.add_argument(
        "--eos-threshold",
        type=float,
        default=None,
        help="Override the model's EOS stopping probability threshold.",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=None,
        help="Override the number of consistency sampling steps per frame.",
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
    _, eos_token_id, _, text_token_offset = build_shared_token_layout(discrete_token_count)

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
            num_quantizers=int(args.mimi_num_quantizers),
            return_dict=True,
        )
        prompt_codes = encoded.audio_codes
        if prompt_codes is None:
            raise RuntimeError("Mimi encode did not return audio_codes.")
        prompt_latents = mimi_model.quantizer.decode(prompt_codes)  # [B, C, T]

    raw_prompt_continuous = prompt_latents[0].transpose(0, 1).to(dtype=torch.float32).cpu()  # [T, C]

    prompt_text_tokens = generate_utils.safe_tokenize(tokenizer, args.prompt_text, "prompt_text")
    target_text_tokens = generate_utils.safe_tokenize(tokenizer, args.text, "text")
    if target_text_tokens.numel() == 0:
        raise ValueError("Target text is empty after tokenization.")

    max_new_frames = args.max_new_frames
    if max_new_frames is None:
        max_new_frames = generate_utils.estimate_max_new_blocks(
            target_text_token_count=int(target_text_tokens.numel()),
            prompt_text_token_count=int(prompt_text_tokens.numel()),
            prompt_frame_count=int(raw_prompt_continuous.shape[0]),
            duration_scale=float(args.duration_scale),
            trailing_pad_blocks=int(args.trailing_pad_frames),
        )
    if max_new_frames < 1:
        raise ValueError("--max-new-frames must be >= 1.")

    text_prompt = prompt_text_tokens.unsqueeze(0).to(device=device, dtype=torch.long)
    continuous_prompt = raw_prompt_continuous.unsqueeze(0).to(device=device, dtype=model_dtype)
    text_target = target_text_tokens.unsqueeze(0).to(device=device, dtype=torch.long)

    generate_kwargs: dict = {"max_new_frames": int(max_new_frames)}
    if args.eos_threshold is not None:
        generate_kwargs["eos_threshold"] = float(args.eos_threshold)
    if args.sample_steps is not None:
        generate_kwargs["num_sample_steps"] = int(args.sample_steps)

    with torch.no_grad():
        generation = model.generate(
            text_prompt=text_prompt,
            continuous_prompt=continuous_prompt,
            text_target=text_target,
            text_prompt_lengths=torch.tensor([text_prompt.shape[1]], device=device, dtype=torch.long),
            speech_prompt_lengths=torch.tensor(
                [raw_prompt_continuous.shape[0]], device=device, dtype=torch.long
            ),
            text_target_lengths=torch.tensor([text_target.shape[1]], device=device, dtype=torch.long),
            return_dict=True,
            **generate_kwargs,
        )

    predicted_latents = generation.continuous_latents
    if predicted_latents is None or predicted_latents.shape[0] == 0 or predicted_latents.shape[1] == 0:
        raise RuntimeError("Generation produced no continuous latents.")
    sample_latents = predicted_latents[0]

    latent_std = float(sample_latents.std(unbiased=False).item())
    if sample_latents.shape[0] > 1:
        latent_delta_std = float((sample_latents[1:] - sample_latents[:-1]).std(unbiased=False).item())
    else:
        latent_delta_std = 0.0
    print(f"[generate.py] latent summary: std={latent_std:.6f}, delta_std={latent_delta_std:.6f}")

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
        f"prompt_text_tokens={text_prompt.shape[1]}, "
        f"generated_frames={int(sample_latents.shape[0])}, sample_rate={output_sample_rate}"
    )


if __name__ == "__main__":
    main()
