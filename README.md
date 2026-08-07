# Prism-TTS

## Batch generation from LibriSpeech pair metadata

`scripts/generate_from_metadata.py` generates one WAV per row of a four-column
metadata file:

```text
<target_file_id>|<target_transcript>|<prompt_file_id>|<prompt_transcript>
```

It uses the prompt utterance as the voice reference and synthesizes the target
transcript to `OUTPUT_DIR/<target_file_id>.wav`. LibriSpeech prompt files are
resolved as `<audio-root>/<split>/<speaker>/<chapter>/<prompt_file_id>.flac`.
For a metadata filename such as `metadata-test-clean.txt`, the `test-clean`
split is inferred automatically.

```bash
cd Prism-TTS
python scripts/generate_from_metadata.py \
  --checkpoint /path/to/model.ckpt \
  --metadata /astro/nghiahuynh/data/librispeech/LibriSpeech/metadata-test-clean.txt \
  --audio-root /astro/nghiahuynh/data/librispeech/LibriSpeech \
  --output-dir synth/librispeech-test-clean \
  --report synth/librispeech-test-clean/report.jsonl \
  --device cuda:0 --dtype bfloat16 --skip-existing
```

The equivalent Make target uses the same LibriSpeech paths by default:

```bash
make generate-metadata CKPT=/path/to/model.ckpt \
  METADATA_ARGS='--device cuda:0 --dtype bfloat16 --skip-existing'
```

Override `METADATA`, `METADATA_AUDIO_ROOT`, `METADATA_SPLIT`,
`METADATA_OUTPUT_DIR`, or `METADATA_REPORT` when using another split or layout.

The model and Mimi codec are loaded once for the full batch. Use `--start-index`
and `--limit` to run a subset, `--continue-on-error` to finish the remaining
rows after a failure, and `--dry-run` to print the resolved paths without
loading the models. FLAC input uses the `soundfile` dependency added to
`requirements.txt`.

## Training from pretrained weights

Use `--pretrained-weights` to initialize the model before a new training run. Unlike
`--ckpt-path`, it does not restore Lightning Trainer state, optimizer or scheduler
state, callbacks, or the global step, so the run starts at step zero.

```bash
python train.py --pretrained-weights path/to/weights.ckpt
# or
make train PRETRAINED_WEIGHTS=path/to/weights.ckpt
```

The initializer accepts a raw model state dict or a Prism-TTS Lightning checkpoint.
It loads compatible tensors by default and reports skipped missing, unexpected, and
shape-mismatched tensors, which supports architecture-changing adaptation. Add
`--pretrained-strict` to require an exact match, or `--pretrained-use-ema` to
initialize from a checkpoint's EMA weights.
