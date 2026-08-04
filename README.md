# Prism-TTS

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
