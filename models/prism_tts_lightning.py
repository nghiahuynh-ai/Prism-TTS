from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from models.prism_tts import PrismTTS, PrismTTSGenerationOutput, PrismTTSOutput

try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import WandbLogger
except ModuleNotFoundError:
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.loggers import WandbLogger
    except ModuleNotFoundError:
        pl = None
        WandbLogger = None


SchedulerFactory = Callable[[torch.optim.Optimizer], Any]
AudioDecoder = Callable[[torch.FloatTensor], torch.Tensor | np.ndarray]


@dataclass
class PrismBatch:
    text: Optional[torch.LongTensor] = None
    discrete: Optional[torch.LongTensor] = None
    continuous: Optional[torch.FloatTensor] = None
    prompt_lengths: Optional[torch.LongTensor] = None
    target_lengths: Optional[torch.LongTensor] = None
    text_target: Optional[torch.LongTensor] = None
    discrete_target: Optional[torch.LongTensor] = None
    continuous_target: Optional[torch.FloatTensor] = None
    text_prompt: Optional[torch.LongTensor] = None
    discrete_prompt: Optional[torch.LongTensor] = None
    continuous_prompt: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None
    flow_timesteps: Optional[torch.FloatTensor] = None
    noise: Optional[torch.FloatTensor] = None


if pl is None:

    class PrismTTSLightning(nn.Module):
        def __init__(self, *_: Any, **__: Any) -> None:
            super().__init__()
            raise ImportError(
                "PrismTTSLightning requires `lightning` or `pytorch_lightning` to be installed."
            )

else:

    class PrismTTSLightning(pl.LightningModule):
        """
        Lightning wrapper for PrismTTS with step-level training logs and periodic
        qualitative evaluation logs to Weights & Biases.
        """

        def __init__(
            self,
            model: PrismTTS,
            learning_rate: float = 3e-4,
            weight_decay: float = 0.01,
            betas: tuple[float, float] = (0.9, 0.95),
            eval_every_n_steps: int = 5000,
            scheduler_factory: Optional[SchedulerFactory] = None,
            audio_decoder: Optional[AudioDecoder] = None,
            audio_sample_rate: int = 24_000,
            max_audio_samples: int = 2,
            ema_decay: float = 0.999,
            ema_start_step: int = 0,
            ema_update_every_n_steps: int = 1,
            ema_warmup_steps: int = 0,
            use_ema_for_validation: bool = True,
            use_ema_for_periodic_eval: bool = True,
            sync_dist_logging: bool = False,
        ) -> None:
            super().__init__()
            if eval_every_n_steps < 1:
                raise ValueError("eval_every_n_steps must be >= 1.")
            if audio_sample_rate < 1:
                raise ValueError("audio_sample_rate must be >= 1.")
            if max_audio_samples < 1:
                raise ValueError("max_audio_samples must be >= 1.")
            if not (0.0 < ema_decay < 1.0):
                raise ValueError("ema_decay must be in (0, 1).")
            if ema_start_step < 0:
                raise ValueError("ema_start_step must be >= 0.")
            if ema_update_every_n_steps < 1:
                raise ValueError("ema_update_every_n_steps must be >= 1.")
            if ema_warmup_steps < 0:
                raise ValueError("ema_warmup_steps must be >= 0.")

            self.model = model
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.betas = betas
            self.eval_every_n_steps = eval_every_n_steps
            self.scheduler_factory = scheduler_factory
            self.audio_decoder = audio_decoder
            self.audio_sample_rate = audio_sample_rate
            self.max_audio_samples = max_audio_samples
            self.ema_decay = ema_decay
            self.ema_start_step = ema_start_step
            self.ema_update_every_n_steps = ema_update_every_n_steps
            self.ema_warmup_steps = ema_warmup_steps
            self.use_ema_for_validation = use_ema_for_validation
            self.use_ema_for_periodic_eval = use_ema_for_periodic_eval
            self.sync_dist_logging = sync_dist_logging

            self._eval_loader_ref: Optional[Any] = None
            self._eval_loader_iter: Optional[Iterator[Any]] = None
            self._periodic_eval_active = False
            self._audio_skip_logged = False
            self._ema_state: dict[str, torch.Tensor] = {}
            self._ema_updates = 0
            self._last_ema_step = -1
            self._ema_validation_backup: Optional[dict[str, torch.Tensor]] = None

            self.save_hyperparameters(
                {
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "betas": betas,
                    "eval_every_n_steps": eval_every_n_steps,
                    "audio_sample_rate": audio_sample_rate,
                    "max_audio_samples": max_audio_samples,
                    "ema_decay": ema_decay,
                    "ema_start_step": ema_start_step,
                    "ema_update_every_n_steps": ema_update_every_n_steps,
                    "ema_warmup_steps": ema_warmup_steps,
                    "use_ema_for_validation": use_ema_for_validation,
                    "use_ema_for_periodic_eval": use_ema_for_periodic_eval,
                    "sync_dist_logging": sync_dist_logging,
                }
            )

        def forward(self, **kwargs: Any) -> PrismTTSOutput:
            return self.model(return_dict=True, **kwargs)

        def configure_optimizers(self) -> Any:
            optimizer = AdamW(
                self.parameters(),
                lr=self.learning_rate,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
            if self.scheduler_factory is None:
                return optimizer

            scheduler = self.scheduler_factory(optimizer)
            if isinstance(scheduler, dict):
                return {"optimizer": optimizer, "lr_scheduler": scheduler}

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
            del batch_idx
            batch_inputs = self._parse_batch(batch)
            outputs = self._forward_batch(batch_inputs)

            batch_size = self._batch_size(batch_inputs)
            self.log(
                "train/loss",
                outputs.loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=self.sync_dist_logging,
            )
            self.log(
                "train/discrete_loss",
                outputs.discrete_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=self.sync_dist_logging,
            )
            self.log(
                "train/flow_loss",
                outputs.flow_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=self.sync_dist_logging,
            )
            if outputs.text_loss is not None:
                self.log(
                    "train/text_loss",
                    outputs.text_loss,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=True,
                    batch_size=batch_size,
                    sync_dist=self.sync_dist_logging,
                )
                text_ppl = torch.exp(outputs.text_loss.detach().clamp(max=20.0))
                self.log(
                    "train/text_ppl",
                    text_ppl,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=False,
                    batch_size=batch_size,
                    sync_dist=self.sync_dist_logging,
                )

            discrete_ppl = torch.exp(outputs.discrete_loss.detach().clamp(max=20.0))
            self.log(
                "train/discrete_ppl",
                discrete_ppl,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
                sync_dist=self.sync_dist_logging,
            )

            current_lr = self._current_lr()
            if current_lr is not None:
                self.log(
                    "train/lr",
                    current_lr,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=False,
                    batch_size=batch_size,
                    sync_dist=False,
                )

            return outputs.loss

        def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
            del batch_idx
            batch_inputs = self._parse_batch(batch)
            outputs = self._forward_batch(batch_inputs)
            batch_size = self._batch_size(batch_inputs)

            self.log(
                "val/loss",
                outputs.loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=self.sync_dist_logging,
            )
            self.log(
                "val/discrete_loss",
                outputs.discrete_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=self.sync_dist_logging,
            )
            self.log(
                "val/flow_loss",
                outputs.flow_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=self.sync_dist_logging,
            )
            if outputs.text_loss is not None:
                self.log(
                    "val/text_loss",
                    outputs.text_loss,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size,
                    sync_dist=self.sync_dist_logging,
                )
            return outputs.loss

        def on_fit_start(self) -> None:
            self._initialize_ema_if_needed()

        def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
            del outputs, batch, batch_idx
            self._maybe_update_ema()
            if self._periodic_eval_active:
                return
            trainer = self.trainer
            if trainer is None or trainer.sanity_checking:
                return
            if self.global_step < 1:
                return
            if self.global_step % self.eval_every_n_steps != 0:
                return

            self._periodic_eval_active = True
            try:
                self._run_periodic_eval()
            finally:
                self._periodic_eval_active = False

        def on_validation_epoch_start(self) -> None:
            self._maybe_apply_ema_for_validation()

        def on_validation_epoch_end(self) -> None:
            self._maybe_restore_after_validation()

        def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
            if not self._ema_state:
                return
            checkpoint["ema_state"] = {
                name: tensor.detach().cpu() for name, tensor in self._ema_state.items()
            }
            checkpoint["ema_updates"] = int(self._ema_updates)
            checkpoint["ema_last_step"] = int(self._last_ema_step)

        def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
            state = checkpoint.get("ema_state")
            if isinstance(state, Mapping):
                self._ema_state = {
                    str(name): tensor.detach().clone()
                    for name, tensor in state.items()
                    if torch.is_tensor(tensor)
                }
            self._ema_updates = int(checkpoint.get("ema_updates", 0))
            self._last_ema_step = int(checkpoint.get("ema_last_step", -1))

        def _forward_batch(self, batch_inputs: PrismBatch) -> PrismTTSOutput:
            if (
                batch_inputs.text is not None
                and batch_inputs.discrete is not None
                and batch_inputs.continuous is not None
            ):
                return self.model(
                    text=batch_inputs.text,
                    discrete=batch_inputs.discrete,
                    continuous=batch_inputs.continuous,
                    prompt_lengths=batch_inputs.prompt_lengths,
                    target_lengths=batch_inputs.target_lengths,
                    attention_mask=batch_inputs.attention_mask,
                    flow_timesteps=batch_inputs.flow_timesteps,
                    noise=batch_inputs.noise,
                    return_dict=True,
                )

            if (
                batch_inputs.text_target is None
                or batch_inputs.discrete_target is None
                or batch_inputs.continuous_target is None
                or batch_inputs.text_prompt is None
                or batch_inputs.discrete_prompt is None
                or batch_inputs.continuous_prompt is None
            ):
                raise ValueError(
                    "Batch is missing required tensors. Expected either concatenated "
                    "inputs (text/discrete/continuous/prompt_lengths) or split "
                    "prompt/target tensors."
                )

            prompt_len = int(batch_inputs.text_prompt.shape[1])
            target_len = int(batch_inputs.text_target.shape[1])
            normalized_discrete_prompt = self.model._normalize_discrete_tokens(
                batch_inputs.discrete_prompt,
                "discrete_prompt",
            )
            normalized_discrete_target = self.model._normalize_discrete_tokens(
                batch_inputs.discrete_target,
                "discrete_target",
            )
            normalized_continuous_prompt = self.model._normalize_continuous_latents(
                batch_inputs.continuous_prompt,
                expected_len=prompt_len,
                name="continuous_prompt",
            )
            normalized_continuous_target = self.model._normalize_continuous_latents(
                batch_inputs.continuous_target,
                expected_len=target_len,
                name="continuous_target",
            )
            full_text = torch.cat([batch_inputs.text_prompt, batch_inputs.text_target], dim=1)
            full_discrete = torch.cat(
                [normalized_discrete_prompt, normalized_discrete_target],
                dim=1,
            )
            full_continuous = torch.cat(
                [normalized_continuous_prompt, normalized_continuous_target],
                dim=1,
            )
            batch_size = int(full_text.shape[0])
            prompt_lengths = batch_inputs.prompt_lengths
            if prompt_lengths is None:
                prompt_lengths = torch.full(
                    (batch_size,),
                    prompt_len,
                    dtype=torch.long,
                    device=full_text.device,
                )
            target_lengths = batch_inputs.target_lengths
            if target_lengths is None:
                target_lengths = torch.full(
                    (batch_size,),
                    target_len,
                    dtype=torch.long,
                    device=full_text.device,
                )

            return self.model(
                text=full_text,
                discrete=full_discrete,
                continuous=full_continuous,
                prompt_lengths=prompt_lengths,
                target_lengths=target_lengths,
                attention_mask=batch_inputs.attention_mask,
                flow_timesteps=batch_inputs.flow_timesteps,
                noise=batch_inputs.noise,
                return_dict=True,
            )

        def _parse_batch(self, batch: Any) -> PrismBatch:
            if isinstance(batch, Mapping):
                return self._parse_mapping_batch(batch)
            if isinstance(batch, (tuple, list)):
                return self._parse_sequence_batch(batch)
            raise TypeError(
                "Batch must be a mapping or a sequence with required PrismTTS tensors."
            )

        def _parse_mapping_batch(self, batch: Mapping[str, Any]) -> PrismBatch:
            concatenated_required = (
                "text",
                "discrete",
                "continuous",
                "prompt_lengths",
            )
            if all(key in batch for key in concatenated_required):
                return PrismBatch(
                    text=batch["text"],
                    discrete=batch["discrete"],
                    continuous=batch["continuous"],
                    prompt_lengths=batch["prompt_lengths"],
                    target_lengths=batch.get("target_lengths"),
                    text_target=batch.get("text_target"),
                    discrete_target=batch.get("discrete_target"),
                    continuous_target=batch.get("continuous_target"),
                    text_prompt=batch.get("text_prompt"),
                    discrete_prompt=batch.get("discrete_prompt"),
                    continuous_prompt=batch.get("continuous_prompt"),
                    attention_mask=batch.get("attention_mask"),
                    flow_timesteps=batch.get("flow_timesteps"),
                    noise=batch.get("noise"),
                )

            required = (
                "text_target",
                "discrete_target",
                "continuous_target",
                "text_prompt",
                "discrete_prompt",
                "continuous_prompt",
            )
            if all(key in batch for key in required):
                return PrismBatch(
                    text_target=batch["text_target"],
                    discrete_target=batch["discrete_target"],
                    continuous_target=batch["continuous_target"],
                    text_prompt=batch["text_prompt"],
                    discrete_prompt=batch["discrete_prompt"],
                    continuous_prompt=batch["continuous_prompt"],
                    prompt_lengths=batch.get("prompt_lengths"),
                    target_lengths=batch.get("target_lengths"),
                    attention_mask=batch.get("attention_mask"),
                    flow_timesteps=batch.get("flow_timesteps"),
                    noise=batch.get("noise"),
                )

            if "prompt" in batch and "target" in batch:
                prompt = batch["prompt"]
                target = batch["target"]
                if not isinstance(prompt, Mapping) or not isinstance(target, Mapping):
                    raise TypeError("`prompt` and `target` must be mappings.")

                return PrismBatch(
                    text_target=target["text"],
                    discrete_target=target["discrete"],
                    continuous_target=target["continuous"],
                    text_prompt=prompt["text"],
                    discrete_prompt=prompt["discrete"],
                    continuous_prompt=prompt["continuous"],
                    prompt_lengths=batch.get("prompt_lengths"),
                    target_lengths=batch.get("target_lengths"),
                    attention_mask=batch.get("attention_mask"),
                    flow_timesteps=batch.get("flow_timesteps"),
                    noise=batch.get("noise"),
                )

            raise KeyError(
                "Mapping batch is missing PrismTTS keys. Required keys: "
                "either (text, discrete, continuous, prompt_lengths) or "
                "(text_target, discrete_target, continuous_target, text_prompt, "
                "discrete_prompt, continuous_prompt)."
            )

        def _parse_sequence_batch(self, batch: list[Any] | tuple[Any, ...]) -> PrismBatch:
            if len(batch) < 6:
                raise ValueError(
                    "Sequence batch must provide at least 6 tensors in order: "
                    "text_target, discrete_target, continuous_target, text_prompt, "
                    "discrete_prompt, continuous_prompt."
                )

            attention_mask = batch[6] if len(batch) > 6 else None
            flow_timesteps = batch[7] if len(batch) > 7 else None
            noise = batch[8] if len(batch) > 8 else None

            return PrismBatch(
                text_target=batch[0],
                discrete_target=batch[1],
                continuous_target=batch[2],
                text_prompt=batch[3],
                discrete_prompt=batch[4],
                continuous_prompt=batch[5],
                attention_mask=attention_mask,
                flow_timesteps=flow_timesteps,
                noise=noise,
            )

        @staticmethod
        def _batch_size(batch_inputs: PrismBatch) -> int:
            for tensor in (
                batch_inputs.text,
                batch_inputs.text_target,
                batch_inputs.text_prompt,
            ):
                if tensor is not None:
                    return int(tensor.shape[0])
            raise ValueError("Unable to infer batch size from PrismBatch.")

        @staticmethod
        def _max_target_length(batch_inputs: PrismBatch) -> int:
            if batch_inputs.target_lengths is not None:
                return int(torch.as_tensor(batch_inputs.target_lengths).max().item())
            if batch_inputs.text_target is not None:
                return int(batch_inputs.text_target.shape[1])
            return 0

        def _current_lr(self) -> Optional[float]:
            trainer = self.trainer
            if trainer is None or not trainer.optimizers:
                return None
            return float(trainer.optimizers[0].param_groups[0]["lr"])

        def _run_periodic_eval(self) -> None:
            eval_batch = self._next_eval_batch()
            if eval_batch is None:
                return

            eval_batch = self._move_to_device(eval_batch, self.device)
            batch_inputs = self._parse_batch(eval_batch)
            if (
                batch_inputs.text_prompt is None
                or batch_inputs.discrete_prompt is None
                or batch_inputs.continuous_prompt is None
                or batch_inputs.text_target is None
            ):
                return
            target_len = self._max_target_length(batch_inputs)
            if target_len < 1:
                return

            was_training = self.model.training
            try:
                with self._ema_scope(enabled=self.use_ema_for_periodic_eval):
                    self.model.eval()
                    with torch.no_grad():
                        eval_outputs = self._forward_batch(batch_inputs)
                        generation = self.model.generate_with_kv_cache(
                            text_prompt=batch_inputs.text_prompt,
                            discrete_prompt=batch_inputs.discrete_prompt,
                            continuous_prompt=batch_inputs.continuous_prompt,
                            text_target=batch_inputs.text_target,
                            max_new_blocks=target_len,
                            do_sample=False,
                            return_dict=True,
                        )
            finally:
                if was_training:
                    self.model.train()

            viz = self._build_eval_visuals(batch_inputs, generation)
            batch_size = self._batch_size(batch_inputs)
            self.log(
                "eval/loss",
                eval_outputs.loss,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
                sync_dist=self.sync_dist_logging,
            )

            if not self._is_global_zero():
                return
            self._log_eval_media_to_wandb(
                batch_inputs=batch_inputs,
                generation=generation,
                eval_outputs=eval_outputs,
                viz=viz,
            )

        def _next_eval_batch(self) -> Optional[Any]:
            val_loader = self._primary_val_loader()
            if val_loader is None:
                return None

            if self._eval_loader_ref is not val_loader:
                self._eval_loader_ref = val_loader
                self._eval_loader_iter = iter(val_loader)

            if self._eval_loader_iter is None:
                self._eval_loader_iter = iter(val_loader)

            try:
                return next(self._eval_loader_iter)
            except StopIteration:
                self._eval_loader_iter = iter(val_loader)
                try:
                    return next(self._eval_loader_iter)
                except StopIteration:
                    return None

        def _primary_val_loader(self) -> Optional[Any]:
            trainer = self.trainer
            if trainer is None:
                return None

            val_loaders = trainer.val_dataloaders
            if val_loaders is None:
                return None
            if isinstance(val_loaders, (list, tuple)):
                return val_loaders[0] if len(val_loaders) > 0 else None
            return val_loaders

        def _build_eval_visuals(
            self,
            batch_inputs: PrismBatch,
            generation: PrismTTSGenerationOutput,
        ) -> dict[str, np.ndarray]:
            if batch_inputs.continuous_target is None or batch_inputs.text_target is None:
                return {
                    "target_latent": np.zeros((1, 1), dtype=np.float32),
                    "pred_latent": np.zeros((1, 1), dtype=np.float32),
                    "abs_error": np.zeros((1, 1), dtype=np.float32),
                }
            target_continuous = self.model._normalize_continuous_latents(
                batch_inputs.continuous_target,
                expected_len=batch_inputs.text_target.shape[1],
                name="continuous_target",
            )
            pred_continuous = generation.continuous_latents

            target_len = min(
                int(target_continuous.shape[1]),
                int(pred_continuous.shape[1]),
            )
            if target_len < 1:
                return {
                    "target_latent": np.zeros((1, 1), dtype=np.float32),
                    "pred_latent": np.zeros((1, 1), dtype=np.float32),
                    "abs_error": np.zeros((1, 1), dtype=np.float32),
                }

            target_continuous = target_continuous[:, :target_len, :]
            pred_continuous = pred_continuous[:, :target_len, :]
            valid_mask = self._resolve_generated_mask(
                attention_mask=batch_inputs.attention_mask,
                target_len=target_len,
                batch_size=int(target_continuous.shape[0]),
                device=target_continuous.device,
            )

            sample_mask = valid_mask[0]
            target_latent = target_continuous[0][sample_mask].detach().float().cpu().numpy()
            pred_latent = pred_continuous[0][sample_mask].detach().float().cpu().numpy()
            if target_latent.shape[0] == 0:
                target_latent = target_continuous[0].detach().float().cpu().numpy()[:1]
                pred_latent = pred_continuous[0].detach().float().cpu().numpy()[:1]

            abs_error = np.abs(pred_latent - target_latent)
            return {
                "target_latent": target_latent,
                "pred_latent": pred_latent,
                "abs_error": abs_error,
            }

        def _resolve_generated_mask(
            self,
            attention_mask: Optional[torch.Tensor],
            target_len: int,
            batch_size: int,
            device: torch.device,
        ) -> torch.Tensor:
            default_mask = torch.ones(batch_size, target_len, dtype=torch.bool, device=device)
            if attention_mask is None or attention_mask.dim() != 2:
                return default_mask

            if attention_mask.shape[1] == target_len:
                return attention_mask.to(device=device, dtype=torch.bool)

            token_len = target_len * self.model.block_size
            if attention_mask.shape[1] == token_len:
                return attention_mask.to(device=device, dtype=torch.bool).reshape(
                    batch_size,
                    target_len,
                    self.model.block_size,
                ).all(dim=-1)

            if attention_mask.shape[1] > target_len:
                return attention_mask[:, -target_len:].to(device=device, dtype=torch.bool)

            return default_mask

        def _initialize_ema_if_needed(self) -> None:
            if self._ema_state:
                return
            with torch.no_grad():
                for name, param in self._ema_parameters():
                    self._ema_state[name] = param.detach().clone().float()

        def _maybe_update_ema(self) -> None:
            trainer = self.trainer
            if trainer is None or trainer.sanity_checking:
                return

            step = int(self.global_step)
            effective_start = max(1, self.ema_start_step)
            if step < effective_start:
                return
            if step % self.ema_update_every_n_steps != 0:
                return
            if step == self._last_ema_step:
                return

            self._initialize_ema_if_needed()
            decay = self._ema_decay_for_step(step)
            with torch.no_grad():
                for name, param in self._ema_parameters():
                    shadow = self._ema_state.get(name)
                    if shadow is None:
                        self._ema_state[name] = param.detach().clone().float()
                        continue
                    if shadow.device != param.device:
                        shadow = shadow.to(device=param.device)
                        self._ema_state[name] = shadow
                    shadow.mul_(decay).add_(param.detach().float(), alpha=1.0 - decay)

            self._ema_updates += 1
            self._last_ema_step = step
            self.log(
                "train/ema_decay",
                decay,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=1,
                sync_dist=False,
            )

        def _ema_decay_for_step(self, step: int) -> float:
            if self.ema_warmup_steps <= 0:
                return self.ema_decay
            effective_start = max(1, self.ema_start_step)
            progress = float(step - effective_start + 1) / float(self.ema_warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return self.ema_decay * progress

        def _maybe_apply_ema_for_validation(self) -> None:
            if not self.use_ema_for_validation:
                return
            if self._ema_validation_backup is not None:
                return
            self._initialize_ema_if_needed()
            if not self._ema_state:
                return
            self._ema_validation_backup = self._backup_current_parameters()
            self._copy_ema_to_model()

        def _maybe_restore_after_validation(self) -> None:
            if self._ema_validation_backup is None:
                return
            self._restore_parameters(self._ema_validation_backup)
            self._ema_validation_backup = None

        @contextmanager
        def _ema_scope(self, enabled: bool) -> Iterator[None]:
            if not enabled or not self._ema_state:
                yield
                return
            backup = self._backup_current_parameters()
            self._copy_ema_to_model()
            try:
                yield
            finally:
                self._restore_parameters(backup)

        def _backup_current_parameters(self) -> dict[str, torch.Tensor]:
            with torch.no_grad():
                return {
                    name: param.detach().clone()
                    for name, param in self._ema_parameters()
                }

        def _restore_parameters(self, backup: Mapping[str, torch.Tensor]) -> None:
            with torch.no_grad():
                for name, param in self._ema_parameters():
                    value = backup.get(name)
                    if value is None:
                        continue
                    if value.device != param.device:
                        value = value.to(device=param.device)
                    param.copy_(value.to(dtype=param.dtype))

        def _copy_ema_to_model(self) -> None:
            with torch.no_grad():
                for name, param in self._ema_parameters():
                    shadow = self._ema_state.get(name)
                    if shadow is None:
                        continue
                    if shadow.device != param.device:
                        shadow = shadow.to(device=param.device)
                        self._ema_state[name] = shadow
                    param.copy_(shadow.to(dtype=param.dtype))

        def _ema_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]:
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if not param.dtype.is_floating_point:
                    continue
                yield name, param

        def _log_eval_media_to_wandb(
            self,
            batch_inputs: PrismBatch,
            generation: PrismTTSGenerationOutput,
            eval_outputs: PrismTTSOutput,
            viz: Mapping[str, np.ndarray],
        ) -> None:
            run = self._wandb_run()
            if run is None:
                return

            try:
                import wandb
            except ModuleNotFoundError:
                return

            payload: dict[str, Any] = {
                "eval/loss": float(eval_outputs.loss.detach().cpu()),
            }
            if eval_outputs.text_loss is not None:
                payload["eval/text_loss"] = float(eval_outputs.text_loss.detach().cpu())

            figure = self._build_professional_eval_figure(
                target_latent=viz["target_latent"],
                pred_latent=viz["pred_latent"],
                abs_error=viz["abs_error"],
                step=int(self.global_step),
            )
            if figure is not None:
                payload["eval/qualitative"] = wandb.Image(figure)
                try:
                    import matplotlib.pyplot as plt

                    plt.close(figure)
                except ModuleNotFoundError:
                    figure.clf()

            audio_table = self._build_audio_table(batch_inputs, generation)
            if audio_table is not None:
                payload["eval/audio_samples"] = audio_table
            elif self.audio_decoder is None and not self._audio_skip_logged:
                payload["eval/audio_logging_skipped"] = 1.0
                self._audio_skip_logged = True

            run.log(payload, step=int(self.global_step))

        def _build_professional_eval_figure(
            self,
            target_latent: np.ndarray,
            pred_latent: np.ndarray,
            abs_error: np.ndarray,
            step: int,
        ) -> Any:
            try:
                import matplotlib.pyplot as plt
            except ModuleNotFoundError:
                return None

            plt.style.use("seaborn-v0_8-whitegrid")
            fig = plt.figure(figsize=(14, 4.8), dpi=140, constrained_layout=True)
            grid = fig.add_gridspec(1, 3)

            value_min = float(min(np.min(target_latent), np.min(pred_latent)))
            value_max = float(max(np.max(target_latent), np.max(pred_latent)))
            if np.isclose(value_min, value_max):
                value_min -= 1e-3
                value_max += 1e-3

            error_max = float(max(np.max(abs_error), 1e-6))

            ax_target = fig.add_subplot(grid[0, 0])
            target_img = ax_target.imshow(
                target_latent.T,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=value_min,
                vmax=value_max,
            )
            ax_target.set_title("Ground Truth Continuous Latents", fontsize=12, weight="bold")
            ax_target.set_xlabel("Time Step")
            ax_target.set_ylabel("Latent Channel")
            fig.colorbar(target_img, ax=ax_target, fraction=0.046, pad=0.02)

            ax_pred = fig.add_subplot(grid[0, 1])
            pred_img = ax_pred.imshow(
                pred_latent.T,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=value_min,
                vmax=value_max,
            )
            ax_pred.set_title("Synthesized Continuous Latents", fontsize=12, weight="bold")
            ax_pred.set_xlabel("Time Step")
            ax_pred.set_ylabel("Latent Channel")
            fig.colorbar(pred_img, ax=ax_pred, fraction=0.046, pad=0.02)

            ax_err = fig.add_subplot(grid[0, 2])
            err_img = ax_err.imshow(
                abs_error.T,
                aspect="auto",
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=error_max,
            )
            ax_err.set_title("Absolute Error |Synth - GT|", fontsize=11, weight="bold")
            ax_err.set_xlabel("Time Step")
            ax_err.set_ylabel("Latent Channel")
            fig.colorbar(err_img, ax=ax_err, fraction=0.046, pad=0.02)

            fig.suptitle(
                f"Prism-TTS Periodic Evaluation Preview (Step {step})",
                fontsize=15,
                weight="bold",
            )
            return fig

        def _build_audio_table(
            self,
            batch_inputs: PrismBatch,
            generation: PrismTTSGenerationOutput,
        ) -> Optional[Any]:
            if self.audio_decoder is None:
                return None
            if batch_inputs.continuous_target is None or batch_inputs.text_target is None:
                return None

            try:
                import wandb
            except ModuleNotFoundError:
                return None

            target_continuous = self.model._normalize_continuous_latents(
                batch_inputs.continuous_target,
                expected_len=batch_inputs.text_target.shape[1],
                name="continuous_target",
            )
            pred_continuous = generation.continuous_latents

            target_len = min(int(target_continuous.shape[1]), int(pred_continuous.shape[1]))
            if target_len < 1:
                return None

            target_continuous = target_continuous[:, :target_len, :]
            pred_continuous = pred_continuous[:, :target_len, :]

            # Keep eval-stage media payload small and deterministic: log exactly one sample.
            num_samples = min(1, self.max_audio_samples, int(target_continuous.shape[0]))
            table = wandb.Table(
                columns=[
                    "sample_id",
                    "target_audio",
                    "target_mel_spectrogram",
                    "synthesized_audio",
                    "synthesized_mel_spectrogram",
                ]
            )

            added = 0
            for sample_idx in range(num_samples):
                target_audio = self._decode_audio(target_continuous[sample_idx])
                synth_audio = self._decode_audio(pred_continuous[sample_idx])
                if target_audio is None or synth_audio is None:
                    continue

                target_mel = self._build_mel_spectrogram_image(
                    target_audio,
                    title=f"Target Mel Spectrogram (sample {sample_idx})",
                )
                synth_mel = self._build_mel_spectrogram_image(
                    synth_audio,
                    title=f"Synthesized Mel Spectrogram (sample {sample_idx})",
                )

                table.add_data(
                    sample_idx,
                    wandb.Audio(
                        target_audio,
                        sample_rate=self.audio_sample_rate,
                        caption=f"target_sample_{sample_idx}",
                    ),
                    target_mel,
                    wandb.Audio(
                        synth_audio,
                        sample_rate=self.audio_sample_rate,
                        caption=f"synth_sample_{sample_idx}",
                    ),
                    synth_mel,
                )
                added += 1

            return table if added > 0 else None

        def _build_mel_spectrogram_image(self, audio: np.ndarray, *, title: str) -> Optional[Any]:
            try:
                import matplotlib.pyplot as plt
                import wandb
            except ModuleNotFoundError:
                return None

            log_mel = self._compute_log_mel_spectrogram(audio)
            if log_mel is None:
                return None

            fig, ax = plt.subplots(figsize=(6.0, 2.8), dpi=130, constrained_layout=True)
            mel_img = ax.imshow(
                log_mel,
                origin="lower",
                aspect="auto",
                cmap="magma",
            )
            ax.set_title(title, fontsize=10, weight="bold")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Mel Bin")
            fig.colorbar(mel_img, ax=ax, fraction=0.046, pad=0.02)

            image = wandb.Image(fig)
            plt.close(fig)
            return image

        def _compute_log_mel_spectrogram(self, audio: np.ndarray) -> Optional[np.ndarray]:
            waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
            if waveform.size < 2:
                return None

            waveform_tensor = torch.from_numpy(waveform)
            n_fft = 1024
            win_length = 1024
            hop_length = 256
            n_mels = 80

            window = torch.hann_window(win_length, dtype=torch.float32, device=waveform_tensor.device)
            stft = torch.stft(
                waveform_tensor,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
                center=True,
            )
            power_spec = stft.abs().pow(2.0)
            if power_spec.numel() == 0:
                return None

            mel_filter = self._build_mel_filter_bank(
                sample_rate=self.audio_sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                f_min=0.0,
                f_max=float(self.audio_sample_rate) * 0.5,
                device=power_spec.device,
            )

            mel_power = mel_filter @ power_spec
            log_mel = torch.log10(mel_power.clamp_min(1e-10))
            return log_mel.detach().cpu().numpy()

        @staticmethod
        def _build_mel_filter_bank(
            *,
            sample_rate: int,
            n_fft: int,
            n_mels: int,
            f_min: float,
            f_max: float,
            device: torch.device,
        ) -> torch.Tensor:
            if sample_rate < 1:
                raise ValueError("sample_rate must be >= 1 for mel filter bank construction.")
            if n_fft < 2:
                raise ValueError("n_fft must be >= 2 for mel filter bank construction.")
            if n_mels < 1:
                raise ValueError("n_mels must be >= 1 for mel filter bank construction.")

            nyquist = float(sample_rate) * 0.5
            f_min = max(0.0, float(f_min))
            f_max = min(max(f_min + 1e-6, float(f_max)), nyquist)
            n_freqs = n_fft // 2 + 1

            def hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
                return 2595.0 * np.log10(1.0 + (freq_hz / 700.0))

            def mel_to_hz(freq_mel: np.ndarray) -> np.ndarray:
                return 700.0 * (np.power(10.0, freq_mel / 2595.0) - 1.0)

            mel_points = np.linspace(
                hz_to_mel(np.array([f_min], dtype=np.float32))[0],
                hz_to_mel(np.array([f_max], dtype=np.float32))[0],
                num=n_mels + 2,
                dtype=np.float32,
            )
            hz_points = mel_to_hz(mel_points)
            fft_bins = np.floor(((n_fft + 1) * hz_points) / float(sample_rate)).astype(np.int64)
            fft_bins = np.clip(fft_bins, 0, n_freqs - 1)

            filters = np.zeros((n_mels, n_freqs), dtype=np.float32)
            for mel_idx in range(1, n_mels + 1):
                left = int(fft_bins[mel_idx - 1])
                center = int(fft_bins[mel_idx])
                right = int(fft_bins[mel_idx + 1])

                if center <= left:
                    center = min(left + 1, n_freqs - 1)
                if right <= center:
                    right = min(center + 1, n_freqs)

                if center > left:
                    filters[mel_idx - 1, left:center] = np.linspace(
                        0.0,
                        1.0,
                        num=center - left,
                        endpoint=False,
                        dtype=np.float32,
                    )
                if right > center:
                    filters[mel_idx - 1, center:right] = np.linspace(
                        1.0,
                        0.0,
                        num=right - center,
                        endpoint=False,
                        dtype=np.float32,
                    )

            return torch.tensor(filters, dtype=torch.float32, device=device)

        def _decode_audio(self, latents: torch.Tensor) -> Optional[np.ndarray]:
            if self.audio_decoder is None:
                return None

            decoded = self.audio_decoder(latents.detach())
            if isinstance(decoded, torch.Tensor):
                decoded = decoded.detach().cpu().float().numpy()
            else:
                decoded = np.asarray(decoded, dtype=np.float32)

            decoded = np.squeeze(decoded)
            if decoded.ndim > 1:
                decoded = decoded.reshape(-1)
            if decoded.size == 0:
                return None

            decoded = np.nan_to_num(decoded, copy=False)
            peak = float(np.max(np.abs(decoded)))
            if peak > 0.0:
                decoded = decoded / peak
            return decoded.astype(np.float32)

        def _wandb_run(self) -> Optional[Any]:
            loggers = self._available_loggers()
            if not loggers:
                return None

            for logger in loggers:
                if WandbLogger is not None and isinstance(logger, WandbLogger):
                    experiment = getattr(logger, "experiment", None)
                    if experiment is not None and hasattr(experiment, "log"):
                        return experiment

            for logger in loggers:
                experiment = getattr(logger, "experiment", None)
                if experiment is not None and hasattr(experiment, "log"):
                    return experiment
            return None

        def _available_loggers(self) -> list[Any]:
            trainer = self.trainer
            if trainer is not None and getattr(trainer, "loggers", None):
                return list(trainer.loggers)
            if self.logger is not None:
                return [self.logger]
            return []

        def _is_global_zero(self) -> bool:
            trainer = self.trainer
            if trainer is None:
                return True
            return bool(trainer.is_global_zero)

        def _move_to_device(self, value: Any, device: torch.device) -> Any:
            if torch.is_tensor(value):
                return value.to(device)
            if isinstance(value, Mapping):
                return {k: self._move_to_device(v, device) for k, v in value.items()}
            if isinstance(value, tuple):
                return tuple(self._move_to_device(v, device) for v in value)
            if isinstance(value, list):
                return [self._move_to_device(v, device) for v in value]
            return value
