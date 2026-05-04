from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from models.prism_tts import PrismTTS
from utils import lightning_utils as LU
from utils.model_utils import (
    PrismTTSGenerationOutput,
    PrismTTSOutput,
    normalize_continuous_latents,
    normalize_discrete_tokens,
)

try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import WandbLogger
except ModuleNotFoundError:
    try:
        import pytorch_lightning as pl
    except ModuleNotFoundError:
        raise ImportError(
                "PrismTTSLightning requires `lightning` or `pytorch_lightning` to be installed."
            )
    try:
        from pytorch_lightning.loggers import WandbLogger
    except ModuleNotFoundError:
        raise ImportError(
                "PrismTTSLightning requires `WandB` to be installed."
            )


SchedulerFactory = Callable[[torch.optim.Optimizer], Any]
AudioDecoder = Callable[[torch.Tensor], torch.Tensor | np.ndarray]

PrismBatch = LU.PrismBatch
PeriodicEvalSample = LU.PeriodicEvalSample


class PrismTTSLightning(pl.LightningModule):
    """
    Lightning wrapper for PrismTTS with step-level training logs and periodic
    audio/mel evaluation media logs to Weights & Biases.
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
        log_media_on_validation_end: bool = True,
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
        decoder_sample_rate = getattr(audio_decoder, "sample_rate", None)
        if isinstance(decoder_sample_rate, (int, np.integer)) and int(decoder_sample_rate) > 0:
            audio_sample_rate = int(decoder_sample_rate)
        self.audio_sample_rate = audio_sample_rate
        self.max_audio_samples = max_audio_samples
        self.log_media_on_validation_end = log_media_on_validation_end
        self.ema_decay = ema_decay
        self.ema_start_step = ema_start_step
        self.ema_update_every_n_steps = ema_update_every_n_steps
        self.ema_warmup_steps = ema_warmup_steps
        self.use_ema_for_validation = use_ema_for_validation
        self.use_ema_for_periodic_eval = use_ema_for_periodic_eval
        self.sync_dist_logging = sync_dist_logging

        self._eval_loader_ref: Optional[Any] = None
        self._eval_loader_iter: Optional[Iterator[Any]] = None
        self._train_loader_ref: Optional[Any] = None
        self._train_loader_iter: Optional[Iterator[Any]] = None
        self._cached_text_id_to_char: Optional[dict[int, str]] = None
        self._periodic_eval_active = False
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
                "log_media_on_validation_end": log_media_on_validation_end,
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

        batch_size = LU.batch_size_from_prism_batch(batch_inputs)
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
            "train/continuous_loss",
            outputs.continuous_loss,
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
        batch_size = LU.batch_size_from_prism_batch(batch_inputs)

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
            "val/continuous_loss",
            outputs.continuous_loss,
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
        return outputs.loss

    def on_fit_start(self) -> None:
        self._initialize_ema_if_needed()

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        del outputs, batch_idx
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
            self._run_periodic_eval(train_batch=batch)
        finally:
            self._periodic_eval_active = False

    def on_validation_epoch_start(self) -> None:
        self._maybe_apply_ema_for_validation()

    def on_validation_epoch_end(self) -> None:
        self._maybe_restore_after_validation()
        if not self.log_media_on_validation_end:
            return
        if self._periodic_eval_active:
            return
        trainer = self.trainer
        if trainer is None or trainer.sanity_checking:
            return

        self._periodic_eval_active = True
        try:
            self._run_periodic_eval()
        finally:
            self._periodic_eval_active = False

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
            batch_inputs.flat_token_ids is None
            or batch_inputs.flat_continuous_values is None
            or batch_inputs.flat_token_type_ids is None
            or batch_inputs.flat_speech_stream_ids is None
            or batch_inputs.flat_target_block_ids is None
        ):
            raise ValueError(
                "Batch is missing required pre-flattened tensors: "
                "flat_token_ids, flat_continuous_values, flat_token_type_ids, "
                "flat_speech_stream_ids, flat_target_block_ids."
            )

        return self.model(
            flat_token_ids=batch_inputs.flat_token_ids,
            flat_continuous_values=batch_inputs.flat_continuous_values,
            flat_token_type_ids=batch_inputs.flat_token_type_ids,
            flat_speech_stream_ids=batch_inputs.flat_speech_stream_ids,
            flat_target_block_ids=batch_inputs.flat_target_block_ids,
            flat_target_block_counts=batch_inputs.flat_target_block_counts,
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
        required_flat = (
            "flat_token_ids",
            "flat_continuous_values",
            "flat_token_type_ids",
            "flat_speech_stream_ids",
            "flat_target_block_ids",
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
                text_prompt_lengths=batch.get("text_prompt_lengths"),
                speech_prompt_lengths=batch.get("speech_prompt_lengths"),
                text_target_lengths=batch.get("text_target_lengths"),
                speech_target_lengths=batch.get("speech_target_lengths"),
                attention_mask=batch.get("attention_mask"),
                flat_token_ids=batch.get("flat_token_ids"),
                flat_continuous_values=batch.get("flat_continuous_values"),
                flat_token_type_ids=batch.get("flat_token_type_ids"),
                flat_speech_stream_ids=batch.get("flat_speech_stream_ids"),
                flat_target_block_ids=batch.get("flat_target_block_ids"),
                flat_target_block_counts=batch.get("flat_target_block_counts"),
                flow_timesteps=batch.get("flow_timesteps"),
                noise=batch.get("noise"),
            )

        if all(key in batch for key in required_flat):
            return PrismBatch(
                attention_mask=batch.get("attention_mask"),
                flat_token_ids=batch.get("flat_token_ids"),
                flat_continuous_values=batch.get("flat_continuous_values"),
                flat_token_type_ids=batch.get("flat_token_type_ids"),
                flat_speech_stream_ids=batch.get("flat_speech_stream_ids"),
                flat_target_block_ids=batch.get("flat_target_block_ids"),
                flat_target_block_counts=batch.get("flat_target_block_counts"),
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
                text_prompt_lengths=batch.get("text_prompt_lengths"),
                speech_prompt_lengths=batch.get("speech_prompt_lengths"),
                text_target_lengths=batch.get("text_target_lengths"),
                speech_target_lengths=batch.get("speech_target_lengths"),
                attention_mask=batch.get("attention_mask"),
                flow_timesteps=batch.get("flow_timesteps"),
                noise=batch.get("noise"),
            )

        raise KeyError(
            "Mapping batch is missing PrismTTS keys. Required keys: "
            "(text_target, discrete_target, continuous_target, text_prompt, "
            "discrete_prompt, continuous_prompt) or "
            "(flat_token_ids, flat_continuous_values, flat_token_type_ids, "
            "flat_speech_stream_ids, flat_target_block_ids)."
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

    def _current_lr(self) -> Optional[float]:
        trainer = self.trainer
        if trainer is None or not trainer.optimizers:
            return None
        return float(trainer.optimizers[0].param_groups[0]["lr"])

    def _run_periodic_eval(self, train_batch: Optional[Any] = None) -> None:
        eval_batch = self._next_eval_batch()
        if train_batch is None:
            train_batch = self._next_train_batch()

        periodic_samples: list[PeriodicEvalSample] = []
        was_training = self.model.training
        try:
            with self._ema_scope(enabled=self.use_ema_for_periodic_eval):
                self.model.eval()
                with torch.no_grad():
                    eval_sample = self._synthesize_periodic_eval_sample(
                        eval_batch,
                        sample_source="validation",
                    )
                    if eval_sample is not None:
                        periodic_samples.append(eval_sample)

                    train_sample = self._synthesize_periodic_eval_sample(
                        train_batch,
                        sample_source="train",
                    )
                    if train_sample is not None:
                        periodic_samples.append(train_sample)
        finally:
            if was_training:
                self.model.train()

        if not periodic_samples:
            return
        if not self._is_global_zero():
            return
        self._log_eval_media_to_wandb(periodic_samples)

    def _synthesize_periodic_eval_sample(
        self,
        batch: Optional[Any],
        *,
        sample_source: str,
    ) -> Optional[PeriodicEvalSample]:
        if batch is None:
            return None

        batch = LU.slice_batch_to_sample(batch, sample_idx=0)
        batch = LU.move_to_device(batch, self.device)
        batch_inputs = self._parse_batch(batch)
        if (
            batch_inputs.text_prompt is None
            or batch_inputs.discrete_prompt is None
            or batch_inputs.text_target is None
            or batch_inputs.discrete_target is None
        ):
            return None
        if (not self.model.discrete_only) and (
            batch_inputs.continuous_prompt is None or batch_inputs.continuous_target is None
        ):
            return None

        prompt_speech_len = (
            int(torch.as_tensor(batch_inputs.speech_prompt_lengths)[0].item())
            if batch_inputs.speech_prompt_lengths is not None
            else int(batch_inputs.discrete_prompt.shape[-2])
        )
        prompt_text_len = (
            int(torch.as_tensor(batch_inputs.text_prompt_lengths)[0].item())
            if batch_inputs.text_prompt_lengths is not None
            else int(batch_inputs.text_prompt.shape[1])
        )
        target_speech_len = (
            int(torch.as_tensor(batch_inputs.speech_target_lengths)[0].item())
            if batch_inputs.speech_target_lengths is not None
            else int(batch_inputs.discrete_target.shape[-2])
        )
        target_text_len = (
            int(torch.as_tensor(batch_inputs.text_target_lengths)[0].item())
            if batch_inputs.text_target_lengths is not None
            else int(batch_inputs.text_target.shape[1])
        )
        if target_speech_len < 1:
            return None

        text_prompt = batch_inputs.text_prompt[:, :prompt_text_len]
        discrete_prompt = normalize_discrete_tokens(
            batch_inputs.discrete_prompt,
            "discrete_prompt",
            num_discrete_tokens=self.model.num_discrete_tokens,
        )[:, :prompt_speech_len, :]
        if self.model.discrete_only:
            continuous_prompt = torch.zeros(
                (
                    int(discrete_prompt.shape[0]),
                    int(prompt_speech_len),
                    int(self.model.continuous_latent_size),
                ),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            continuous_prompt = normalize_continuous_latents(
                batch_inputs.continuous_prompt,
                expected_len=int(batch_inputs.discrete_prompt.shape[-2]),
                name="continuous_prompt",
                continuous_latent_size=self.model.continuous_latent_size,
            )[:, :prompt_speech_len, :]
        text_target = batch_inputs.text_target[:, :target_text_len]

        generation_outputs: dict[str, PrismTTSGenerationOutput] = {}
        for generation_method in ("causal", "parallel", "parallel_stable"):
            generation_outputs[generation_method] = self.model.generate(
                text_prompt=text_prompt,
                discrete_prompt=discrete_prompt,
                continuous_prompt=continuous_prompt,
                text_target=text_target,
                text_prompt_lengths=torch.tensor([prompt_text_len], device=self.device),
                speech_prompt_lengths=torch.tensor([prompt_speech_len], device=self.device),
                text_target_lengths=torch.tensor([target_text_len], device=self.device),
                speech_target_lengths=torch.tensor([target_speech_len], device=self.device),
                max_new_blocks=target_speech_len,
                do_sample=False,
                force_silent_special_tokens=True,
                generation_method=generation_method,
                return_dict=True,
            )

        text_prompt_str = self._decode_text_tokens(text_prompt[0], length=prompt_text_len)
        text_target_str = self._decode_text_tokens(text_target[0], length=target_text_len)
        synthesized_text_by_method: dict[str, str] = {}
        for generation_method, generation in generation_outputs.items():
            generated_text = ""
            if generation.text_ids is not None and int(generation.text_ids.shape[0]) > 0:
                generated_text = self._decode_text_tokens(
                    generation.text_ids[0],
                    length=int(generation.text_ids.shape[1]),
                )
            synthesized_text_by_method[generation_method] = generated_text or text_target_str

        return PeriodicEvalSample(
            sample_source=str(sample_source),
            batch_inputs=PrismBatch(
                discrete_target=batch_inputs.discrete_target,
                continuous_target=batch_inputs.continuous_target,
                speech_target_lengths=torch.tensor([target_speech_len], device=self.device),
            ),
            generations=generation_outputs,
            text_prompt=text_prompt_str,
            text_target=text_target_str,
            synthesized_text_by_method=synthesized_text_by_method,
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

    def _next_train_batch(self) -> Optional[Any]:
        train_loader = self._primary_train_loader()
        if train_loader is None:
            return None

        if self._train_loader_ref is not train_loader:
            self._train_loader_ref = train_loader
            self._train_loader_iter = iter(train_loader)

        if self._train_loader_iter is None:
            self._train_loader_iter = iter(train_loader)

        try:
            return next(self._train_loader_iter)
        except StopIteration:
            self._train_loader_iter = iter(train_loader)
            try:
                return next(self._train_loader_iter)
            except StopIteration:
                return None

    def _primary_train_loader(self) -> Optional[Any]:
        trainer = self.trainer
        if trainer is None:
            return None

        train_loader = getattr(trainer, "train_dataloader", None)
        if train_loader is None:
            return None
        if callable(train_loader) and not isinstance(train_loader, (list, tuple, Mapping)):
            try:
                train_loader = train_loader()
            except TypeError:
                return None
        if isinstance(train_loader, (list, tuple)):
            return train_loader[0] if len(train_loader) > 0 else None
        if isinstance(train_loader, Mapping):
            for loader in train_loader.values():
                return loader
            return None
        return train_loader

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
        periodic_samples: Sequence[PeriodicEvalSample],
    ) -> None:
        if not periodic_samples:
            return
        run = self._wandb_run()
        if run is None:
            return

        audio_table = self._build_audio_table(periodic_samples)
        mel_table = self._build_mel_table(periodic_samples)
        text_table = self._build_text_table(periodic_samples)

        payload: dict[str, Any] = {}
        if audio_table is not None:
            payload["eval/audio_samples"] = audio_table
        if mel_table is not None:
            payload["eval/mel_spectrogram_samples"] = mel_table
        if text_table is not None:
            payload["eval/synthesized_text_samples"] = text_table
        if not payload:
            return

        run.log(payload, step=int(self.global_step))

    def _build_audio_table(
        self,
        periodic_samples: Sequence[PeriodicEvalSample],
    ) -> Optional[Any]:
        media_rows = self._build_eval_media_rows(periodic_samples)
        if not media_rows:
            return None

        try:
            import wandb
        except ModuleNotFoundError:
            return None

        step = int(self.global_step)
        table = wandb.Table(
            columns=[
                "step",
                "sample_source",
                "sample_id",
                "generation_method",
                "groundtruth_audio",
                "groundtruth_from_discrete_audio",
                "prior_audio",
                "predicted_audio",
            ]
        )

        for row in media_rows:
            sample_source = str(row["sample_source"])
            sample_idx = int(row["sample_id"])
            method = str(row["generation_method"])
            target_audio = np.asarray(row["target_audio"], dtype=np.float32)
            pred_audio = np.asarray(row["predicted_audio"], dtype=np.float32)
            target_discrete_audio_raw = row.get("target_discrete_audio")
            target_discrete_audio = (
                None
                if target_discrete_audio_raw is None
                else np.asarray(target_discrete_audio_raw, dtype=np.float32)
            )
            prior_audio_raw = row.get("prior_audio")
            prior_audio = (
                None
                if prior_audio_raw is None
                else np.asarray(prior_audio_raw, dtype=np.float32)
            )
            table.add_data(
                step,
                sample_source,
                sample_idx,
                method,
                wandb.Audio(
                    target_audio,
                    sample_rate=self.audio_sample_rate,
                    caption=(
                        f"{sample_source}_groundtruth_step_{step}_sample_{sample_idx}"
                    ),
                ),
                (
                    None
                    if target_discrete_audio is None
                    else wandb.Audio(
                        target_discrete_audio,
                        sample_rate=self.audio_sample_rate,
                        caption=(
                            f"{sample_source}_groundtruth_discrete_step_{step}_sample_{sample_idx}"
                        ),
                    )
                ),
                (
                    None
                    if prior_audio is None
                    else wandb.Audio(
                        prior_audio,
                        sample_rate=self.audio_sample_rate,
                        caption=(
                            f"{sample_source}_{method}_prior_step_{step}_sample_{sample_idx}"
                        ),
                    )
                ),
                wandb.Audio(
                    pred_audio,
                    sample_rate=self.audio_sample_rate,
                    caption=(
                        f"{sample_source}_{method}_pred_step_{step}_sample_{sample_idx}"
                    ),
                ),
            )
        return table

    def _build_mel_table(
        self,
        periodic_samples: Sequence[PeriodicEvalSample],
    ) -> Optional[Any]:
        media_rows = self._build_eval_media_rows(periodic_samples)
        if not media_rows:
            return None

        if self.audio_decoder is None:
            return None

        try:
            import wandb
        except ModuleNotFoundError:
            return None

        step = int(self.global_step)
        table = wandb.Table(
            columns=[
                "step",
                "sample_source",
                "sample_id",
                "generation_method",
                "groundtruth_mel_spectrogram",
                "prior_mel_spectrogram",
                "predicted_mel_spectrogram",
            ]
        )

        added = 0
        for row in media_rows:
            sample_source = str(row["sample_source"])
            sample_idx = int(row["sample_id"])
            method = str(row["generation_method"])
            target_audio = np.asarray(row["target_audio"], dtype=np.float32)
            pred_audio = np.asarray(row["predicted_audio"], dtype=np.float32)
            prior_audio_raw = row.get("prior_audio")
            prior_audio = (
                None
                if prior_audio_raw is None
                else np.asarray(prior_audio_raw, dtype=np.float32)
            )
            target_mel = self._build_mel_spectrogram_image(
                target_audio,
                title=(
                    f"Groundtruth Mel ({sample_source}, sample {sample_idx}, step {step})"
                ),
            )
            prior_mel = None
            if prior_audio is not None:
                prior_mel = self._build_mel_spectrogram_image(
                    prior_audio,
                    title=(
                        "Prior Mel "
                        f"({sample_source}, {method}, sample {sample_idx}, step {step})"
                    ),
                )
            pred_mel = self._build_mel_spectrogram_image(
                pred_audio,
                title=(
                    "Predicted Mel "
                    f"({sample_source}, {method}, sample {sample_idx}, step {step})"
                ),
            )
            if target_mel is None or pred_mel is None:
                continue

            table.add_data(
                step,
                sample_source,
                sample_idx,
                method,
                target_mel,
                prior_mel,
                pred_mel,
            )
            added += 1

        return table if added > 0 else None

    def _build_text_table(
        self,
        periodic_samples: Sequence[PeriodicEvalSample],
    ) -> Optional[Any]:
        try:
            import wandb
        except ModuleNotFoundError:
            return None

        step = int(self.global_step)
        table = wandb.Table(
            columns=[
                "step",
                "sample_source",
                "sample_id",
                "generation_method",
                "text_prompt",
                "text_target",
                "synthesized_text",
            ]
        )

        added = 0
        for sample_idx, sample in enumerate(periodic_samples):
            for method_name, synthesized_text in sample.synthesized_text_by_method.items():
                table.add_data(
                    step,
                    str(sample.sample_source),
                    sample_idx,
                    str(method_name),
                    sample.text_prompt,
                    sample.text_target,
                    str(synthesized_text),
                )
                added += 1

        return table if added > 0 else None

    def _build_eval_media_rows(
        self,
        periodic_samples: Sequence[PeriodicEvalSample],
    ) -> list[dict[str, Any]]:
        if self.audio_decoder is None:
            return []
        rows: list[dict[str, Any]] = []
        for sample_idx, sample in enumerate(periodic_samples):
            batch_inputs = sample.batch_inputs
            generations = sample.generations
            if self.model.discrete_only:
                if batch_inputs.discrete_target is None:
                    continue

                target_discrete = normalize_discrete_tokens(
                    batch_inputs.discrete_target,
                    "discrete_target",
                    num_discrete_tokens=self.model.num_discrete_tokens,
                )
                normalized_generations: list[tuple[str, torch.LongTensor]] = []
                for method_name, generation in generations.items():
                    if generation.discrete_ids is None:
                        continue
                    normalized_generations.append(
                        (
                            str(method_name),
                            normalize_discrete_tokens(
                                generation.discrete_ids,
                                f"generation[{method_name}].discrete_ids",
                                num_discrete_tokens=self.model.num_discrete_tokens,
                            ),
                        )
                    )
                if not normalized_generations:
                    continue

                sample_target_len = int(target_discrete.shape[1])
                if batch_inputs.speech_target_lengths is not None:
                    target_lengths = torch.as_tensor(
                        batch_inputs.speech_target_lengths,
                        device=self.device,
                    )
                    if target_lengths.dim() == 0:
                        target_lengths = target_lengths.unsqueeze(0)
                    if int(target_lengths.shape[0]) > 0:
                        sample_target_len = min(sample_target_len, int(target_lengths[0].item()))
                if sample_target_len < 1:
                    continue

                for method, pred_discrete in normalized_generations:
                    if int(pred_discrete.shape[0]) < 1:
                        continue
                    sample_eval_len = min(sample_target_len, int(pred_discrete.shape[1]))
                    if sample_eval_len < 1:
                        continue

                    sample_target_discrete = target_discrete[0, :sample_eval_len, :]
                    sample_pred_discrete = pred_discrete[0, :sample_eval_len, :]
                    target_audio = LU.decode_audio(
                        sample_target_discrete,
                        audio_decoder=self.audio_decoder,
                    )
                    pred_audio = LU.decode_audio(
                        sample_pred_discrete,
                        audio_decoder=self.audio_decoder,
                    )
                    if target_audio is None or pred_audio is None:
                        continue

                    rows.append(
                        {
                            "sample_source": str(sample.sample_source),
                            "sample_id": sample_idx,
                            "generation_method": method,
                            "target_audio": target_audio,
                            "target_discrete_audio": target_audio,
                            "prior_audio": None,
                            "predicted_audio": pred_audio,
                        }
                    )
                continue

            if batch_inputs.continuous_target is None:
                continue
            if batch_inputs.discrete_target is None:
                continue

            target_continuous = normalize_continuous_latents(
                batch_inputs.continuous_target,
                expected_len=batch_inputs.continuous_target.shape[1],
                name="continuous_target",
                continuous_latent_size=self.model.continuous_latent_size,
            )
            target_discrete = normalize_discrete_tokens(
                batch_inputs.discrete_target,
                "discrete_target",
                num_discrete_tokens=self.model.num_discrete_tokens,
            )

            normalized_generations: list[tuple[str, torch.LongTensor]] = []
            for method_name, generation in generations.items():
                if generation.discrete_ids is None:
                    continue
                normalized_generations.append(
                    (
                        str(method_name),
                        normalize_discrete_tokens(
                            generation.discrete_ids,
                            f"generation[{method_name}].discrete_ids",
                            num_discrete_tokens=self.model.num_discrete_tokens,
                        ),
                    )
                )
            if not normalized_generations:
                continue

            sample_target_len = min(int(target_continuous.shape[1]), int(target_discrete.shape[1]))
            if batch_inputs.speech_target_lengths is not None:
                target_lengths = torch.as_tensor(
                    batch_inputs.speech_target_lengths,
                    device=self.device,
                )
                if target_lengths.dim() == 0:
                    target_lengths = target_lengths.unsqueeze(0)
                if int(target_lengths.shape[0]) > 0:
                    sample_target_len = min(sample_target_len, int(target_lengths[0].item()))
            if sample_target_len < 1:
                continue

            for method, pred_discrete in normalized_generations:
                if int(pred_discrete.shape[0]) < 1:
                    continue
                sample_eval_len = min(sample_target_len, int(pred_discrete.shape[1]))
                if sample_eval_len < 1:
                    continue

                sample_target_continuous = target_continuous[0, :sample_eval_len, :]
                sample_target_discrete = target_discrete[0, :sample_eval_len, :]
                sample_pred_discrete = pred_discrete[0, :sample_eval_len, :]
                target_audio = LU.decode_audio(
                    sample_target_continuous,
                    audio_decoder=self.audio_decoder,
                )
                target_discrete_audio = LU.decode_audio(
                    sample_target_discrete,
                    audio_decoder=self.audio_decoder,
                )
                pred_audio = LU.decode_audio(
                    sample_pred_discrete,
                    audio_decoder=self.audio_decoder,
                )
                if target_audio is None or pred_audio is None:
                    continue

                rows.append(
                    {
                        "sample_source": str(sample.sample_source),
                        "sample_id": sample_idx,
                        "generation_method": method,
                        "target_audio": target_audio,
                        "target_discrete_audio": target_discrete_audio,
                        "prior_audio": None,
                        "predicted_audio": pred_audio,
                    }
                )

        return rows

    def _build_mel_spectrogram_image(self, audio: np.ndarray, *, title: str) -> Optional[Any]:
        try:
            import matplotlib.pyplot as plt
            import wandb
        except ModuleNotFoundError:
            return None

        log_mel = LU.compute_log_mel_spectrogram(
            audio,
            sample_rate=self.audio_sample_rate,
        )
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

    def _decode_text_tokens(
        self,
        text_tokens: Optional[torch.Tensor],
        *,
        length: Optional[int] = None,
    ) -> str:
        if text_tokens is None:
            return ""
        tokens = torch.as_tensor(text_tokens, dtype=torch.long).detach().cpu().reshape(-1)
        if length is not None:
            tokens = tokens[: max(0, int(length))]
        if tokens.numel() == 0:
            return ""

        token_to_char = self._text_id_to_char()
        pad_token_id = int(getattr(self.model, "pad_token_id", -1))
        eos_token_id = int(getattr(self.model, "eos_token_id", -1))
        eot_token_id = int(getattr(self.model, "eot_token_id", -1))

        decoded_chars: list[str] = []
        fallback_token_ids: list[str] = []
        for token_id in (int(item) for item in tokens.tolist()):
            if token_id in (pad_token_id, eos_token_id, eot_token_id):
                continue
            char = token_to_char.get(token_id)
            if char is not None:
                decoded_chars.append(char)
            else:
                fallback_token_ids.append(str(token_id))

        if decoded_chars:
            return "".join(decoded_chars)
        if fallback_token_ids:
            return " ".join(fallback_token_ids)
        return ""

    def _text_id_to_char(self) -> dict[int, str]:
        if self._cached_text_id_to_char is not None:
            return self._cached_text_id_to_char

        tokenizer = self._resolve_text_tokenizer()
        if tokenizer is None:
            self._cached_text_id_to_char = {}
            return self._cached_text_id_to_char

        char_to_id = getattr(tokenizer, "char_to_id", None)
        text_token_offset = getattr(tokenizer, "text_token_offset", None)
        if not isinstance(char_to_id, Mapping) or text_token_offset is None:
            self._cached_text_id_to_char = {}
            return self._cached_text_id_to_char

        token_to_char: dict[int, str] = {}
        for char, local_id in char_to_id.items():
            try:
                token_to_char[int(text_token_offset) + int(local_id)] = str(char)
            except Exception:
                continue
        self._cached_text_id_to_char = token_to_char
        return token_to_char

    def _resolve_text_tokenizer(self) -> Optional[Any]:
        for loader in (self._primary_train_loader(), self._primary_val_loader()):
            tokenizer = LU.extract_tokenizer_from_loader(loader)
            if tokenizer is not None:
                return tokenizer
        return None

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
