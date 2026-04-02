from __future__ import annotations

import argparse
import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import LlamaConfig

from dataset.dataset import BatchCollate, PrismDataset, build_shared_token_layout
from models.prism_tts import PrismTTS
from models.prism_tts_lightning import PrismTTSLightning

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

    try:
        from lightning.pytorch.loggers import WandbLogger
    except ModuleNotFoundError:
        WandbLogger = None
except ModuleNotFoundError:
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
        from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

        try:
            from pytorch_lightning.loggers import WandbLogger
        except ModuleNotFoundError:
            WandbLogger = None
    except ModuleNotFoundError as exc:
        raise ImportError(
            "This training script requires `lightning` or `pytorch_lightning`."
        ) from exc

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ImportError(
        "This training script requires `PyYAML` (`pip install pyyaml`)."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Prism-TTS using PyTorch Lightning and YAML configs."
    )
    parser.add_argument(
        "--trainer-config",
        type=Path,
        default=Path("config/trainer.yaml"),
        help="Path to trainer YAML config.",
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
        "--experiment-config",
        type=Path,
        default=Path("config/experiment.yaml"),
        help="Optional experiment override config. Empty file is allowed.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Checkpoint path to resume from for fit/validate/test.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation only (no training).",
    )
    parser.add_argument(
        "--test-after-fit",
        action="store_true",
        help="Run test loop after fit if test manifest is configured.",
    )
    return parser.parse_args()


def _read_yaml(path: Path, *, required: bool) -> dict[str, Any]:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()
    if not resolved.exists():
        if required:
            raise FileNotFoundError(f"Config file not found: {resolved}")
        return {}
    if not resolved.is_file():
        raise ValueError(f"Config path is not a file: {resolved}")

    text = resolved.read_text(encoding="utf-8")
    if text.strip() == "":
        return {}

    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {resolved}, got {type(data).__name__}.")
    return data


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _maybe_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip() != "":
        return value
    return None


def _extract_config_path_overrides(experiment_cfg: dict[str, Any]) -> dict[str, str]:
    overrides: dict[str, str] = {}

    candidates = [experiment_cfg]
    nested = experiment_cfg.get("experiment")
    if isinstance(nested, dict):
        candidates.append(nested)

    for cfg in candidates:
        trainer_path = _maybe_str(cfg.get("trainer_config"))
        model_path = _maybe_str(cfg.get("model_config"))
        data_path = _maybe_str(cfg.get("data_config"))
        if trainer_path is not None:
            overrides["trainer_config"] = trainer_path
        if model_path is not None:
            overrides["model_config"] = model_path
        if data_path is not None:
            overrides["data_config"] = data_path

        for section_name in ("configs", "config_paths"):
            section = cfg.get(section_name)
            if not isinstance(section, dict):
                continue
            trainer_path = _maybe_str(section.get("trainer"))
            model_path = _maybe_str(section.get("model"))
            data_path = _maybe_str(section.get("data"))
            if trainer_path is not None:
                overrides["trainer_config"] = trainer_path
            if model_path is not None:
                overrides["model_config"] = model_path
            if data_path is not None:
                overrides["data_config"] = data_path

    return overrides


@dataclass
class ResolvedConfigs:
    merged: dict[str, Any]
    trainer_config_path: Path
    model_config_path: Path
    data_config_path: Path
    experiment_config_path: Path


def _resolve_config_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    experiment_path = args.experiment_config.expanduser()
    if not experiment_path.is_absolute():
        experiment_path = (Path.cwd() / experiment_path).resolve()
    else:
        experiment_path = experiment_path.resolve()

    experiment_cfg = _read_yaml(experiment_path, required=False)
    path_overrides = _extract_config_path_overrides(experiment_cfg)

    trainer_path = Path(path_overrides.get("trainer_config", str(args.trainer_config))).expanduser()
    model_path = Path(path_overrides.get("model_config", str(args.model_config))).expanduser()
    data_path = Path(path_overrides.get("data_config", str(args.data_config))).expanduser()

    if not trainer_path.is_absolute():
        trainer_path = (Path.cwd() / trainer_path).resolve()
    else:
        trainer_path = trainer_path.resolve()

    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()
    else:
        model_path = model_path.resolve()

    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()
    else:
        data_path = data_path.resolve()

    return trainer_path, model_path, data_path, experiment_path


def _load_merged_configs(args: argparse.Namespace) -> ResolvedConfigs:
    trainer_path, model_path, data_path, experiment_path = _resolve_config_paths(args)

    trainer_cfg = _read_yaml(trainer_path, required=True)
    model_cfg = _read_yaml(model_path, required=True)
    data_cfg = _read_yaml(data_path, required=True)
    experiment_cfg = _read_yaml(experiment_path, required=False)

    merged: dict[str, Any] = {}
    _deep_update(merged, trainer_cfg)
    _deep_update(merged, model_cfg)
    _deep_update(merged, data_cfg)
    _deep_update(merged, experiment_cfg)

    nested_experiment = experiment_cfg.get("experiment")
    if isinstance(nested_experiment, dict):
        _deep_update(merged, nested_experiment)

    return ResolvedConfigs(
        merged=merged,
        trainer_config_path=trainer_path,
        model_config_path=model_path,
        data_config_path=data_path,
        experiment_config_path=experiment_path,
    )


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required mapping '{key}' in merged config.")
    return value


def _coerce_betas(value: Any) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("trainer.lightning_module.betas must be a list/tuple of length 2.")
    return float(value[0]), float(value[1])


def _validate_config_consistency(config: dict[str, Any]) -> None:
    data_cfg = _require_mapping(config, "data")
    model_cfg = _require_mapping(config, "model")

    shared_layout = _require_mapping(data_cfg, "shared_layout")
    discrete_token_count = int(shared_layout["discrete_token_count"])
    delay_id, eos_id, pad_id, text_offset = build_shared_token_layout(discrete_token_count)

    expected = {
        "delay_token_id": delay_id,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "text_token_offset": text_offset,
    }
    for key, expected_value in expected.items():
        if key in shared_layout and int(shared_layout[key]) != expected_value:
            raise ValueError(
                f"data.shared_layout.{key} mismatch: expected {expected_value}, "
                f"got {shared_layout[key]}"
            )

    prism_cfg = _require_mapping(model_cfg, "prism_tts")
    llama_cfg = _require_mapping(model_cfg, "llama_config")
    dataset_cfg = _require_mapping(data_cfg, "dataset")

    num_discrete_tokens = int(prism_cfg["num_discrete_tokens"])
    dataset_stream_count_raw = dataset_cfg.get("discrete_stream_count")
    if dataset_stream_count_raw is not None:
        dataset_stream_count = int(dataset_stream_count_raw)
        if num_discrete_tokens != dataset_stream_count:
            raise ValueError(
                "model.prism_tts.num_discrete_tokens must match "
                f"data.dataset.discrete_stream_count ({dataset_stream_count})."
            )

    continuous_latent_size = int(prism_cfg["continuous_latent_size"])
    dataset_continuous_dim_raw = dataset_cfg.get("continuous_feature_dim")
    if dataset_continuous_dim_raw is not None:
        dataset_continuous_dim = int(dataset_continuous_dim_raw)
        if continuous_latent_size != dataset_continuous_dim:
            raise ValueError(
                "model.prism_tts.continuous_latent_size must match "
                f"data.dataset.continuous_feature_dim ({dataset_continuous_dim})."
            )

    discrete_vocab_size = int(prism_cfg["discrete_vocab_size"])
    if discrete_vocab_size < text_offset:
        raise ValueError(
            "model.prism_tts.discrete_vocab_size is too small for shared token layout. "
            f"Need at least {text_offset}, got {discrete_vocab_size}."
        )

    if "pad_token_id" in llama_cfg and int(llama_cfg["pad_token_id"]) != pad_id:
        raise ValueError(
            f"model.llama_config.pad_token_id must equal data shared pad token id ({pad_id})."
        )
    if "eos_token_id" in llama_cfg and int(llama_cfg["eos_token_id"]) != eos_id:
        raise ValueError(
            f"model.llama_config.eos_token_id must equal data shared eos token id ({eos_id})."
        )


def _build_model(config: dict[str, Any]) -> PrismTTS:
    model_cfg = _require_mapping(config, "model")
    model_name = model_cfg.get("name", "prism_tts")
    if model_name != "prism_tts":
        raise ValueError(f"Unsupported model.name={model_name!r}. Only 'prism_tts' is supported.")

    prism_cfg = _require_mapping(model_cfg, "prism_tts")
    llama_cfg = _require_mapping(model_cfg, "llama_config")
    llama_config = LlamaConfig(**llama_cfg)

    return PrismTTS(
        llama_config=llama_config,
        num_discrete_tokens=int(prism_cfg["num_discrete_tokens"]),
        discrete_vocab_size=int(prism_cfg["discrete_vocab_size"]),
        continuous_latent_size=int(prism_cfg["continuous_latent_size"]),
        flow_num_res_blocks=int(prism_cfg.get("flow_num_res_blocks", 4)),
        flow_model_channels=prism_cfg.get("flow_model_channels"),
        flow_loss_weight=float(prism_cfg.get("flow_loss_weight", 1.0)),
        flow_sample_steps=int(prism_cfg.get("flow_sample_steps", 16)),
    )


def _build_scheduler_factory(
    scheduler_cfg: dict[str, Any],
    *,
    max_steps: int,
):
    enabled = bool(scheduler_cfg.get("enabled", True))
    if not enabled:
        return None

    scheduler_name = str(scheduler_cfg.get("name", "cosine_with_warmup")).lower()
    if scheduler_name != "cosine_with_warmup":
        raise ValueError(
            f"Unsupported trainer.scheduler.name={scheduler_name!r}. "
            "Only 'cosine_with_warmup' is supported."
        )
    if max_steps < 1:
        raise ValueError(
            "trainer.lightning_trainer.max_steps must be >= 1 when scheduler is enabled."
        )

    warmup_steps = max(0, int(scheduler_cfg.get("warmup_steps", 0)))
    min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.0))
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

    def scheduler_factory(optimizer: torch.optim.Optimizer):
        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            if max_steps <= warmup_steps:
                return 1.0

            progress = float(step - warmup_steps) / float(max_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return {
            "scheduler": LambdaLR(optimizer, lr_lambda=lr_lambda),
            "interval": "step",
            "frequency": 1,
        }

    return scheduler_factory


def _build_lightning_module(
    config: dict[str, Any],
    model: PrismTTS,
) -> PrismTTSLightning:
    trainer_cfg = _require_mapping(config, "trainer")
    module_cfg = _require_mapping(trainer_cfg, "lightning_module")
    optimizer_cfg = _require_mapping(trainer_cfg, "optimizer")
    scheduler_cfg = _require_mapping(trainer_cfg, "scheduler")
    lightning_trainer_cfg = _require_mapping(trainer_cfg, "lightning_trainer")

    optimizer_name = str(optimizer_cfg.get("name", "adamw")).lower()
    if optimizer_name != "adamw":
        raise ValueError(
            f"Unsupported trainer.optimizer.name={optimizer_name!r}. Only 'adamw' is supported."
        )

    max_steps = int(lightning_trainer_cfg.get("max_steps", 0))
    scheduler_factory = _build_scheduler_factory(
        scheduler_cfg=scheduler_cfg,
        max_steps=max_steps,
    )

    return PrismTTSLightning(
        model=model,
        learning_rate=float(module_cfg.get("learning_rate", 3.0e-4)),
        weight_decay=float(module_cfg.get("weight_decay", 0.01)),
        betas=_coerce_betas(module_cfg.get("betas", [0.9, 0.95])),
        eval_every_n_steps=int(module_cfg.get("eval_every_n_steps", 5000)),
        scheduler_factory=scheduler_factory,
        audio_sample_rate=int(module_cfg.get("audio_sample_rate", 24_000)),
        max_audio_samples=int(module_cfg.get("max_audio_samples", 2)),
        ema_decay=float(module_cfg.get("ema_decay", 0.999)),
        ema_start_step=int(module_cfg.get("ema_start_step", 0)),
        ema_update_every_n_steps=int(module_cfg.get("ema_update_every_n_steps", 1)),
        ema_warmup_steps=int(module_cfg.get("ema_warmup_steps", 0)),
        use_ema_for_validation=bool(module_cfg.get("use_ema_for_validation", True)),
        use_ema_for_periodic_eval=bool(module_cfg.get("use_ema_for_periodic_eval", True)),
        sync_dist_logging=bool(module_cfg.get("sync_dist_logging", False)),
    )


def _optional_path(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if value == "":
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())


def _build_data_objects(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    data_cfg = _require_mapping(config, "data")
    loader_cfg = _require_mapping(data_cfg, "loader")
    dataset_cfg = _require_mapping(data_cfg, "dataset")
    collate_cfg = _require_mapping(data_cfg, "collate")
    shared_layout = _require_mapping(data_cfg, "shared_layout")

    discrete_token_count = int(shared_layout["discrete_token_count"])

    dataset_kwargs: dict[str, Any] = {
        "prompt_length": dataset_cfg.get("prompt_length"),
        "min_prompt_length": int(dataset_cfg.get("min_prompt_length", 1)),
        "min_target_length": int(dataset_cfg.get("min_target_length", 1)),
        "vocab_path": _optional_path(data_cfg.get("vocab_path")),
        "manifest_root": _optional_path(data_cfg.get("manifest_root")),
        "discrete_token_count": discrete_token_count,
        "discrete_stream_count": dataset_cfg.get("discrete_stream_count"),
        "continuous_feature_dim": dataset_cfg.get("continuous_feature_dim"),
        "append_eos_to_text": bool(dataset_cfg.get("append_eos_to_text", False)),
        "cache_npy": bool(dataset_cfg.get("cache_npy", False)),
    }

    train_manifest = _optional_path(data_cfg.get("train_manifest"))
    if train_manifest is None:
        raise ValueError("data.train_manifest must be set for training.")
    train_dataset = PrismDataset(source=train_manifest, **dataset_kwargs)

    val_manifest = _optional_path(data_cfg.get("val_manifest"))
    val_dataset = PrismDataset(source=val_manifest, **dataset_kwargs) if val_manifest else None

    test_manifest = _optional_path(data_cfg.get("test_manifest"))
    test_dataset = PrismDataset(source=test_manifest, **dataset_kwargs) if test_manifest else None

    collate = BatchCollate(
        text_pad_value=collate_cfg.get("text_pad_value"),
        discrete_pad_value=collate_cfg.get("discrete_pad_value"),
        continuous_pad_value=float(collate_cfg.get("continuous_pad_value", 0.0)),
        include_attention_mask=bool(collate_cfg.get("include_attention_mask", True)),
        discrete_token_count=discrete_token_count,
        prompt_length=collate_cfg.get("prompt_length"),
        min_prompt_length=int(collate_cfg.get("min_prompt_length", 1)),
        min_target_length=int(collate_cfg.get("min_target_length", 1)),
        stream_delay=collate_cfg.get("stream_delay"),
        discrete_stream_delay_ms=collate_cfg.get("discrete_stream_delay_ms"),
        codec_frame_rate_hz=float(collate_cfg.get("codec_frame_rate_hz", 12.5)),
    )

    num_workers = int(loader_cfg.get("num_workers", 0))
    persistent_workers = bool(loader_cfg.get("persistent_workers", False)) and num_workers > 0
    pin_memory = bool(loader_cfg.get("pin_memory", False))

    common_loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "collate_fn": collate,
    }
    prefetch_factor = loader_cfg.get("prefetch_factor")
    if num_workers > 0 and prefetch_factor is not None:
        common_loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(loader_cfg.get("train_batch_size", 8)),
        shuffle=bool(loader_cfg.get("shuffle_train", True)),
        drop_last=bool(loader_cfg.get("drop_last_train", True)),
        **common_loader_kwargs,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(loader_cfg.get("val_batch_size", 8)),
            shuffle=bool(loader_cfg.get("shuffle_val", False)),
            drop_last=False,
            **common_loader_kwargs,
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(loader_cfg.get("test_batch_size", 8)),
            shuffle=bool(loader_cfg.get("shuffle_test", False)),
            drop_last=False,
            **common_loader_kwargs,
        )

    return train_loader, val_loader, test_loader


def _filter_kwargs_for_callable(
    fn: Any,
    kwargs: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    signature = inspect.signature(fn)
    params = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs

    allowed = set(params.keys())
    filtered: dict[str, Any] = {}
    ignored: list[str] = []
    for key, value in kwargs.items():
        if key in allowed:
            filtered[key] = value
        else:
            ignored.append(key)
    if ignored:
        print(f"[train.py] Ignoring unsupported {context} kwargs: {sorted(ignored)}")
    return filtered


def _build_logger(logger_cfg: dict[str, Any]) -> Any:
    logger_type = str(logger_cfg.get("type", "none")).lower()
    if logger_type in {"", "none", "false", "null", "disabled"}:
        return False

    save_dir = logger_cfg.get("save_dir", "logs")
    if logger_type == "wandb":
        if WandbLogger is None:
            print("[train.py] WandB logger requested but unavailable. Falling back to CSV logger.")
            return CSVLogger(save_dir=save_dir, name="prism_tts")
        kwargs = {
            "project": logger_cfg.get("project"),
            "name": logger_cfg.get("name"),
            "save_dir": save_dir,
            "offline": bool(logger_cfg.get("offline", False)),
            "log_model": logger_cfg.get("log_model", False),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs = _filter_kwargs_for_callable(
            WandbLogger.__init__,
            kwargs,
            context="WandbLogger",
        )
        return WandbLogger(**kwargs)

    if logger_type == "tensorboard":
        kwargs = {
            "save_dir": save_dir,
            "name": logger_cfg.get("name", "prism_tts"),
        }
        kwargs = _filter_kwargs_for_callable(
            TensorBoardLogger.__init__,
            kwargs,
            context="TensorBoardLogger",
        )
        return TensorBoardLogger(**kwargs)

    if logger_type == "csv":
        kwargs = {
            "save_dir": save_dir,
            "name": logger_cfg.get("name", "prism_tts"),
        }
        kwargs = _filter_kwargs_for_callable(
            CSVLogger.__init__,
            kwargs,
            context="CSVLogger",
        )
        return CSVLogger(**kwargs)

    raise ValueError(
        f"Unsupported trainer.logger.type={logger_type!r}. "
        "Supported values: wandb, tensorboard, csv, none."
    )


def _build_callbacks(config: dict[str, Any]) -> list[Any]:
    trainer_cfg = _require_mapping(config, "trainer")
    lightning_trainer_cfg = _require_mapping(trainer_cfg, "lightning_trainer")
    checkpoint_cfg = _require_mapping(trainer_cfg, "checkpoint")
    scheduler_cfg = _require_mapping(trainer_cfg, "scheduler")

    callbacks: list[Any] = []

    enable_checkpointing = bool(lightning_trainer_cfg.get("enable_checkpointing", True))
    if enable_checkpointing:
        checkpoint_kwargs = dict(checkpoint_cfg)
        dirpath = checkpoint_kwargs.get("dirpath")
        if dirpath is not None:
            ckpt_dir = Path(str(dirpath)).expanduser()
            if not ckpt_dir.is_absolute():
                ckpt_dir = Path.cwd() / ckpt_dir
            ckpt_dir = ckpt_dir.resolve()
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_kwargs["dirpath"] = str(ckpt_dir)

        checkpoint_kwargs = _filter_kwargs_for_callable(
            ModelCheckpoint.__init__,
            checkpoint_kwargs,
            context="ModelCheckpoint",
        )
        callbacks.append(ModelCheckpoint(**checkpoint_kwargs))

    if bool(scheduler_cfg.get("enabled", True)):
        lr_monitor_kwargs = _filter_kwargs_for_callable(
            LearningRateMonitor.__init__,
            {"logging_interval": "step"},
            context="LearningRateMonitor",
        )
        callbacks.append(LearningRateMonitor(**lr_monitor_kwargs))

    return callbacks


def _trainer_supports_ckpt_path() -> bool:
    return "ckpt_path" in inspect.signature(pl.Trainer.fit).parameters


def _coerce_gpu_indices(value: Any) -> list[int]:
    indices: list[int]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip() != ""]
        if not parts:
            raise ValueError(
                "trainer.distributed.gpu_indices string must contain comma-separated GPU indices."
            )
        try:
            indices = [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError(
                "trainer.distributed.gpu_indices string must contain only integer values."
            ) from exc
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("trainer.distributed.gpu_indices cannot be empty.")
        try:
            indices = [int(index) for index in value]
        except ValueError as exc:
            raise ValueError(
                "trainer.distributed.gpu_indices list must contain only integer values."
            ) from exc
    else:
        raise ValueError(
            "trainer.distributed.gpu_indices must be a list/tuple of ints "
            "or a comma-separated string."
        )

    if any(index < 0 for index in indices):
        raise ValueError("trainer.distributed.gpu_indices must contain non-negative indices.")
    if len(set(indices)) != len(indices):
        raise ValueError("trainer.distributed.gpu_indices must not contain duplicates.")
    return indices


def _apply_distributed_training_config(config: dict[str, Any]) -> None:
    trainer_cfg = _require_mapping(config, "trainer")
    distributed_cfg = trainer_cfg.get("distributed")
    if not isinstance(distributed_cfg, dict) or not bool(distributed_cfg.get("enabled", False)):
        return

    lightning_trainer_cfg = _require_mapping(trainer_cfg, "lightning_trainer")
    lightning_trainer_cfg["accelerator"] = distributed_cfg.get("accelerator", "gpu")
    gpu_indices = distributed_cfg.get("gpu_indices")
    if gpu_indices is not None:
        lightning_trainer_cfg["devices"] = _coerce_gpu_indices(gpu_indices)
    else:
        lightning_trainer_cfg["devices"] = distributed_cfg.get("devices", "auto")
    lightning_trainer_cfg["strategy"] = distributed_cfg.get(
        "strategy",
        "ddp_find_unused_parameters_false",
    )

    num_nodes = distributed_cfg.get("num_nodes")
    if num_nodes is not None:
        lightning_trainer_cfg["num_nodes"] = int(num_nodes)

    module_cfg = _require_mapping(trainer_cfg, "lightning_module")
    if "sync_dist_logging" in distributed_cfg:
        module_cfg["sync_dist_logging"] = bool(distributed_cfg["sync_dist_logging"])


def _build_trainer(config: dict[str, Any], *, logger: Any, callbacks: list[Any]) -> Any:
    trainer_cfg = _require_mapping(config, "trainer")
    lightning_trainer_cfg = dict(_require_mapping(trainer_cfg, "lightning_trainer"))
    lightning_trainer_cfg["callbacks"] = callbacks
    lightning_trainer_cfg["logger"] = logger

    trainer_kwargs = _filter_kwargs_for_callable(
        pl.Trainer.__init__,
        lightning_trainer_cfg,
        context="Trainer",
    )
    return pl.Trainer(**trainer_kwargs)


def run(args: argparse.Namespace) -> None:
    resolved = _load_merged_configs(args)
    config = resolved.merged

    _validate_config_consistency(config)
    _apply_distributed_training_config(config)

    trainer_cfg = _require_mapping(config, "trainer")
    seed = trainer_cfg.get("seed")
    if seed is not None:
        pl.seed_everything(int(seed), workers=bool(trainer_cfg.get("seed_workers", True)))

    train_loader, val_loader, test_loader = _build_data_objects(config)
    model = _build_model(config)
    lightning_module = _build_lightning_module(config, model=model)

    logger = _build_logger(_require_mapping(trainer_cfg, "logger"))
    callbacks = _build_callbacks(config)

    trainer = _build_trainer(config, logger=logger, callbacks=callbacks)

    supports_ckpt_path = _trainer_supports_ckpt_path()
    if args.ckpt_path is not None and not supports_ckpt_path:
        raise RuntimeError(
            "This installed Lightning version does not support Trainer.fit(..., ckpt_path=...). "
            "Please upgrade Lightning or remove --ckpt-path."
        )

    if args.validate_only:
        if val_loader is None:
            raise ValueError("Validation requested but data.val_manifest is not configured.")
        validate_kwargs: dict[str, Any] = {
            "model": lightning_module,
            "dataloaders": val_loader,
        }
        if args.ckpt_path is not None and "ckpt_path" in inspect.signature(trainer.validate).parameters:
            validate_kwargs["ckpt_path"] = args.ckpt_path
        trainer.validate(**validate_kwargs)
        return

    fit_kwargs: dict[str, Any] = {
        "model": lightning_module,
        "train_dataloaders": train_loader,
    }
    if val_loader is not None:
        fit_kwargs["val_dataloaders"] = val_loader
    if args.ckpt_path is not None and supports_ckpt_path:
        fit_kwargs["ckpt_path"] = args.ckpt_path
    trainer.fit(**fit_kwargs)

    if args.test_after_fit and test_loader is not None:
        test_kwargs: dict[str, Any] = {
            "model": lightning_module,
            "dataloaders": test_loader,
        }
        if "ckpt_path" in inspect.signature(trainer.test).parameters:
            if args.ckpt_path is not None:
                test_kwargs["ckpt_path"] = args.ckpt_path
            else:
                test_kwargs["ckpt_path"] = "best"
        trainer.test(**test_kwargs)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
