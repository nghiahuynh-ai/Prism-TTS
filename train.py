from __future__ import annotations

import argparse
import importlib
import inspect
import math
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULTED_CUDA_ALLOC_CONF = False
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    _DEFAULTED_CUDA_ALLOC_CONF = True

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import LlamaConfig

from dataset.adaptive_batching import AdaptiveMemoryBatchSampler, estimate_prism_sample_lengths
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
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Override trainer.logger.project for WandB runs.",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Override trainer.logger.name for WandB runs.",
    )
    parser.add_argument(
        "--wandb-save-dir",
        type=str,
        default=None,
        help="Override trainer.logger.save_dir for WandB runs.",
    )
    parser.add_argument(
        "--wandb-offline",
        type=str,
        default=None,
        help="Override trainer.logger.offline (true/false).",
    )
    parser.add_argument(
        "--wandb-log-model",
        type=str,
        default=None,
        help="Override trainer.logger.log_model (true/false/all).",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Override trainer.logger.entity for WandB runs.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Override trainer.logger.group for WandB runs.",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help="Override trainer.logger.tags as comma-separated values.",
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


def _parse_bool_string(value: str, *, field_name: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{field_name} must be a boolean string (true/false), got {value!r}.")


def _coerce_wandb_log_model(value: str) -> bool | str:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return value


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


def _apply_wandb_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    trainer_cfg = _require_mapping(config, "trainer")
    logger_cfg = _require_mapping(trainer_cfg, "logger")

    if args.wandb_project is not None:
        logger_cfg["project"] = args.wandb_project
    if args.wandb_name is not None:
        logger_cfg["name"] = args.wandb_name
    if args.wandb_save_dir is not None:
        logger_cfg["save_dir"] = args.wandb_save_dir
    if args.wandb_offline is not None:
        logger_cfg["offline"] = _parse_bool_string(
            args.wandb_offline,
            field_name="--wandb-offline",
        )
    if args.wandb_log_model is not None:
        logger_cfg["log_model"] = _coerce_wandb_log_model(args.wandb_log_model)
    if args.wandb_entity is not None:
        logger_cfg["entity"] = args.wandb_entity
    if args.wandb_group is not None:
        logger_cfg["group"] = args.wandb_group
    if args.wandb_tags is not None:
        logger_cfg["tags"] = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]


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
    eot_id, eos_id, pad_id, text_offset = build_shared_token_layout(discrete_token_count)

    expected = {
        "eot_token_id": eot_id,
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
    collate_cfg = _require_mapping(data_cfg, "collate")

    num_discrete_tokens = int(prism_cfg["num_discrete_tokens"])
    dataset_stream_count_raw = dataset_cfg.get("discrete_stream_count")
    if dataset_stream_count_raw is not None:
        dataset_stream_count = int(dataset_stream_count_raw)
        if num_discrete_tokens != dataset_stream_count:
            raise ValueError(
                "model.prism_tts.num_discrete_tokens must match "
                f"data.dataset.discrete_stream_count ({dataset_stream_count})."
            )

    min_active_stream_count = int(collate_cfg.get("min_active_discrete_stream_count", 1))
    if min_active_stream_count < 1:
        raise ValueError("data.collate.min_active_discrete_stream_count must be >= 1.")
    max_active_stream_count_raw = collate_cfg.get("max_active_discrete_stream_count")
    if max_active_stream_count_raw is not None:
        max_active_stream_count = int(max_active_stream_count_raw)
        if max_active_stream_count < 1:
            raise ValueError("data.collate.max_active_discrete_stream_count must be >= 1.")
        if max_active_stream_count > num_discrete_tokens:
            raise ValueError(
                "data.collate.max_active_discrete_stream_count must be <= "
                f"model.prism_tts.num_discrete_tokens ({num_discrete_tokens})."
            )
        if min_active_stream_count > max_active_stream_count:
            raise ValueError(
                "data.collate.min_active_discrete_stream_count must be <= "
                "data.collate.max_active_discrete_stream_count."
            )
    elif min_active_stream_count > num_discrete_tokens:
        raise ValueError(
            "data.collate.min_active_discrete_stream_count must be <= "
            f"model.prism_tts.num_discrete_tokens ({num_discrete_tokens})."
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
    continuous_loss_weight = float(prism_cfg.get("continuous_loss_weight", 1.0))
    if continuous_loss_weight < 0.0:
        raise ValueError("model.prism_tts.continuous_loss_weight must be >= 0.")
    discrete_regular_token_loss_weight = float(
        prism_cfg.get("discrete_regular_token_loss_weight", 1.0)
    )
    if discrete_regular_token_loss_weight < 0.0:
        raise ValueError("model.prism_tts.discrete_regular_token_loss_weight must be >= 0.")
    discrete_special_token_loss_weight = float(
        prism_cfg.get("discrete_special_token_loss_weight", 1.0)
    )
    if discrete_special_token_loss_weight < 0.0:
        raise ValueError("model.prism_tts.discrete_special_token_loss_weight must be >= 0.")
    if (
        discrete_regular_token_loss_weight == 0.0
        and discrete_special_token_loss_weight == 0.0
    ):
        raise ValueError(
            "At least one of model.prism_tts.discrete_regular_token_loss_weight or "
            "model.prism_tts.discrete_special_token_loss_weight must be > 0."
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
    llama_cfg = dict(_require_mapping(model_cfg, "llama_config"))
    attn_impl = str(llama_cfg.get("_attn_implementation", "eager")).lower()
    force_eager = os.environ.get("PRISM_TTS_FORCE_EAGER_ATTN", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if torch.cuda.is_available() and attn_impl == "eager" and not force_eager:
        llama_cfg["_attn_implementation"] = "sdpa"
        print(
            "[train.py] model.llama_config._attn_implementation='eager' detected on CUDA; "
            "overriding to 'sdpa' to avoid eager-attention allocator instability. "
            "Set PRISM_TTS_FORCE_EAGER_ATTN=true to keep eager."
        )
    llama_config = LlamaConfig(**llama_cfg)

    return PrismTTS(
        llama_config=llama_config,
        num_discrete_tokens=int(prism_cfg["num_discrete_tokens"]),
        discrete_vocab_size=int(prism_cfg["discrete_vocab_size"]),
        continuous_latent_size=int(prism_cfg["continuous_latent_size"]),
        flow_num_res_blocks=int(prism_cfg.get("flow_num_res_blocks", 4)),
        flow_model_channels=prism_cfg.get("flow_model_channels"),
        flow_loss_weight=float(prism_cfg.get("flow_loss_weight", 1.0)),
        continuous_loss_weight=float(prism_cfg.get("continuous_loss_weight", 1.0)),
        discrete_regular_token_loss_weight=float(
            prism_cfg.get("discrete_regular_token_loss_weight", 1.0)
        ),
        discrete_special_token_loss_weight=float(
            prism_cfg.get("discrete_special_token_loss_weight", 1.0)
        ),
        flow_sample_steps=int(prism_cfg.get("flow_sample_steps", 64)),
        parallel_sample_steps=int(prism_cfg.get("parallel_sample_steps", 64)),
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
    audio_decoder = _build_audio_decoder(module_cfg)

    return PrismTTSLightning(
        model=model,
        learning_rate=float(module_cfg.get("learning_rate", 3.0e-4)),
        weight_decay=float(module_cfg.get("weight_decay", 0.01)),
        betas=_coerce_betas(module_cfg.get("betas", [0.9, 0.95])),
        eval_every_n_steps=int(module_cfg.get("eval_every_n_steps", 5000)),
        scheduler_factory=scheduler_factory,
        audio_decoder=audio_decoder,
        audio_sample_rate=int(module_cfg.get("audio_sample_rate", 24_000)),
        max_audio_samples=int(module_cfg.get("max_audio_samples", 2)),
        log_media_on_validation_end=bool(module_cfg.get("log_media_on_validation_end", True)),
        ema_decay=float(module_cfg.get("ema_decay", 0.999)),
        ema_start_step=int(module_cfg.get("ema_start_step", 0)),
        ema_update_every_n_steps=int(module_cfg.get("ema_update_every_n_steps", 1)),
        ema_warmup_steps=int(module_cfg.get("ema_warmup_steps", 0)),
        use_ema_for_validation=bool(module_cfg.get("use_ema_for_validation", True)),
        use_ema_for_periodic_eval=bool(module_cfg.get("use_ema_for_periodic_eval", True)),
        sync_dist_logging=bool(module_cfg.get("sync_dist_logging", False)),
    )


def _resolve_import_string(path: str, *, field_name: str) -> Any:
    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, sep, attr_name = path.rpartition(".")
        if sep == "":
            raise ValueError(
                f"{field_name} must be an import path like 'pkg.mod:obj' or 'pkg.mod.obj'."
            )

    module_name = module_name.strip()
    attr_name = attr_name.strip()
    if module_name == "" or attr_name == "":
        raise ValueError(
            f"{field_name} must be an import path like 'pkg.mod:obj' or 'pkg.mod.obj'."
        )

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Unable to import module {module_name!r} referenced by {field_name}."
        ) from exc

    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(
            f"Unable to resolve attribute {attr_name!r} in module {module_name!r} "
            f"for {field_name}."
        ) from exc


def _build_audio_decoder(module_cfg: dict[str, Any]) -> Any | None:
    decoder_spec = module_cfg.get("audio_decoder")
    if decoder_spec is None:
        return None
    if isinstance(decoder_spec, str) and decoder_spec.strip() == "":
        return None
    if not isinstance(decoder_spec, str):
        raise ValueError(
            "trainer.lightning_module.audio_decoder must be a string import path "
            "or null."
        )

    decoder_kwargs = module_cfg.get("audio_decoder_kwargs", {})
    if decoder_kwargs is None:
        decoder_kwargs = {}
    if not isinstance(decoder_kwargs, dict):
        raise ValueError(
            "trainer.lightning_module.audio_decoder_kwargs must be a mapping when set."
        )

    decoder_obj = _resolve_import_string(
        decoder_spec.strip(),
        field_name="trainer.lightning_module.audio_decoder",
    )
    if not callable(decoder_obj):
        raise ValueError(
            "trainer.lightning_module.audio_decoder must resolve to a callable."
        )

    if inspect.isclass(decoder_obj):
        instance = decoder_obj(**decoder_kwargs)
        if callable(instance):
            return instance
        decode_method = getattr(instance, "decode", None)
        if callable(decode_method):
            return decode_method
        raise ValueError(
            "Audio decoder class instance must be callable or expose a callable `decode` method."
        )

    if decoder_kwargs:
        return lambda latents, fn=decoder_obj, kwargs=dict(decoder_kwargs): fn(latents, **kwargs)
    return decoder_obj


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


def _shared_memory_total_bytes() -> int | None:
    try:
        return int(shutil.disk_usage("/dev/shm").total)
    except OSError:
        return None


def _parse_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return _parse_bool_string(raw, field_name=name)
    except ValueError:
        print(f"[train.py] Ignoring invalid {name}={raw!r}; using default={default}.")
        return default


def _parse_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[train.py] Ignoring invalid {name}={raw!r}; using default={default}.")
        return default


def _length_quantile(values: Sequence[int], quantile: float) -> int:
    if not values:
        raise ValueError("values must not be empty.")
    if quantile <= 0.0 or quantile > 1.0:
        raise ValueError(f"quantile must be in (0, 1], got {quantile}.")

    sorted_values = sorted(max(1, int(value)) for value in values)
    rank = max(0, min(len(sorted_values) - 1, int(math.ceil(quantile * len(sorted_values))) - 1))
    return int(sorted_values[rank])


def _should_force_single_process_loader(num_workers: int) -> bool:
    if num_workers <= 0:
        return False

    # Allow explicit opt-out for environments where low /dev/shm is still acceptable.
    if _parse_env_bool("PRISM_TTS_DISABLE_SHM_GUARD", False):
        return False

    total_bytes = _shared_memory_total_bytes()
    if total_bytes is None:
        return False

    default_threshold = 512 * 1024 * 1024
    threshold = _parse_env_int("PRISM_TTS_MIN_SHM_BYTES", default_threshold)
    return total_bytes < threshold


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

    dataset_stream_count = dataset_cfg.get("discrete_stream_count")
    max_discrete_stream_count = None
    if dataset_stream_count is not None:
        max_discrete_stream_count = int(dataset_stream_count)

    max_active_stream_raw = collate_cfg.get("max_active_discrete_stream_count")
    max_active_stream_count = (
        max_discrete_stream_count
        if max_active_stream_raw is None
        else int(max_active_stream_raw)
    )
    min_active_stream_count = int(collate_cfg.get("min_active_discrete_stream_count", 1))
    random_active_stream_train = bool(
        collate_cfg.get("random_active_discrete_stream_count_train", True)
    )

    train_collate = BatchCollate(
        text_pad_value=collate_cfg.get("text_pad_value"),
        discrete_pad_value=collate_cfg.get("discrete_pad_value"),
        continuous_pad_value=float(collate_cfg.get("continuous_pad_value", 0.0)),
        include_attention_mask=bool(collate_cfg.get("include_attention_mask", True)),
        discrete_token_count=discrete_token_count,
        random_active_discrete_stream_count=random_active_stream_train,
        min_active_discrete_stream_count=min_active_stream_count,
        max_active_discrete_stream_count=max_active_stream_count,
        fixed_continuous_stream_idx=max_discrete_stream_count,
    )
    eval_collate = BatchCollate(
        text_pad_value=collate_cfg.get("text_pad_value"),
        discrete_pad_value=collate_cfg.get("discrete_pad_value"),
        continuous_pad_value=float(collate_cfg.get("continuous_pad_value", 0.0)),
        include_attention_mask=bool(collate_cfg.get("include_attention_mask", True)),
        discrete_token_count=discrete_token_count,
        random_active_discrete_stream_count=False,
        min_active_discrete_stream_count=min_active_stream_count,
        max_active_discrete_stream_count=max_discrete_stream_count,
        fixed_continuous_stream_idx=max_discrete_stream_count,
    )

    num_workers = int(loader_cfg.get("num_workers", 0))
    persistent_workers = bool(loader_cfg.get("persistent_workers", False)) and num_workers > 0
    pin_memory = bool(loader_cfg.get("pin_memory", False))

    if _should_force_single_process_loader(num_workers):
        shm_total = _shared_memory_total_bytes()
        shm_mb = "unknown"
        if shm_total is not None:
            shm_mb = f"{(shm_total / (1024 * 1024)):.1f}"
        print(
            "[train.py] Detected low shared memory "
            f"(/dev/shm={shm_mb} MB). "
            "Overriding data.loader.num_workers=0 and persistent_workers=false "
            "to avoid DataLoader bus errors. "
            "Set PRISM_TTS_DISABLE_SHM_GUARD=true to keep configured worker settings."
        )
        num_workers = 0
        persistent_workers = False

    common_loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    prefetch_factor = loader_cfg.get("prefetch_factor")
    if num_workers > 0 and prefetch_factor is not None:
        common_loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_batch_size = int(loader_cfg.get("train_batch_size", 8))
    shuffle_train = bool(loader_cfg.get("shuffle_train", True))
    configured_drop_last_train = bool(loader_cfg.get("drop_last_train", False))
    if configured_drop_last_train:
        print(
            "[train.py] data.loader.drop_last_train=true is not allowed; "
            "overriding to false so all samples are used in training."
        )
    drop_last_train = False

    adaptive_cfg_raw = loader_cfg.get("adaptive_batching")
    adaptive_cfg: dict[str, Any] = {}
    if adaptive_cfg_raw is not None:
        if not isinstance(adaptive_cfg_raw, dict):
            raise ValueError("data.loader.adaptive_batching must be a mapping when provided.")
        adaptive_cfg = adaptive_cfg_raw
    adaptive_enabled = bool(adaptive_cfg.get("enabled", False))

    if adaptive_enabled:
        target_memory_utilization = float(adaptive_cfg.get("target_memory_utilization", 0.8))
        if target_memory_utilization <= 0.0 or target_memory_utilization > 1.0:
            raise ValueError(
                "data.loader.adaptive_batching.target_memory_utilization must be in (0, 1]."
            )

        max_batch_size = int(adaptive_cfg.get("max_batch_size", max(1, train_batch_size * 2)))
        if max_batch_size < 1:
            raise ValueError("data.loader.adaptive_batching.max_batch_size must be >= 1.")

        reference_quantile = float(adaptive_cfg.get("reference_length_quantile", 0.95))
        if reference_quantile <= 0.0 or reference_quantile > 1.0:
            raise ValueError(
                "data.loader.adaptive_batching.reference_length_quantile must be in (0, 1]."
            )

        sample_lengths = estimate_prism_sample_lengths(
            train_dataset,
            codec_frame_rate_hz=float(collate_cfg.get("codec_frame_rate_hz", 12.5)),
        )

        reference_length = _length_quantile(sample_lengths, reference_quantile)
        memory_budget_raw = adaptive_cfg.get("memory_budget")
        if memory_budget_raw is None:
            memory_budget = train_batch_size * (reference_length**2)
        else:
            memory_budget = int(memory_budget_raw)
        if memory_budget < 1:
            raise ValueError("data.loader.adaptive_batching.memory_budget must be >= 1.")

        target_batch_cost_raw = adaptive_cfg.get("target_batch_cost")
        if target_batch_cost_raw is None:
            target_batch_cost = int(max(1, round(memory_budget * target_memory_utilization)))
        else:
            target_batch_cost = int(target_batch_cost_raw)
        if target_batch_cost < 1:
            raise ValueError("data.loader.adaptive_batching.target_batch_cost must be >= 1.")

        trainer_cfg = config.get("trainer")
        seed_fallback = 0
        if isinstance(trainer_cfg, dict):
            trainer_seed = trainer_cfg.get("seed")
            if trainer_seed is not None:
                seed_fallback = int(trainer_seed)
        adaptive_seed_raw = adaptive_cfg.get("seed", seed_fallback)
        if adaptive_seed_raw is None:
            sampler_seed = int.from_bytes(os.urandom(8), byteorder="big") & 0x7FFF_FFFF
            print(
                "[train.py] data.loader.adaptive_batching.seed is null; "
                f"generated random sampler seed={sampler_seed}."
            )
        else:
            sampler_seed = int(adaptive_seed_raw)

        train_batch_sampler = AdaptiveMemoryBatchSampler(
            sample_lengths=sample_lengths,
            target_batch_cost=target_batch_cost,
            max_batch_size=max_batch_size,
            shuffle=shuffle_train,
            drop_last=drop_last_train,
            seed=sampler_seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=train_collate,
            **common_loader_kwargs,
        )

        print(
            "[train.py] Adaptive train batching enabled: "
            f"target_memory_utilization={target_memory_utilization:.2f}, "
            f"target_batch_cost={target_batch_cost}, "
            f"memory_budget={memory_budget}, "
            f"reference_length(q={reference_quantile:.2f})={reference_length}, "
            f"max_batch_size={max_batch_size}, "
            f"seed={sampler_seed}."
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=shuffle_train,
            drop_last=drop_last_train,
            collate_fn=train_collate,
            **common_loader_kwargs,
        )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(loader_cfg.get("val_batch_size", 8)),
            shuffle=bool(loader_cfg.get("shuffle_val", False)),
            drop_last=False,
            collate_fn=eval_collate,
            **common_loader_kwargs,
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(loader_cfg.get("test_batch_size", 8)),
            shuffle=bool(loader_cfg.get("shuffle_test", False)),
            drop_last=False,
            collate_fn=eval_collate,
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
            "entity": logger_cfg.get("entity"),
            "group": logger_cfg.get("group"),
            "tags": logger_cfg.get("tags"),
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
        save_every_validation_stage = bool(
            checkpoint_kwargs.pop("save_every_validation_stage", False)
        )
        every_val_filename = str(
            checkpoint_kwargs.pop(
                "every_val_filename",
                "prism_tts-val-stage={val_stage:05d}-step={step:07d}",
            )
        )
        every_val_save_weights_only = bool(
            checkpoint_kwargs.pop("every_val_save_weights_only", False)
        )
        dirpath = checkpoint_kwargs.get("dirpath")
        resolved_ckpt_dir = Path.cwd() / "checkpoints"
        if dirpath is not None:
            ckpt_dir = Path(str(dirpath)).expanduser()
            if not ckpt_dir.is_absolute():
                ckpt_dir = Path.cwd() / ckpt_dir
            ckpt_dir = ckpt_dir.resolve()
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_kwargs["dirpath"] = str(ckpt_dir)
            resolved_ckpt_dir = ckpt_dir
        else:
            resolved_ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_kwargs = _filter_kwargs_for_callable(
            ModelCheckpoint.__init__,
            checkpoint_kwargs,
            context="ModelCheckpoint",
        )
        callbacks.append(ModelCheckpoint(**checkpoint_kwargs))
        if save_every_validation_stage:
            callbacks.append(
                SaveEveryValidationStageCheckpoint(
                    dirpath=resolved_ckpt_dir,
                    filename=every_val_filename,
                    save_weights_only=every_val_save_weights_only,
                )
            )

    if bool(scheduler_cfg.get("enabled", True)):
        lr_monitor_kwargs = _filter_kwargs_for_callable(
            LearningRateMonitor.__init__,
            {"logging_interval": "step"},
            context="LearningRateMonitor",
        )
        callbacks.append(LearningRateMonitor(**lr_monitor_kwargs))

    return callbacks


class SaveEveryValidationStageCheckpoint(pl.Callback):
    """Persist a checkpoint at the end of every validation stage."""

    def __init__(
        self,
        *,
        dirpath: Path,
        filename: str,
        save_weights_only: bool,
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath).expanduser().resolve()
        self.filename = filename
        self.save_weights_only = save_weights_only
        self._val_stage = 0

    def state_dict(self) -> dict[str, Any]:
        return {"val_stage": int(self._val_stage)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._val_stage = int(state_dict.get("val_stage", 0))

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        if trainer.sanity_checking:
            return
        if not bool(getattr(trainer, "is_global_zero", True)):
            return

        self._val_stage += 1
        format_values = {
            "step": int(trainer.global_step),
            "epoch": int(trainer.current_epoch),
            "val_stage": int(self._val_stage),
        }
        filename = self.filename.format(**format_values)
        if not filename.endswith(".ckpt"):
            filename = f"{filename}.ckpt"

        self.dirpath.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self._next_available_path(self.dirpath / filename)
        trainer.save_checkpoint(str(checkpoint_path), weights_only=self.save_weights_only)

    @staticmethod
    def _next_available_path(path: Path) -> Path:
        if not path.exists():
            return path
        version = 1
        while True:
            candidate = path.with_name(f"{path.stem}-v{version}{path.suffix}")
            if not candidate.exists():
                return candidate
            version += 1


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


def _resolve_requested_device_count(
    *,
    accelerator: str,
    devices: Any,
    gpu_indices: list[int] | None,
    num_nodes: int,
) -> int | None:
    if num_nodes > 1:
        # Keep distributed strategy for multi-node setups even if per-node device count is 1.
        return None

    if gpu_indices is not None:
        return len(gpu_indices)

    if isinstance(devices, int):
        return devices
    if isinstance(devices, (list, tuple)):
        return len(devices)
    if isinstance(devices, str):
        lowered = devices.strip().lower()
        if lowered == "auto":
            if accelerator in {"gpu", "cuda"} and torch.cuda.is_available():
                return torch.cuda.device_count()
            return 1
        if lowered.isdigit():
            return int(lowered)
    return None


def _apply_distributed_training_config(config: dict[str, Any]) -> None:
    trainer_cfg = _require_mapping(config, "trainer")
    distributed_cfg = trainer_cfg.get("distributed")
    if not isinstance(distributed_cfg, dict) or not bool(distributed_cfg.get("enabled", False)):
        return

    lightning_trainer_cfg = _require_mapping(trainer_cfg, "lightning_trainer")
    accelerator = str(distributed_cfg.get("accelerator", "gpu"))
    lightning_trainer_cfg["accelerator"] = accelerator
    gpu_indices = distributed_cfg.get("gpu_indices")
    resolved_gpu_indices: list[int] | None = None
    if gpu_indices is not None:
        resolved_gpu_indices = _coerce_gpu_indices(gpu_indices)
        lightning_trainer_cfg["devices"] = resolved_gpu_indices
    else:
        lightning_trainer_cfg["devices"] = distributed_cfg.get("devices", "auto")

    num_nodes = distributed_cfg.get("num_nodes")
    resolved_num_nodes = 1
    if num_nodes is not None:
        resolved_num_nodes = int(num_nodes)
        lightning_trainer_cfg["num_nodes"] = resolved_num_nodes

    requested_strategy = str(
        distributed_cfg.get("strategy", "ddp_find_unused_parameters_false")
    )
    device_count = _resolve_requested_device_count(
        accelerator=accelerator,
        devices=lightning_trainer_cfg.get("devices"),
        gpu_indices=resolved_gpu_indices,
        num_nodes=resolved_num_nodes,
    )
    if device_count is not None and device_count <= 1 and requested_strategy.startswith("ddp"):
        lightning_trainer_cfg["strategy"] = "auto"
        print(
            "[train.py] Distributed mode requested with <=1 device; "
            "overriding strategy to 'auto' to avoid single-process DDP overhead/issues."
        )
    else:
        lightning_trainer_cfg["strategy"] = requested_strategy

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
    if _DEFAULTED_CUDA_ALLOC_CONF:
        print(
            "[train.py] PYTORCH_CUDA_ALLOC_CONF was unset; defaulting to "
            "'backend:cudaMallocAsync' to avoid NVML-related allocator assertions."
        )
    _apply_wandb_cli_overrides(config, args)

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
