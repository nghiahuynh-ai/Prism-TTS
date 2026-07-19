from __future__ import annotations

import argparse
import importlib
import inspect
import math
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULTED_CUDA_ALLOC_CONF = False
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    _DEFAULTED_CUDA_ALLOC_CONF = True

# Keep a process-level timestamp so the startup report includes expensive imports
# (notably torch, transformers, and Lightning), which occur before ``main``.
_PROCESS_START = time.perf_counter()

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
from transformers import LlamaConfig

from dataset.adaptive_batching import AdaptiveMemoryBatchSampler, estimate_prism_sample_lengths
from dataset.dataset import (
    BatchCollate,
    LazyPrismDataset,
    PrismDataset,
    build_shared_token_layout,
)
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

    continuous_latent_size = int(prism_cfg["continuous_latent_size"])
    dataset_continuous_dim_raw = dataset_cfg.get("continuous_feature_dim")
    if dataset_continuous_dim_raw is not None:
        dataset_continuous_dim = int(dataset_continuous_dim_raw)
        if continuous_latent_size != dataset_continuous_dim:
            raise ValueError(
                "model.prism_tts.continuous_latent_size must match "
                f"data.dataset.continuous_feature_dim ({dataset_continuous_dim})."
            )

    eos_loss_weight = float(prism_cfg.get("eos_loss_weight", 1.0))
    if eos_loss_weight < 0.0:
        raise ValueError("model.prism_tts.eos_loss_weight must be >= 0.")

    # The unified embedding table must cover text tokens (which begin at text_offset).
    llama_vocab_size = int(llama_cfg["vocab_size"])
    if llama_vocab_size < text_offset:
        raise ValueError(
            "model.llama_config.vocab_size is too small for the shared token layout. "
            f"Need at least {text_offset} to cover special + text tokens, got {llama_vocab_size}."
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
        continuous_latent_size=int(prism_cfg["continuous_latent_size"]),
        flow_num_res_blocks=int(prism_cfg.get("flow_num_res_blocks", 4)),
        flow_model_channels=prism_cfg.get("flow_model_channels"),
        eos_loss_weight=float(prism_cfg.get("eos_loss_weight", 1.0)),
        sigma_data=float(prism_cfg.get("sigma_data", 1.0)),
        p_mean=float(prism_cfg.get("p_mean", -1.0)),
        p_std=float(prism_cfg.get("p_std", 1.6)),
        tangent_warmup_steps=int(prism_cfg.get("tangent_warmup_steps", 1000)),
        tangent_norm_const=float(prism_cfg.get("tangent_norm_const", 0.1)),
        sample_steps=int(prism_cfg.get("sample_steps", 1)),
        eos_threshold=float(prism_cfg.get("eos_threshold", 0.5)),
        latent_stats_decay=float(prism_cfg.get("latent_stats_decay", 0.999)),
        inject_backbone_noise=bool(prism_cfg.get("inject_backbone_noise", True)),
        use_short_context=bool(prism_cfg.get("use_short_context", True)),
        short_context_layers=int(prism_cfg.get("short_context_layers", 2)),
        short_context_window=int(prism_cfg.get("short_context_window", 10)),
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


def _instantiate_audio_decoder(
    decoder_obj: Any,
    decoder_kwargs: dict[str, Any],
) -> Any:
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


class LazyAudioDecoder:
    """Initialize an optional audio decoder only when media logging needs it."""

    def __init__(self, decoder_spec: str, decoder_kwargs: dict[str, Any]) -> None:
        self.decoder_spec = decoder_spec
        self.decoder_kwargs = dict(decoder_kwargs)
        self._decoder: Any | None = None
        self._declared_sample_rate = self.decoder_kwargs.get("sample_rate")

    @property
    def sample_rate(self) -> Any | None:
        if self._decoder is None:
            return self._declared_sample_rate
        return getattr(self._decoder, "sample_rate", self._declared_sample_rate)

    def _load(self) -> Any:
        if self._decoder is not None:
            return self._decoder

        start = time.perf_counter()
        print(f"[train.py] Initializing lazy audio decoder: {self.decoder_spec}")
        decoder_obj = _resolve_import_string(
            self.decoder_spec,
            field_name="trainer.lightning_module.audio_decoder",
        )
        if not callable(decoder_obj):
            raise ValueError(
                "trainer.lightning_module.audio_decoder must resolve to a callable."
            )

        self._decoder = _instantiate_audio_decoder(decoder_obj, self.decoder_kwargs)
        print(
            f"[train.py] Audio decoder initialized in {time.perf_counter() - start:.2f}s."
        )
        return self._decoder

    def __call__(self, latents: torch.FloatTensor) -> torch.Tensor | Any:
        return self._load()(latents)


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

    decoder_spec = decoder_spec.strip()
    lazy = bool(module_cfg.get("audio_decoder_lazy", True))
    if lazy:
        print(
            "[train.py] Audio decoder initialization is lazy; "
            "first media decode will load it."
        )
        return LazyAudioDecoder(decoder_spec, decoder_kwargs)

    decoder_obj = _resolve_import_string(
        decoder_spec,
        field_name="trainer.lightning_module.audio_decoder",
    )
    if not callable(decoder_obj):
        raise ValueError(
            "trainer.lightning_module.audio_decoder must resolve to a callable."
        )

    return _instantiate_audio_decoder(decoder_obj, decoder_kwargs)


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


class StartupPrefetchDataLoader(DataLoader):
    """Reuse a worker-prefetched iterator for the first training epoch only.

    ``DataLoader.__iter__`` resets persistent workers on every call.  A normal
    early ``iter(loader)`` would therefore cause Lightning to discard and
    re-read its startup prefetch.  This small specialization returns the primed
    iterator exactly once, then preserves normal DataLoader behavior for all
    later epochs.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reuse_startup_iterator = False

    def warmup_workers(self) -> bool:
        """Start persistent workers and retain their first prefetched iterator."""
        if self.num_workers <= 0 or not self.persistent_workers:
            return False
        if self._iterator is not None:
            return True

        start = time.perf_counter()
        # Call the base implementation directly so this first iterator is
        # marked for reuse rather than being returned through ``__iter__``.
        DataLoader.__iter__(self)
        if self._iterator is None:  # defensive for incompatible DataLoader variants
            return False

        self._reuse_startup_iterator = True
        print(
            f"[train.py] Started {self.num_workers} persistent train DataLoader workers in "
            f"{time.perf_counter() - start:.2f}s; initial prefetch overlaps model setup."
        )
        return True

    def __iter__(self):
        if self._reuse_startup_iterator and self._iterator is not None:
            self._reuse_startup_iterator = False
            return self._iterator
        return super().__iter__()


def _warmup_persistent_loader_workers(loader: DataLoader) -> bool:
    """Start persistent loader workers without discarding their first prefetch.

    PyTorch starts DataLoader workers lazily on the first ``iter(loader)``.  If
    that happens in ``Trainer.fit``, Unix ``fork`` workers inherit the fully
    constructed model (and sometimes initialized CUDA/logger state).  Starting
    them before model construction avoids that expensive fork and lets their
    initial prefetch overlap model initialization.

    ``StartupPrefetchDataLoader`` returns the primed iterator once, so
    Lightning consumes that prefetch rather than resetting and re-reading it.
    """
    warmup = getattr(loader, "warmup_workers", None)
    return bool(warmup()) if callable(warmup) else False


def _build_data_objects(
    config: dict[str, Any],
    *,
    warmup_train_workers: bool = True,
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
        "continuous_feature_dim": dataset_cfg.get("continuous_feature_dim"),
        "append_eos_to_text": bool(dataset_cfg.get("append_eos_to_text", False)),
        "cache_npy": bool(dataset_cfg.get("cache_npy", False)),
        "load_prompt": bool(dataset_cfg.get("load_prompt", False)),
        "manifest_progress_every": int(dataset_cfg.get("manifest_progress_every", 0)),
        "manifest_read_buffer_bytes": int(
            dataset_cfg.get("manifest_read_buffer_bytes", 4 * 1024 * 1024)
        ),
    }

    metadata_mode = str(dataset_cfg.get("metadata_mode", "eager")).strip().lower()
    if metadata_mode not in {"eager", "lazy"}:
        raise ValueError("data.dataset.metadata_mode must be either 'eager' or 'lazy'.")
    shuffle_train = bool(loader_cfg.get("shuffle_train", True))
    metadata_shuffle_buffer_size = int(
        dataset_cfg.get("metadata_shuffle_buffer_size", 256)
    )
    metadata_shuffle_seed_raw = dataset_cfg.get("metadata_shuffle_seed")
    metadata_shuffle_seed = (
        None if metadata_shuffle_seed_raw is None else int(metadata_shuffle_seed_raw)
    )

    def build_manifest_dataset(
        manifest: str,
        *,
        split: str,
    ) -> PrismDataset | LazyPrismDataset:
        if metadata_mode == "eager":
            return PrismDataset(source=manifest, **dataset_kwargs)
        return LazyPrismDataset(
            source=manifest,
            shuffle_manifest=split == "train" and shuffle_train,
            shuffle_buffer_size=metadata_shuffle_buffer_size,
            shuffle_seed=metadata_shuffle_seed,
            **dataset_kwargs,
        )

    def dataset_size_label(dataset: PrismDataset | LazyPrismDataset) -> str:
        if isinstance(dataset, IterableDataset):
            return "streaming metadata"
        return f"{len(dataset):,} samples"

    train_manifest = _optional_path(data_cfg.get("train_manifest"))
    if train_manifest is None:
        raise ValueError("data.train_manifest must be set for training.")
    _t0 = time.perf_counter()
    print(f"[train.py] Loading train manifest: {train_manifest}", flush=True)
    train_start = time.perf_counter()
    train_dataset = build_manifest_dataset(train_manifest, split="train")
    print(
        f"[train.py] Train manifest ready in {time.perf_counter() - train_start:.2f}s "
        f"({dataset_size_label(train_dataset)}; metadata_mode={metadata_mode}).",
        flush=True,
    )

    val_manifest = _optional_path(data_cfg.get("val_manifest"))
    val_dataset = None
    if val_manifest:
        print(f"[train.py] Loading validation manifest: {val_manifest}", flush=True)
        val_start = time.perf_counter()
        val_dataset = build_manifest_dataset(val_manifest, split="val")
        print(
            f"[train.py] Validation manifest ready in {time.perf_counter() - val_start:.2f}s "
            f"({dataset_size_label(val_dataset)}; metadata_mode={metadata_mode}).",
            flush=True,
        )

    test_manifest = _optional_path(data_cfg.get("test_manifest"))
    test_dataset = None
    if test_manifest:
        print(f"[train.py] Loading test manifest: {test_manifest}", flush=True)
        test_start = time.perf_counter()
        test_dataset = build_manifest_dataset(test_manifest, split="test")
        print(
            f"[train.py] Test manifest ready in {time.perf_counter() - test_start:.2f}s "
            f"({dataset_size_label(test_dataset)}; metadata_mode={metadata_mode}).",
            flush=True,
        )
    print(
        f"[train.py] Datasets built in {time.perf_counter() - _t0:.2f}s "
        f"(train={dataset_size_label(train_dataset)}"
        + (f", val={dataset_size_label(val_dataset)}" if val_dataset is not None else "")
        + (f", test={dataset_size_label(test_dataset)}" if test_dataset is not None else "")
        + f", metadata_mode={metadata_mode}, load_prompt={dataset_kwargs['load_prompt']})."
    )

    collate = BatchCollate(
        text_pad_value=collate_cfg.get("text_pad_value"),
        continuous_pad_value=float(collate_cfg.get("continuous_pad_value", 0.0)),
        include_attention_mask=bool(collate_cfg.get("include_attention_mask", True)),
        discrete_token_count=discrete_token_count,
    )

    num_workers = int(loader_cfg.get("num_workers", 0))
    persistent_workers = bool(loader_cfg.get("persistent_workers", False)) and num_workers > 0
    pin_memory = bool(loader_cfg.get("pin_memory", False))

    if _should_force_single_process_loader(num_workers):
        shm_total = _shared_memory_total_bytes()
        shm_mb = "unknown"
        if shm_total is not None:
            shm_mb = f"{(shm_total / (1024 * 1024)):.1f}"
        # Low /dev/shm causes DataLoader bus errors under the default
        # 'file_descriptor' sharing strategy. Prefer switching to 'file_system'
        # sharing so parallel workers are kept -- dropping to num_workers=0 would
        # serialize all (networked) npy reads and make training crawl from step 0.
        force_single = _parse_env_bool("PRISM_TTS_FORCE_SINGLE_PROCESS_LOADER", False)
        switched = False
        if not force_single:
            try:
                import torch.multiprocessing as _mp

                _mp.set_sharing_strategy("file_system")
                switched = True
            except Exception as exc:  # pragma: no cover - platform dependent
                print(f"[train.py] Could not set 'file_system' sharing strategy: {exc}")
        if switched:
            print(
                f"[train.py] Low shared memory (/dev/shm={shm_mb} MB): set multiprocessing "
                f"sharing_strategy='file_system' to keep num_workers={num_workers} without bus "
                "errors. Set PRISM_TTS_FORCE_SINGLE_PROCESS_LOADER=true to fall back to "
                "num_workers=0, or PRISM_TTS_DISABLE_SHM_GUARD=true to ignore this guard."
            )
        else:
            print(
                f"[train.py] Low shared memory (/dev/shm={shm_mb} MB): overriding "
                "num_workers=0 and persistent_workers=false to avoid DataLoader bus errors."
            )
            num_workers = 0
            persistent_workers = False

    print(
        "[train.py] Effective dataloader: "
        f"num_workers={num_workers}, persistent_workers={persistent_workers}, "
        f"pin_memory={pin_memory}, prefetch_factor={loader_cfg.get('prefetch_factor')}, "
        f"train_batch_size={int(loader_cfg.get('train_batch_size', 8))}."
    )

    common_loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "collate_fn": collate,
    }
    prefetch_factor = loader_cfg.get("prefetch_factor")
    if num_workers > 0 and prefetch_factor is not None:
        common_loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_batch_size = int(loader_cfg.get("train_batch_size", 8))
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
        if isinstance(train_dataset, IterableDataset):
            raise ValueError(
                "Adaptive batching requires data.dataset.metadata_mode='eager' because "
                "it needs every manifest sample length before training starts."
            )
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
        train_loader = StartupPrefetchDataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
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
        train_loader = StartupPrefetchDataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=False if isinstance(train_dataset, IterableDataset) else shuffle_train,
            drop_last=drop_last_train,
            **common_loader_kwargs,
        )

    # DataLoader worker creation is otherwise deferred until Trainer.fit, at
    # which point the 333M-parameter model is already resident in the parent.
    # Only prime local, persistent workers; DDP launches its own rank-local
    # loaders after Lightning has spawned the training processes.
    distributed_cfg = _require_mapping(config, "trainer").get("distributed")
    distributed_enabled = isinstance(distributed_cfg, dict) and bool(
        distributed_cfg.get("enabled", False)
    )
    worker_warmup_enabled = bool(loader_cfg.get("warmup_workers", False))
    if worker_warmup_enabled and warmup_train_workers and not distributed_enabled:
        if not _warmup_persistent_loader_workers(train_loader):
            print(
                "[train.py] Train-worker warmup skipped: it requires "
                "num_workers > 0 and persistent_workers=true."
            )
    elif worker_warmup_enabled and distributed_enabled:
        print(
            "[train.py] Train-worker warmup skipped for distributed training; "
            "each Lightning rank starts its own workers."
        )

    val_loader = None
    if val_dataset is not None:
        shuffle_val = bool(loader_cfg.get("shuffle_val", False))
        if isinstance(val_dataset, IterableDataset) and shuffle_val:
            print(
                "[train.py] Ignoring data.loader.shuffle_val=true for lazy metadata; "
                "validation streams in manifest order."
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(loader_cfg.get("val_batch_size", 8)),
            shuffle=False if isinstance(val_dataset, IterableDataset) else shuffle_val,
            drop_last=False,
            **common_loader_kwargs,
        )

    test_loader = None
    if test_dataset is not None:
        shuffle_test = bool(loader_cfg.get("shuffle_test", False))
        if isinstance(test_dataset, IterableDataset) and shuffle_test:
            print(
                "[train.py] Ignoring data.loader.shuffle_test=true for lazy metadata; "
                "test streams in manifest order."
            )
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(loader_cfg.get("test_batch_size", 8)),
            shuffle=False if isinstance(test_dataset, IterableDataset) else shuffle_test,
            drop_last=False,
            **common_loader_kwargs,
        )

    if _parse_env_bool("PRISM_TTS_PROFILE_FIRST_BATCH", False):
        _profile_loader_throughput(train_loader)

    return train_loader, val_loader, test_loader


def _profile_loader_throughput(loader: DataLoader, num_batches: int = 5) -> None:
    """Time the first few batches to separate data-pipeline cost from model cost.

    Opt-in via PRISM_TTS_PROFILE_FIRST_BATCH=true. Uses a throwaway iterator, so it
    consumes a few samples but does not affect the real training iterator.
    """
    print(f"[train.py] Profiling first {num_batches} train batches (data pipeline only)...")
    iterator = iter(loader)
    start = time.perf_counter()
    first_batch_time = None
    seen = 0
    for _ in range(num_batches):
        step_start = time.perf_counter()
        try:
            next(iterator)
        except StopIteration:
            break
        now = time.perf_counter()
        if first_batch_time is None:
            first_batch_time = now - step_start
        seen += 1
    total = time.perf_counter() - start
    if seen > 0 and first_batch_time is not None:
        steady = (total - first_batch_time) / max(1, seen - 1) if seen > 1 else first_batch_time
        print(
            f"[train.py] Data pipeline: first batch {first_batch_time:.2f}s, "
            f"~{steady:.3f}s/batch steady over {seen} batches "
            "(if this is fast, remaining slowness is model/CUDA, not data)."
        )
    del iterator


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


class TrainingStartupTimer(pl.Callback):
    """Report the time from process launch through the first training batch."""

    def __init__(self, process_start: float) -> None:
        super().__init__()
        self.process_start = float(process_start)
        self.fit_start: float | None = None
        self.first_batch_start: float | None = None
        self._reported_first_batch = False
        self._reported_first_batch_end = False

    @staticmethod
    def _is_global_zero(trainer: Any) -> bool:
        return bool(getattr(trainer, "is_global_zero", True))

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        if not self._is_global_zero(trainer):
            return
        self.fit_start = time.perf_counter()
        print(
            "[train.py] Trainer.fit entered after "
            f"{self.fit_start - self.process_start:.2f}s from process start."
        )

    def on_train_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, batch
        if batch_idx != 0 or self._reported_first_batch or not self._is_global_zero(trainer):
            return
        self.first_batch_start = time.perf_counter()
        fit_elapsed = (
            self.first_batch_start - self.fit_start
            if self.fit_start is not None
            else float("nan")
        )
        print(
            "[train.py] First train batch reached the training loop after "
            f"{self.first_batch_start - self.process_start:.2f}s from process start "
            f"({fit_elapsed:.2f}s inside Trainer.fit)."
        )
        self._reported_first_batch = True

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch
        if batch_idx != 0 or self._reported_first_batch_end or not self._is_global_zero(trainer):
            return
        now = time.perf_counter()
        batch_elapsed = (
            now - self.first_batch_start if self.first_batch_start is not None else float("nan")
        )
        print(
            "[train.py] First train batch finished after "
            f"{now - self.process_start:.2f}s from process start "
            f"({batch_elapsed:.2f}s in the batch loop)."
        )
        self._reported_first_batch_end = True


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


def _log_startup_stage(stage_name: str, stage_start: float, run_start: float) -> None:
    print(
        f"[train.py] {stage_name} completed in {time.perf_counter() - stage_start:.2f}s "
        f"(startup total {time.perf_counter() - run_start:.2f}s)."
    )


def run(args: argparse.Namespace, *, process_start: float | None = None) -> None:
    run_start = time.perf_counter() if process_start is None else float(process_start)
    stage_start = time.perf_counter()
    resolved = _load_merged_configs(args)
    _log_startup_stage("Config loading", stage_start, run_start)

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

    stage_start = time.perf_counter()
    train_loader, val_loader, test_loader = _build_data_objects(
        config,
        warmup_train_workers=not args.validate_only,
    )
    _log_startup_stage("Data setup", stage_start, run_start)

    stage_start = time.perf_counter()
    model = _build_model(config)
    _log_startup_stage("Model construction", stage_start, run_start)

    stage_start = time.perf_counter()
    lightning_module = _build_lightning_module(config, model=model)
    _log_startup_stage("Lightning module construction", stage_start, run_start)

    stage_start = time.perf_counter()
    logger = _build_logger(_require_mapping(trainer_cfg, "logger"))
    callbacks = _build_callbacks(config)
    callbacks.append(TrainingStartupTimer(process_start=run_start))

    trainer = _build_trainer(config, logger=logger, callbacks=callbacks)
    _log_startup_stage("Logger/callback/trainer setup", stage_start, run_start)

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
    run(args, process_start=_PROCESS_START)


if __name__ == "__main__":
    main()
