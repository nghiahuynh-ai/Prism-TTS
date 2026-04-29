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

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ImportError(
        "Training utilities require `PyYAML` (`pip install pyyaml`)."
    ) from exc


def read_yaml(path: Path, *, required: bool) -> dict[str, Any]:
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


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def maybe_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip() != "":
        return value
    return None


def parse_bool_string(value: str, *, field_name: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{field_name} must be a boolean string (true/false), got {value!r}.")


def coerce_wandb_log_model(value: str) -> bool | str:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return value


def extract_config_path_overrides(experiment_cfg: dict[str, Any]) -> dict[str, str]:
    overrides: dict[str, str] = {}

    candidates = [experiment_cfg]
    nested = experiment_cfg.get("experiment")
    if isinstance(nested, dict):
        candidates.append(nested)

    for cfg in candidates:
        trainer_path = maybe_str(cfg.get("trainer_config"))
        model_path = maybe_str(cfg.get("model_config"))
        data_path = maybe_str(cfg.get("data_config"))
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
            trainer_path = maybe_str(section.get("trainer"))
            model_path = maybe_str(section.get("model"))
            data_path = maybe_str(section.get("data"))
            if trainer_path is not None:
                overrides["trainer_config"] = trainer_path
            if model_path is not None:
                overrides["model_config"] = model_path
            if data_path is not None:
                overrides["data_config"] = data_path

    return overrides


def apply_wandb_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    trainer_cfg = require_mapping(config, "trainer")
    logger_cfg = require_mapping(trainer_cfg, "logger")

    if args.wandb_project is not None:
        logger_cfg["project"] = args.wandb_project
    if args.wandb_name is not None:
        logger_cfg["name"] = args.wandb_name
    if args.wandb_save_dir is not None:
        logger_cfg["save_dir"] = args.wandb_save_dir
    if args.wandb_offline is not None:
        logger_cfg["offline"] = parse_bool_string(
            args.wandb_offline,
            field_name="--wandb-offline",
        )
    if args.wandb_log_model is not None:
        logger_cfg["log_model"] = coerce_wandb_log_model(args.wandb_log_model)
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


def resolve_config_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    experiment_path = args.experiment_config.expanduser()
    if not experiment_path.is_absolute():
        experiment_path = (Path.cwd() / experiment_path).resolve()
    else:
        experiment_path = experiment_path.resolve()

    experiment_cfg = read_yaml(experiment_path, required=False)
    path_overrides = extract_config_path_overrides(experiment_cfg)

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


def load_merged_configs(args: argparse.Namespace) -> ResolvedConfigs:
    trainer_path, model_path, data_path, experiment_path = resolve_config_paths(args)

    trainer_cfg = read_yaml(trainer_path, required=True)
    model_cfg = read_yaml(model_path, required=True)
    data_cfg = read_yaml(data_path, required=True)
    experiment_cfg = read_yaml(experiment_path, required=False)

    merged: dict[str, Any] = {}
    deep_update(merged, trainer_cfg)
    deep_update(merged, model_cfg)
    deep_update(merged, data_cfg)
    deep_update(merged, experiment_cfg)

    nested_experiment = experiment_cfg.get("experiment")
    if isinstance(nested_experiment, dict):
        deep_update(merged, nested_experiment)

    return ResolvedConfigs(
        merged=merged,
        trainer_config_path=trainer_path,
        model_config_path=model_path,
        data_config_path=data_path,
        experiment_config_path=experiment_path,
    )


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required mapping '{key}' in merged config.")
    return value


def coerce_betas(value: Any) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("trainer.lightning_module.betas must be a list/tuple of length 2.")
    return float(value[0]), float(value[1])


def resolve_import_string(path: str, *, field_name: str) -> Any:
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


def build_audio_decoder(module_cfg: dict[str, Any]) -> Any | None:
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

    decoder_obj = resolve_import_string(
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


def optional_path(value: Any) -> str | None:
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


def shared_memory_total_bytes() -> int | None:
    try:
        return int(shutil.disk_usage("/dev/shm").total)
    except OSError:
        return None


def parse_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return parse_bool_string(raw, field_name=name)
    except ValueError:
        print(f"[train.py] Ignoring invalid {name}={raw!r}; using default={default}.")
        return default


def parse_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[train.py] Ignoring invalid {name}={raw!r}; using default={default}.")
        return default


def length_quantile(values: Sequence[int], quantile: float) -> int:
    if not values:
        raise ValueError("values must not be empty.")
    if quantile <= 0.0 or quantile > 1.0:
        raise ValueError(f"quantile must be in (0, 1], got {quantile}.")

    sorted_values = sorted(max(1, int(value)) for value in values)
    rank = max(0, min(len(sorted_values) - 1, int(math.ceil(quantile * len(sorted_values))) - 1))
    return int(sorted_values[rank])


def should_force_single_process_loader(num_workers: int) -> bool:
    if num_workers <= 0:
        return False

    # Allow explicit opt-out for environments where low /dev/shm is still acceptable.
    if parse_env_bool("PRISM_TTS_DISABLE_SHM_GUARD", False):
        return False

    total_bytes = shared_memory_total_bytes()
    if total_bytes is None:
        return False

    default_threshold = 512 * 1024 * 1024
    threshold = parse_env_int("PRISM_TTS_MIN_SHM_BYTES", default_threshold)
    return total_bytes < threshold


def filter_kwargs_for_callable(
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


def coerce_gpu_indices(value: Any) -> list[int]:
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

