from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train


def _tracking_config(*, experiment_name: str) -> dict:
    return {
        "experiment": {"name": experiment_name},
        "trainer": {
            "logger": {"type": "csv", "name": None, "save_dir": "logs"},
            "checkpoint": {"dirpath": "checkpoints"},
        },
    }


def test_experiment_name_scopes_default_logger_and_checkpoint_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _tracking_config(experiment_name="meanflow / ablation 1")

    tracking = train._apply_experiment_tracking_defaults(config)

    assert tracking.name == "meanflow / ablation 1"
    assert tracking.artifact_name == "meanflow-ablation-1"
    assert config["trainer"]["logger"]["name"] == "meanflow-ablation-1"
    assert config["trainer"]["checkpoint"]["dirpath"] == "checkpoints/meanflow-ablation-1"
    assert tracking.checkpoint_dir == tmp_path / "checkpoints" / "meanflow-ablation-1"


def test_experiment_tracking_preserves_explicit_logger_and_checkpoint_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _tracking_config(experiment_name="meanflow")
    config["trainer"]["logger"]["name"] = "comparison-run"
    config["trainer"]["checkpoint"]["dirpath"] = "checkpoints/comparison"

    tracking = train._apply_experiment_tracking_defaults(config)

    assert tracking.artifact_name == "meanflow"
    assert config["trainer"]["logger"]["name"] == "comparison-run"
    assert config["trainer"]["checkpoint"]["dirpath"] == "checkpoints/comparison"
    assert tracking.checkpoint_dir == tmp_path / "checkpoints" / "comparison"


def test_model_checkpoint_saves_same_stem_resolved_config(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "prism_tts-step=0000100.ckpt"
    config = {"experiment": {"name": "meanflow"}, "trainer": {"max_steps": 100}}

    def fake_save_checkpoint(self, trainer, filepath):
        del self, trainer
        Path(filepath).write_bytes(b"checkpoint")

    monkeypatch.setattr(train.ModelCheckpoint, "_save_checkpoint", fake_save_checkpoint)
    callback = train.ConfigSnapshotModelCheckpoint(
        dirpath=tmp_path,
        config_snapshot=config,
    )
    callback._save_checkpoint(SimpleNamespace(is_global_zero=True), str(checkpoint_path))

    assert checkpoint_path.read_bytes() == b"checkpoint"
    assert train._checkpoint_config_path(checkpoint_path) == checkpoint_path.with_suffix(".yaml")
    assert yaml.safe_load(checkpoint_path.with_suffix(".yaml").read_text()) == config


def test_validation_stage_checkpoint_saves_same_stem_resolved_config(tmp_path):
    saved_paths: list[Path] = []

    class FakeTrainer:
        sanity_checking = False
        is_global_zero = True
        global_step = 100
        current_epoch = 2

        def save_checkpoint(self, path, weights_only):
            assert not weights_only
            checkpoint_path = Path(path)
            checkpoint_path.write_bytes(b"checkpoint")
            saved_paths.append(checkpoint_path)

    config = {"experiment": {"name": "meanflow"}, "model": {"name": "prism_tts"}}
    callback = train.SaveEveryValidationStageCheckpoint(
        dirpath=tmp_path,
        filename="prism_tts-val-stage={val_stage:05d}-step={step:07d}",
        save_weights_only=False,
        config_snapshot=config,
    )
    callback.on_validation_end(FakeTrainer(), None)

    assert len(saved_paths) == 1
    assert yaml.safe_load(saved_paths[0].with_suffix(".yaml").read_text()) == config


def test_train_model_builder_honors_meanflow_and_memory_options():
    config = {
        "model": {
            "name": "prism_tts",
            "prism_tts": {
                "continuous_latent_size": 4,
                "flow_num_res_blocks": 1,
                "head_mode": "meanflow",
                "attn_mode": "bidirectional",
                "use_short_context": True,
                "short_context_layers": 1,
                "short_context_window": 2,
                "short_context_chunk_size": 3,
                "gradient_checkpointing": True,
            },
            "llama_config": {
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "max_position_embeddings": 32,
                "pad_token_id": 0,
                "eos_token_id": 2,
                "use_cache": False,
                "_attn_implementation": "eager",
            },
        }
    }

    model = train._build_model(config)

    assert model.head_mode == "meanflow"
    assert model.gradient_checkpointing
    assert model.backbone.gradient_checkpointing
    assert model.short_encoder.gradient_checkpointing
    assert model.short_encoder.chunk_size == 3


def test_lightning_module_uses_low_peak_optimizer_and_cpu_ema():
    config = {
        "trainer": {
            "lightning_module": {
                "audio_decoder": None,
                "ema_device": "cpu",
            },
            "optimizer": {
                "name": "adamw",
                "foreach": False,
            },
            "scheduler": {"enabled": False},
            "lightning_trainer": {"max_steps": 1},
        }
    }
    model_config = {
        "model": {
            "name": "prism_tts",
            "prism_tts": {
                "continuous_latent_size": 4,
                "flow_num_res_blocks": 1,
                "use_short_context": False,
            },
            "llama_config": {
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "max_position_embeddings": 32,
                "pad_token_id": 0,
                "eos_token_id": 2,
                "use_cache": False,
                "_attn_implementation": "eager",
            },
        }
    }
    model = train._build_model(model_config)
    module = train._build_lightning_module(config, model)

    optimizer = module.configure_optimizers()
    module._initialize_ema_if_needed()

    assert optimizer.defaults["foreach"] is False
    assert module.ema_device == "cpu"
    assert module._ema_state
    assert all(tensor.device.type == "cpu" for tensor in module._ema_state.values())


def test_audio_decoder_is_lazy_by_default(tmp_path, monkeypatch):
    module_path = tmp_path / "lazy_decoder_target.py"
    module_path.write_text(
        "\n".join(
            [
                "CALLS = []",
                "",
                "class TrackingDecoder:",
                "    sample_rate = 12345",
                "",
                "    def __init__(self, marker):",
                "        CALLS.append(('init', marker))",
                "",
                "    def __call__(self, latents):",
                "        CALLS.append(('call', tuple(latents.shape)))",
                "        return latents",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    decoder = train._build_audio_decoder(
        {
            "audio_decoder": "lazy_decoder_target.TrackingDecoder",
            "audio_decoder_kwargs": {"marker": "ok"},
        }
    )

    assert isinstance(decoder, train.LazyAudioDecoder)
    assert "lazy_decoder_target" not in sys.modules
    assert decoder.sample_rate is None

    latents = torch.zeros(2, 3)
    assert torch.equal(decoder(latents), latents)

    imported = sys.modules["lazy_decoder_target"]
    assert imported.CALLS == [("init", "ok"), ("call", (2, 3))]
    assert decoder.sample_rate == 12345


def test_worker_warmup_reuses_prefetched_iterator_once(monkeypatch):
    def fake_base_iterator(loader):
        loader.base_iterator_calls = getattr(loader, "base_iterator_calls", 0) + 1
        loader._iterator = iter((object(),))
        return loader._iterator

    monkeypatch.setattr(train.DataLoader, "__iter__", fake_base_iterator)
    loader = train.StartupPrefetchDataLoader(
        [1],
        batch_size=1,
        num_workers=1,
        persistent_workers=True,
    )
    assert train._warmup_persistent_loader_workers(loader)
    primed_iterator = loader._iterator
    assert loader.base_iterator_calls == 1

    # Lightning's first iter(loader) gets the worker-prefetched batches instead
    # of resetting the iterator and re-reading them.
    assert iter(loader) is primed_iterator
    assert loader.base_iterator_calls == 1

    # Later epochs use the standard persistent-worker reset path again.
    assert iter(loader) is not primed_iterator
    assert loader.base_iterator_calls == 2


def test_lazy_metadata_build_does_not_open_manifest_during_data_setup(tmp_path):
    missing_manifest = tmp_path / "not-opened-until-training.txt"
    config = {
        "data": {
            "train_manifest": str(missing_manifest),
            "val_manifest": None,
            "test_manifest": None,
            "manifest_root": None,
            "vocab_path": str(PROJECT_ROOT / "dataset" / "vocab.txt"),
            "loader": {
                "train_batch_size": 1,
                "num_workers": 0,
                "persistent_workers": False,
                "pin_memory": False,
                "shuffle_train": True,
            },
            "shared_layout": {"discrete_token_count": 16},
            "dataset": {"metadata_mode": "lazy"},
            "collate": {"continuous_pad_value": 0.0, "include_attention_mask": True},
        },
        "trainer": {"distributed": {"enabled": False}},
    }

    train_loader, val_loader, test_loader = train._build_data_objects(
        config,
        warmup_train_workers=False,
    )

    assert isinstance(train_loader.dataset, train.LazyPrismDataset)
    assert val_loader is None
    assert test_loader is None


def test_indexed_metadata_builds_sized_map_dataset_for_distributed_sampler(tmp_path):
    manifest = tmp_path / "train.txt"
    manifest.write_text(
        "\n".join(
            f"{index}.wav|1.0|a|{index}.npy|p.wav|1.0|a|p.npy"
            for index in range(5)
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "data": {
            "train_manifest": str(manifest),
            "val_manifest": None,
            "test_manifest": None,
            "manifest_root": None,
            "vocab_path": str(PROJECT_ROOT / "dataset" / "vocab.txt"),
            "loader": {
                "train_batch_size": 2,
                "num_workers": 0,
                "persistent_workers": False,
                "pin_memory": False,
                "shuffle_train": True,
            },
            "shared_layout": {"discrete_token_count": 16},
            "dataset": {"metadata_mode": "indexed"},
            "collate": {"continuous_pad_value": 0.0, "include_attention_mask": True},
        },
        "trainer": {"distributed": {"enabled": True}},
    }

    train_loader, _, _ = train._build_data_objects(
        config,
        warmup_train_workers=False,
    )

    assert isinstance(train_loader.dataset, train.PrismDataset)
    assert not isinstance(train_loader.dataset, train.IterableDataset)
    assert train_loader.dataset.index_manifest
    assert len(train_loader.dataset) == 5

    rank_zero = torch.utils.data.DistributedSampler(
        train_loader.dataset,
        num_replicas=2,
        rank=0,
        shuffle=False,
    )
    rank_one = torch.utils.data.DistributedSampler(
        train_loader.dataset,
        num_replicas=2,
        rank=1,
        shuffle=False,
    )
    assert len(rank_zero) == len(rank_one) == 3
    assert set(rank_zero).union(rank_one) == set(range(5))


@pytest.mark.filterwarnings("ignore:GPU available but not used.*")
def test_lightning_consumes_startup_prefetch_without_resetting_workers(monkeypatch):
    class TinyModule(train.pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def training_step(self, batch, batch_idx):
            del batch_idx
            return self.linear(batch).sum()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    def fake_base_iterator(loader):
        loader.base_iterator_calls = getattr(loader, "base_iterator_calls", 0) + 1
        loader._iterator = iter((torch.ones(1, 1),))
        return loader._iterator

    monkeypatch.setattr(train.DataLoader, "__iter__", fake_base_iterator)
    loader = train.StartupPrefetchDataLoader(
        [1],
        batch_size=1,
        num_workers=9,
        persistent_workers=True,
    )
    assert train._warmup_persistent_loader_workers(loader)

    trainer = train.pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(TinyModule(), train_dataloaders=loader)

    assert loader.base_iterator_calls == 1
