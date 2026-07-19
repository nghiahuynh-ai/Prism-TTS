from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train


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
