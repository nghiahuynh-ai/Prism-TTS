from __future__ import annotations

import sys
from pathlib import Path

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
