from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.metadata_utils import (  # noqa: E402
    infer_split_from_metadata_path,
    load_metadata,
    parse_metadata_line,
    resolve_librispeech_audio_path,
)


def test_load_metadata_preserves_target_and_prompt_fields(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata-test-clean.txt"
    metadata_path.write_text(
        "# target_id|target_text|prompt_id|prompt_text\n"
        "1089-134686-0000|target words|1089-134686-0009|prompt words\n"
        "1089-134686-0001|next target|1089-134691-0009|\n",
        encoding="utf-8",
    )

    entries = load_metadata(metadata_path)

    assert [(entry.index, entry.line_number) for entry in entries] == [(0, 2), (1, 3)]
    assert entries[0].target_file_id == "1089-134686-0000"
    assert entries[0].target_transcript == "target words"
    assert entries[0].prompt_file_id == "1089-134686-0009"
    assert entries[1].prompt_transcript == ""


def test_resolve_librispeech_prompt_path_and_infer_split(tmp_path: Path) -> None:
    path = resolve_librispeech_audio_path(
        audio_root=tmp_path / "LibriSpeech",
        split=infer_split_from_metadata_path(tmp_path / "metadata-test-clean.txt"),
        file_id="1089-134686-0009",
        extension="flac",
        line_number=1,
    )

    assert path == tmp_path / "LibriSpeech/test-clean/1089/134686/1089-134686-0009.flac"
    assert infer_split_from_metadata_path(tmp_path / "pairs.txt") is None


def test_parse_metadata_line_rejects_invalid_row() -> None:
    with pytest.raises(ValueError, match="expected 4 fields"):
        parse_metadata_line("only|three|fields\n", line_number=7, index=0)

    with pytest.raises(ValueError, match="LibriSpeech utterance ID"):
        parse_metadata_line(
            "not-an-id|target|1089-134686-0009|prompt\n",
            line_number=8,
            index=0,
        )
