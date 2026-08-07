"""Dependency-free parsing and path resolution for LibriSpeech pair metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "MetadataEntry",
    "infer_split_from_metadata_path",
    "load_metadata",
    "parse_metadata_line",
    "resolve_librispeech_audio_path",
]


_SPLIT_FROM_METADATA_NAME = re.compile(r"^metadata-(.+)\.txt$", re.IGNORECASE)


@dataclass(frozen=True)
class MetadataEntry:
    """One target/prompt pair from a four-column metadata row."""

    index: int
    line_number: int
    target_file_id: str
    target_transcript: str
    prompt_file_id: str
    prompt_transcript: str


def _resolve_path(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def _split_librispeech_id(file_id: str, *, field_name: str, line_number: int) -> tuple[str, str]:
    """Return the speaker and chapter portions of a LibriSpeech utterance ID."""
    parts = file_id.split("-")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError(
            f"Line {line_number}: {field_name} must be a LibriSpeech utterance ID "
            f"(<speaker>-<chapter>-<utterance>), got {file_id!r}."
        )
    return parts[0], parts[1]


def parse_metadata_line(raw_line: str, *, line_number: int, index: int) -> MetadataEntry | None:
    """Parse one four-column row, ignoring blank lines and comments."""
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    fields = [field.strip() for field in raw_line.rstrip("\r\n").split("|")]
    if len(fields) != 4:
        raise ValueError(
            f"Line {line_number}: expected 4 fields separated by '|', found {len(fields)}."
        )

    target_file_id, target_transcript, prompt_file_id, prompt_transcript = fields
    if not target_file_id or not prompt_file_id:
        raise ValueError(f"Line {line_number}: target and prompt file IDs must not be empty.")
    if not target_transcript:
        raise ValueError(f"Line {line_number}: target transcript must not be empty.")
    _split_librispeech_id(target_file_id, field_name="target_file_id", line_number=line_number)
    _split_librispeech_id(prompt_file_id, field_name="prompt_file_id", line_number=line_number)
    return MetadataEntry(
        index=index,
        line_number=line_number,
        target_file_id=target_file_id,
        target_transcript=target_transcript,
        prompt_file_id=prompt_file_id,
        prompt_transcript=prompt_transcript,
    )


def load_metadata(path: Path) -> list[MetadataEntry]:
    """Read all valid rows from ``path`` and retain their source line numbers."""
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Metadata file not found: {resolved}")

    entries: list[MetadataEntry] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            entry = parse_metadata_line(raw_line, line_number=line_number, index=len(entries))
            if entry is not None:
                entries.append(entry)
    if not entries:
        raise ValueError(f"No valid metadata rows found in {resolved}")
    return entries


def infer_split_from_metadata_path(metadata_path: Path) -> str | None:
    """Infer ``test-clean`` from a filename such as ``metadata-test-clean.txt``."""
    match = _SPLIT_FROM_METADATA_NAME.match(metadata_path.name)
    return match.group(1) if match else None


def resolve_librispeech_audio_path(
    *,
    audio_root: Path,
    split: str | None,
    file_id: str,
    extension: str,
    line_number: int,
) -> Path:
    """Resolve a LibriSpeech utterance ID to its audio file path."""
    speaker_id, chapter_id = _split_librispeech_id(
        file_id,
        field_name="file_id",
        line_number=line_number,
    )
    suffix = extension if extension.startswith(".") else f".{extension}"
    base = audio_root / split if split else audio_root
    return base / speaker_id / chapter_id / f"{file_id}{suffix}"
