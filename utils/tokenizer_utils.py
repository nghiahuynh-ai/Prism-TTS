from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from utils import backbone_utils as BU

__all__ = [
    "build_generation_text_tokenizer",
    "infer_shared_discrete_token_count",
    "resolve_default_text_tokenizer",
    "tokenize_text_with_any_tokenizer",
]


def tokenize_text_with_any_tokenizer(
    tokenizer: Callable[[str], Sequence[int]] | Any,
    text: str,
) -> list[int]:
    from dataset.dataset import tokenize_with_external_tokenizer

    token_ids = tokenize_with_external_tokenizer(tokenizer, text)
    return [int(token_id) for token_id in token_ids]


def infer_shared_discrete_token_count(
    *,
    discrete_vocab_size: int,
    eot_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> int:
    if (
        int(eos_token_id) == int(eot_token_id) + 1
        and int(pad_token_id) == int(eos_token_id) + 1
        and int(eot_token_id) >= 1
    ):
        return int(eot_token_id)
    discrete_token_count = int(discrete_vocab_size) - 3
    if discrete_token_count < 1:
        raise ValueError(
            "Cannot infer tokenizer shared layout: discrete_vocab_size must be >= 4."
        )
    return discrete_token_count


def _resolve_vocab_path(path_like: str | Path) -> Path:
    resolved = Path(path_like).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    else:
        resolved = resolved.resolve()
    return resolved


def resolve_default_text_tokenizer(
    *,
    cached_tokenizer: Any | None,
    use_separate_codec_embedding: bool,
    backbone_name: Any,
    backbone_hf_checkpoint: Optional[str],
    backbone_hf_kwargs: Optional[Mapping[str, Any]],
    discrete_vocab_size: int,
    eot_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    default_shared_vocab_path: str | Path | None = None,
) -> Any:
    if cached_tokenizer is not None:
        return cached_tokenizer

    if use_separate_codec_embedding:
        return BU.build_backbone_text_tokenizer(
            backbone_name=backbone_name,
            backbone_hf_checkpoint=backbone_hf_checkpoint,
            backbone_hf_kwargs=backbone_hf_kwargs,
            require_checkpoint=True,
        )

    from dataset.dataset import SharedVocabTokenizer, build_shared_token_layout

    discrete_token_count = infer_shared_discrete_token_count(
        discrete_vocab_size=discrete_vocab_size,
        eot_token_id=eot_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    _, resolved_eos_token_id, _, text_token_offset = build_shared_token_layout(discrete_token_count)

    if default_shared_vocab_path is None:
        resolved_vocab_path = Path(__file__).resolve().parents[1] / "dataset" / "vocab.txt"
    else:
        resolved_vocab_path = _resolve_vocab_path(default_shared_vocab_path)
    return SharedVocabTokenizer(
        vocab_path=resolved_vocab_path,
        text_token_offset=text_token_offset,
        eos_token_id=resolved_eos_token_id,
        append_eos=False,
    )


def build_generation_text_tokenizer(
    *,
    cached_default_text_tokenizer: Any | None,
    use_separate_codec_embedding: bool,
    backbone_name: Any,
    backbone_hf_checkpoint: Optional[str],
    backbone_hf_kwargs: Optional[Mapping[str, Any]],
    discrete_vocab_size: int,
    eot_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    append_eos_to_text: bool = False,
    vocab_path: str | Path | None = None,
    default_shared_vocab_path: str | Path | None = None,
) -> tuple[Callable[[str], Sequence[int]], Any]:
    from dataset.dataset import BackboneTextTokenizer, SharedVocabTokenizer, build_shared_token_layout

    if use_separate_codec_embedding:
        default_tokenizer = resolve_default_text_tokenizer(
            cached_tokenizer=cached_default_text_tokenizer,
            use_separate_codec_embedding=use_separate_codec_embedding,
            backbone_name=backbone_name,
            backbone_hf_checkpoint=backbone_hf_checkpoint,
            backbone_hf_kwargs=backbone_hf_kwargs,
            discrete_vocab_size=discrete_vocab_size,
            eot_token_id=eot_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            default_shared_vocab_path=default_shared_vocab_path,
        )
        return (
            BackboneTextTokenizer(
                tokenizer=default_tokenizer,
                append_eos=append_eos_to_text,
            ),
            default_tokenizer,
        )

    discrete_token_count = infer_shared_discrete_token_count(
        discrete_vocab_size=discrete_vocab_size,
        eot_token_id=eot_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    _, resolved_eos_token_id, _, text_token_offset = build_shared_token_layout(discrete_token_count)

    cached_tokenizer = cached_default_text_tokenizer
    if vocab_path is None and not append_eos_to_text:
        default_tokenizer = resolve_default_text_tokenizer(
            cached_tokenizer=cached_default_text_tokenizer,
            use_separate_codec_embedding=use_separate_codec_embedding,
            backbone_name=backbone_name,
            backbone_hf_checkpoint=backbone_hf_checkpoint,
            backbone_hf_kwargs=backbone_hf_kwargs,
            discrete_vocab_size=discrete_vocab_size,
            eot_token_id=eot_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            default_shared_vocab_path=default_shared_vocab_path,
        )
        cached_tokenizer = default_tokenizer
        if isinstance(default_tokenizer, SharedVocabTokenizer):
            return default_tokenizer, default_tokenizer

    if vocab_path is None:
        if default_shared_vocab_path is None:
            resolved_vocab_path = Path(__file__).resolve().parents[1] / "dataset" / "vocab.txt"
        else:
            resolved_vocab_path = _resolve_vocab_path(default_shared_vocab_path)
    else:
        resolved_vocab_path = _resolve_vocab_path(vocab_path)

    return (
        SharedVocabTokenizer(
            vocab_path=resolved_vocab_path,
            text_token_offset=text_token_offset,
            eos_token_id=resolved_eos_token_id,
            append_eos=append_eos_to_text,
        ),
        cached_tokenizer,
    )
