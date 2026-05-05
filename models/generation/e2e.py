from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch

from utils import model_utils as MU

if TYPE_CHECKING:
    from models.prism_tts import PrismTTS


def generate_e2e(
    model: PrismTTS,
    raw_text_prompt: str | Sequence[str],
    raw_speech_prompt: Any | list[Any],
    raw_text_target: str | Sequence[str],
    *,
    text_tokenizer: Optional[Callable[[str], Sequence[int]]] = None,
    speech_encoder: Optional[Callable[[Any], tuple[torch.Tensor, torch.Tensor]]] = None,
    speech_decoder: Optional[Callable[[torch.Tensor], Any]] = None,
    output_type: str = "tensor",
    return_dict: bool = True,
    mimi_model_name_or_path: str = "kyutai/mimi",
    mimi_revision: str = "main",
    mimi_token: str | bool | None = None,
    mimi_local_files_only: bool = False,
    **generate_kwargs: Any,
) -> MU.PrismTTSGenerationOutput | tuple[torch.LongTensor, torch.FloatTensor] | torch.Tensor:
    """
    End-to-end generation from raw text/audio-like inputs.

    `text_tokenizer` converts raw string input to text token ids.
    `speech_encoder` converts one raw speech prompt object to:
    - discrete tokens with shape [L, N] or [N, L]
    - continuous latents with shape [L, D] (or [D, L], accepted and transposed)
    where N=num_discrete_tokens and D=continuous_latent_size.

    If `text_tokenizer` is None, initialize a `SharedVocabTokenizer` using
    the same default behavior as dataset construction.
    If `speech_encoder` and/or `speech_decoder` is None, initialize the
    missing component(s) from Mimi.

    `output_type` controls the return value:
    - "tensor": return token/latent tensors (same behavior as `generate`).
    - "speech": return decoded waveform tensor from `speech_decoder`.
    """
    output_type_normalized = str(output_type).strip().lower()
    if output_type_normalized not in ("tensor", "speech"):
        raise ValueError("output_type must be one of {'tensor', 'speech'}.")

    prompt_text_list = MU.normalize_raw_text_batch(raw_text_prompt, "raw_text_prompt")
    target_text_list = MU.normalize_raw_text_batch(raw_text_target, "raw_text_target")

    if isinstance(raw_speech_prompt, list):
        speech_prompt_list = list(raw_speech_prompt)
    else:
        speech_prompt_list = [raw_speech_prompt]

    batch_size = len(prompt_text_list)
    if len(target_text_list) != batch_size:
        raise ValueError("raw_text_target batch size must match raw_text_prompt.")
    if len(speech_prompt_list) != batch_size:
        raise ValueError("raw_speech_prompt batch size must match raw_text_prompt.")

    params = list(model.parameters())
    if len(params) == 0:
        raise RuntimeError("Model has no parameters.")
    device = params[0].device
    continuous_dtype = model.continuous_proj.weight.dtype

    if text_tokenizer is None:
        from dataset.dataset import SharedVocabTokenizer, build_shared_token_layout

        discrete_token_count = int(model.discrete_vocab_size) - 3
        if discrete_token_count < 1:
            raise ValueError(
                "Cannot infer tokenizer shared layout: discrete_vocab_size must be >= 4."
            )
        _, eos_token_id, _, text_token_offset = build_shared_token_layout(discrete_token_count)
        vocab_path = Path(__file__).resolve().parents[2] / "dataset" / "vocab.txt"
        text_tokenizer = SharedVocabTokenizer(
            vocab_path=vocab_path,
            text_token_offset=text_token_offset,
            eos_token_id=eos_token_id,
            append_eos=False,
        )

    needs_default_mimi_encoder = speech_encoder is None
    needs_default_mimi_decoder = speech_decoder is None
    if needs_default_mimi_encoder:
        speech_encoder = MU.build_default_mimi_speech_encoder(
            num_discrete_tokens=int(model.num_discrete_tokens),
            device=device,
            continuous_dtype=continuous_dtype,
            mimi_model_name_or_path=mimi_model_name_or_path,
            mimi_revision=mimi_revision,
            mimi_token=mimi_token,
            mimi_local_files_only=bool(mimi_local_files_only),
            raw_prompt_name="raw_speech_prompt",
        )

    if needs_default_mimi_decoder:
        speech_decoder = MU.build_lazy_mimi_speech_decoder(
            device=device,
            continuous_dtype=continuous_dtype,
            mimi_model_name_or_path=mimi_model_name_or_path,
            mimi_revision=mimi_revision,
            mimi_token=mimi_token,
            mimi_local_files_only=bool(mimi_local_files_only),
        )

    prompt_token_lists: list[list[int]] = []
    target_token_lists: list[list[int]] = []
    speech_discrete_list: list[torch.LongTensor] = []
    speech_continuous_list: list[torch.FloatTensor] = []

    for sample_idx in range(batch_size):
        prompt_tokens = [int(token) for token in text_tokenizer(prompt_text_list[sample_idx])]
        target_tokens = [int(token) for token in text_tokenizer(target_text_list[sample_idx])]
        prompt_token_lists.append(prompt_tokens)
        target_token_lists.append(target_tokens)

        encoded = speech_encoder(speech_prompt_list[sample_idx])
        if not isinstance(encoded, tuple) or len(encoded) != 2:
            raise ValueError(
                "speech_encoder must return a tuple of "
                "(discrete_tokens, continuous_latents)."
            )
        discrete_tokens, continuous_latents = encoded

        discrete = torch.as_tensor(discrete_tokens, device=device, dtype=torch.long)
        if discrete.dim() == 3 and discrete.shape[0] == 1:
            discrete = discrete.squeeze(0)
        if discrete.dim() != 2:
            raise ValueError(
                "Encoded discrete prompt must have shape [L, N] or [N, L]."
            )
        if discrete.shape[1] == model.num_discrete_tokens:
            discrete = discrete.contiguous()
        elif discrete.shape[0] == model.num_discrete_tokens:
            discrete = discrete.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                "Encoded discrete prompt must contain one axis with "
                f"num_discrete_tokens={model.num_discrete_tokens}, got {tuple(discrete.shape)}."
            )

        continuous = torch.as_tensor(
            continuous_latents,
            device=device,
            dtype=continuous_dtype,
        )
        if continuous.dim() == 3 and continuous.shape[0] == 1:
            continuous = continuous.squeeze(0)
        if continuous.dim() == 1:
            if model.continuous_latent_size != 1:
                raise ValueError(
                    "Encoded continuous prompt with shape [L] is only valid when "
                    "continuous_latent_size == 1."
                )
            continuous = continuous.unsqueeze(-1)
        if continuous.dim() != 2:
            raise ValueError(
                "Encoded continuous prompt must have shape [L, D] or [D, L]."
            )
        if (
            continuous.shape[0] == model.continuous_latent_size
            and continuous.shape[1] == discrete.shape[0]
        ):
            continuous = continuous.transpose(0, 1).contiguous()
        if continuous.shape[0] != discrete.shape[0]:
            raise ValueError(
                "Speech prompt length mismatch between discrete and continuous tensors: "
                f"{discrete.shape[0]} vs {continuous.shape[0]}."
            )
        if continuous.shape[1] != model.continuous_latent_size:
            raise ValueError(
                "Encoded continuous prompt channel mismatch: expected "
                f"{model.continuous_latent_size}, got {continuous.shape[1]}."
            )

        speech_discrete_list.append(discrete)
        speech_continuous_list.append(continuous)

    max_prompt_text = max(len(tokens) for tokens in prompt_token_lists) if batch_size > 0 else 0
    max_target_text = max(len(tokens) for tokens in target_token_lists) if batch_size > 0 else 0
    max_speech_prompt = max(
        int(discrete.shape[0]) for discrete in speech_discrete_list
    ) if batch_size > 0 else 0

    text_prompt = torch.full(
        (batch_size, max_prompt_text),
        model.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    text_target = torch.full(
        (batch_size, max_target_text),
        model.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    discrete_prompt = torch.full(
        (batch_size, model.num_discrete_tokens, max_speech_prompt),
        model.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    continuous_prompt = torch.zeros(
        (batch_size, max_speech_prompt, model.continuous_latent_size),
        dtype=continuous_dtype,
        device=device,
    )

    text_prompt_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    speech_prompt_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    text_target_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

    for sample_idx in range(batch_size):
        prompt_tokens = prompt_token_lists[sample_idx]
        target_tokens = target_token_lists[sample_idx]
        speech_discrete = speech_discrete_list[sample_idx]
        speech_continuous = speech_continuous_list[sample_idx]

        text_prompt_lengths[sample_idx] = int(len(prompt_tokens))
        speech_prompt_lengths[sample_idx] = int(speech_discrete.shape[0])
        text_target_lengths[sample_idx] = int(len(target_tokens))

        if len(prompt_tokens) > 0:
            text_prompt[sample_idx, : len(prompt_tokens)] = torch.tensor(
                prompt_tokens,
                dtype=torch.long,
                device=device,
            )
        if len(target_tokens) > 0:
            text_target[sample_idx, : len(target_tokens)] = torch.tensor(
                target_tokens,
                dtype=torch.long,
                device=device,
            )
        if speech_discrete.shape[0] > 0:
            discrete_prompt[sample_idx, :, : speech_discrete.shape[0]] = speech_discrete.transpose(
                0, 1
            )
            continuous_prompt[sample_idx, : speech_continuous.shape[0], :] = speech_continuous

    if "text_prompt_lengths" in generate_kwargs:
        raise ValueError("Do not pass text_prompt_lengths when using generate_e2e.")
    if "speech_prompt_lengths" in generate_kwargs:
        raise ValueError("Do not pass speech_prompt_lengths when using generate_e2e.")
    if "text_target_lengths" in generate_kwargs:
        raise ValueError("Do not pass text_target_lengths when using generate_e2e.")
    if "return_dict" in generate_kwargs:
        raise ValueError("Use generate_e2e(return_dict=...) instead of return_dict in kwargs.")

    generation = model.generate(
        text_prompt=text_prompt,
        discrete_prompt=discrete_prompt,
        continuous_prompt=continuous_prompt,
        text_target=text_target,
        text_prompt_lengths=text_prompt_lengths,
        speech_prompt_lengths=speech_prompt_lengths,
        text_target_lengths=text_target_lengths,
        return_dict=True,
        **generate_kwargs,
    )

    if output_type_normalized == "tensor":
        if return_dict:
            return generation
        if generation.discrete_ids is None or generation.continuous_latents is None:
            raise RuntimeError("generate_e2e(tensor) requires generated discrete and continuous outputs.")
        return generation.discrete_ids, generation.continuous_latents

    if speech_decoder is None:
        raise RuntimeError("speech_decoder is required for output_type='speech'.")
    if generation.continuous_latents is None:
        raise RuntimeError("Generation produced no continuous latents to decode.")

    decoded_speech = speech_decoder(generation.continuous_latents)
    if torch.is_tensor(decoded_speech):
        speech_tensor = decoded_speech
    else:
        speech_tensor = torch.as_tensor(decoded_speech)
    if speech_tensor.dim() == 3 and speech_tensor.shape[1] == 1:
        speech_tensor = speech_tensor[:, 0, :]
    if speech_tensor.dim() == 1:
        speech_tensor = speech_tensor.unsqueeze(0)
    return speech_tensor
