from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from utils import model_utils as MU

if TYPE_CHECKING:
    from models.prism_tts import PrismTTS


def generate_causal(
    model: PrismTTS,
    *,
    text_prompt: torch.LongTensor,
    discrete_prompt: torch.LongTensor,
    continuous_prompt: torch.FloatTensor,
    text_target: torch.LongTensor,
    text_prompt_lengths: torch.LongTensor,
    speech_prompt_lengths: torch.LongTensor,
    text_target_lengths: torch.LongTensor,
    speech_target_lengths: torch.LongTensor,
    discrete_eos_id: int,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    flow_num_steps: Optional[int],
    parallel_num_steps: int,
    special_discrete_token_ids: tuple[int, ...],
    active_discrete_streams: int,
) -> tuple[
    torch.LongTensor,
    torch.FloatTensor,
    torch.FloatTensor,
    torch.LongTensor,
    list[torch.Tensor],
]:
    """Causal left-to-right generation with a fixed terminal EOS speech block."""
    del parallel_num_steps
    batch_size = int(text_prompt.shape[0])
    device = text_prompt.device
    num_active_streams = int(active_discrete_streams)
    max_target = int(speech_target_lengths.max().item()) if batch_size > 0 else 0
    safe_discrete_fill_id = (
        int(discrete_eos_id)
        if 0 <= int(discrete_eos_id) < model.discrete_vocab_size
        else model.pad_token_id
    )
    terminal_discrete_id = safe_discrete_fill_id

    predicted_discrete = discrete_prompt.new_full(
        (batch_size, max_target, num_active_streams),
        safe_discrete_fill_id,
    )
    predicted_continuous = continuous_prompt.new_zeros(
        (batch_size, max_target, model.continuous_latent_size)
    )
    predicted_prior = continuous_prompt.new_zeros(
        (batch_size, max_target, model.continuous_latent_size)
    )
    generated_lengths = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=device,
    )
    collected_logits: list[torch.Tensor] = []
    if max_target <= 0:
        return (
            predicted_discrete,
            predicted_continuous,
            predicted_prior,
            generated_lengths,
            collected_logits,
        )

    valid_target_mask = (
        torch.arange(max_target, device=device).unsqueeze(0)
        < speech_target_lengths.unsqueeze(1)
    )
    maskable_target_mask = valid_target_mask.clone()
    has_target = speech_target_lengths > 0
    if has_target.any():
        eos_sample_idx = torch.nonzero(has_target, as_tuple=False).squeeze(1)
        eos_block_idx = speech_target_lengths[eos_sample_idx] - 1
        predicted_discrete[eos_sample_idx, eos_block_idx, :] = terminal_discrete_id
        predicted_continuous[eos_sample_idx, eos_block_idx, :] = 0.0
        maskable_target_mask[eos_sample_idx, eos_block_idx] = False

    step_indices = torch.arange(max_target, device=device).view(max_target, 1, 1)
    block_indices = torch.arange(max_target, device=device).view(1, 1, max_target)
    left_to_right_masks = block_indices >= step_indices
    mask_schedule = maskable_target_mask.unsqueeze(0) & left_to_right_masks

    maskable_lengths = torch.clamp(speech_target_lengths - 1, min=0)
    finished = maskable_lengths <= 0
    for step_idx in range(max_target):
        masked_blocks = mask_schedule[step_idx] & (~finished).unsqueeze(1)
        if not bool(masked_blocks.any().item()):
            break

        current_discrete = predicted_discrete.clone()
        current_continuous = predicted_continuous.clone()
        current_discrete[masked_blocks] = model.pad_token_id
        current_continuous[masked_blocks] = 0.0

        flat = MU.assemble_flat_batch(
            text_prompt=text_prompt,
            discrete_prompt=discrete_prompt,
            continuous_prompt=continuous_prompt,
            text_target=text_target,
            discrete_target=current_discrete,
            continuous_target=current_continuous,
            text_prompt_lengths=text_prompt_lengths,
            speech_prompt_lengths=speech_prompt_lengths,
            text_target_lengths=text_target_lengths,
            speech_target_lengths=speech_target_lengths,
            attention_mask=None,
            pad_token_id=model.pad_token_id,
            eos_token_id=model.eos_token_id,
            eot_token_id=model.eot_token_id,
            continuous_latent_size=model.continuous_latent_size,
            num_discrete_tokens=num_active_streams,
            continuous_stream_id=model.num_discrete_tokens,
        )
        hidden_states, masked_discrete_positions, masked_continuous_positions, _ = model._encode(
            flat=flat,
            masked_target_blocks=masked_blocks,
        )
        batch_indices = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, hidden_states.shape[1])
        )
        step_target_mask = flat.target_block_ids == step_idx

        step_logits = torch.full(
            (batch_size, num_active_streams, model.discrete_vocab_size),
            fill_value=float("-inf"),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        active_step_samples = (~finished) & (maskable_lengths > step_idx)

        step_discrete_positions = masked_discrete_positions & step_target_mask
        if step_discrete_positions.any():
            discrete_hidden = hidden_states[step_discrete_positions]
            discrete_stream_idx = flat.speech_stream_ids[step_discrete_positions]
            discrete_logits = model._project_discrete_logits(
                hidden_states=discrete_hidden,
                discrete_stream_ids=discrete_stream_idx,
            )
            sampled_discrete = model._sample_discrete_ids(
                discrete_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            discrete_batch_idx = batch_indices[step_discrete_positions]
            predicted_discrete[discrete_batch_idx, step_idx, discrete_stream_idx] = sampled_discrete
            step_logits[discrete_batch_idx, discrete_stream_idx, :] = discrete_logits

        step_continuous_positions = masked_continuous_positions & step_target_mask
        if step_continuous_positions.any():
            continuous_hidden = hidden_states[step_continuous_positions]
            prior_prediction = model.continuous_prior_head(continuous_hidden)
            continuous_batch_idx = batch_indices[step_continuous_positions]
            predicted_prior[continuous_batch_idx, step_idx, :] = prior_prediction
            denoise_step_mask = torch.zeros(
                (batch_size, step_idx + 1),
                dtype=torch.bool,
                device=device,
            )
            denoise_step_mask[continuous_batch_idx, step_idx] = True
            denoised_step_context = model._sample_continuous_with_clean_context(
                prior_target=predicted_prior[:, : step_idx + 1, :],
                clean_target=predicted_continuous[:, : step_idx + 1, :],
                denoise_target_mask=denoise_step_mask,
                valid_target_mask=valid_target_mask[:, : step_idx + 1],
                prompt_latents=continuous_prompt,
                prompt_lengths=speech_prompt_lengths,
                num_steps=flow_num_steps,
                temperature=temperature,
            )
            predicted_continuous[continuous_batch_idx, step_idx, :] = denoised_step_context[
                continuous_batch_idx,
                step_idx,
                :,
            ]

            if len(special_discrete_token_ids) > 0:
                step_discrete = predicted_discrete[continuous_batch_idx, step_idx, :]
                special_step_mask = MU.build_special_block_mask(
                    discrete_tokens=step_discrete,
                    special_token_ids=special_discrete_token_ids,
                )
                if special_step_mask.any():
                    predicted_continuous[
                        continuous_batch_idx[special_step_mask],
                        step_idx,
                        :,
                    ] = 0.0

        active_sample_idx = torch.nonzero(active_step_samples, as_tuple=False).squeeze(1)
        if active_sample_idx.numel() > 0:
            generated_lengths[active_sample_idx] = step_idx + 1
            finished[active_sample_idx] = (
                generated_lengths[active_sample_idx] >= maskable_lengths[active_sample_idx]
            )
        collected_logits.append(step_logits)

    generated_lengths = speech_target_lengths.clone()
    return (
        predicted_discrete,
        predicted_continuous,
        predicted_prior,
        generated_lengths,
        collected_logits,
    )
