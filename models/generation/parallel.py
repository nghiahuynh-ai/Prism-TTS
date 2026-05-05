from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from utils import model_utils as MU

if TYPE_CHECKING:
    from models.prism_tts import PrismTTS


def generate_parallel(
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
    """
    Block-parallel generation with confidence remasking and fixed terminal EOS block.

    We decode masked non-terminal blocks in parallel, score each block by
    discrete-token confidence, and remask only low-confidence blocks according
    to a cosine schedule.
    """
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
    if max_target <= 0:
        return (
            predicted_discrete,
            predicted_continuous,
            predicted_prior,
            speech_target_lengths.new_zeros(batch_size),
            [],
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

    masked_blocks = maskable_target_mask.clone()
    block_confidence = torch.full(
        (batch_size, max_target),
        fill_value=-1e9,
        dtype=torch.float32,
        device=device,
    )
    collected_logits: list[torch.Tensor] = []

    num_parallel_steps = int(parallel_num_steps)
    for step_idx in range(num_parallel_steps):
        if not bool(masked_blocks.any().item()):
            break

        current_discrete = predicted_discrete.clone()
        current_continuous = predicted_continuous.clone()
        if masked_blocks.any():
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

        if masked_discrete_positions.any():
            discrete_hidden = hidden_states[masked_discrete_positions]
            discrete_stream_idx = flat.speech_stream_ids[masked_discrete_positions]
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
            discrete_batch_idx = batch_indices[masked_discrete_positions]
            discrete_block_idx = flat.target_block_ids[masked_discrete_positions]
            predicted_discrete[discrete_batch_idx, discrete_block_idx, discrete_stream_idx] = (
                sampled_discrete
            )

            sampled_log_probs = F.log_softmax(discrete_logits, dim=-1).gather(
                -1,
                sampled_discrete.unsqueeze(-1),
            ).squeeze(-1)
            step_conf_sum = torch.zeros(
                (batch_size, max_target),
                dtype=torch.float32,
                device=device,
            )
            step_conf_count = torch.zeros(
                (batch_size, max_target),
                dtype=torch.float32,
                device=device,
            )
            step_conf_sum.index_put_(
                (discrete_batch_idx, discrete_block_idx),
                sampled_log_probs.to(dtype=torch.float32),
                accumulate=True,
            )
            step_conf_count.index_put_(
                (discrete_batch_idx, discrete_block_idx),
                torch.ones_like(sampled_log_probs, dtype=torch.float32),
                accumulate=True,
            )
            has_confidence = step_conf_count > 0
            averaged_confidence = torch.zeros_like(step_conf_sum)
            averaged_confidence[has_confidence] = (
                step_conf_sum[has_confidence] / step_conf_count[has_confidence]
            )
            block_confidence = torch.where(
                has_confidence,
                averaged_confidence,
                block_confidence,
            )
            collected_logits.append(discrete_logits)
        else:
            collected_logits.append(
                torch.empty(
                    0,
                    model.discrete_vocab_size,
                    dtype=hidden_states.dtype,
                    device=device,
                )
            )

        if masked_continuous_positions.any():
            continuous_hidden = hidden_states[masked_continuous_positions]
            prior_prediction = model.continuous_prior_head(continuous_hidden)
            continuous_batch_idx = batch_indices[masked_continuous_positions]
            continuous_block_idx = flat.target_block_ids[masked_continuous_positions]
            predicted_prior[continuous_batch_idx, continuous_block_idx, :] = prior_prediction
            denoise_target_mask = torch.zeros_like(masked_blocks)
            denoise_target_mask[continuous_batch_idx, continuous_block_idx] = True
            denoised_targets = model._sample_continuous_with_clean_context(
                prior_target=predicted_prior,
                clean_target=predicted_continuous,
                denoise_target_mask=denoise_target_mask,
                valid_target_mask=valid_target_mask,
                prompt_latents=continuous_prompt,
                prompt_lengths=speech_prompt_lengths,
                num_steps=flow_num_steps,
                temperature=temperature,
            )
            predicted_continuous[denoise_target_mask] = denoised_targets[denoise_target_mask]

            if len(special_discrete_token_ids) > 0:
                step_discrete = predicted_discrete[continuous_batch_idx, continuous_block_idx, :]
                special_step_mask = MU.build_special_block_mask(
                    discrete_tokens=step_discrete,
                    special_token_ids=special_discrete_token_ids,
                )
                if special_step_mask.any():
                    predicted_continuous[
                        continuous_batch_idx[special_step_mask],
                        continuous_block_idx[special_step_mask],
                        :,
                    ] = 0.0

        if step_idx >= num_parallel_steps - 1:
            break

        next_masked_blocks = torch.zeros_like(masked_blocks)
        remaining_ratio = math.cos(
            (math.pi * float(step_idx + 1)) / (2.0 * float(num_parallel_steps))
        )
        for sample_idx in range(batch_size):
            target_len = int(speech_target_lengths[sample_idx].item())
            maskable_len = max(0, target_len - 1)
            if maskable_len <= 0:
                continue

            next_mask_count = int(round(remaining_ratio * float(maskable_len)))
            next_mask_count = min(maskable_len, max(0, next_mask_count))
            if next_mask_count <= 0:
                continue

            sample_confidence = block_confidence[sample_idx, :maskable_len]
            low_confidence_blocks = torch.topk(
                sample_confidence,
                k=next_mask_count,
                largest=False,
                dim=-1,
            ).indices
            next_masked_blocks[sample_idx, low_confidence_blocks] = True

        masked_blocks = next_masked_blocks & maskable_target_mask

    generated_lengths = speech_target_lengths.clone()
    return (
        predicted_discrete,
        predicted_continuous,
        predicted_prior,
        generated_lengths,
        collected_logits,
    )
