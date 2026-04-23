from __future__ import annotations

from typing import Sequence

import torch

from .models import CausalLMWrapper


def greedy_generate_answers(
    *,
    model_wrapper: CausalLMWrapper,
    prompts: Sequence[str],
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    if not prompts:
        return []

    tokenizer = model_wrapper.tokenizer
    model = model_wrapper.model
    device = model_wrapper.device

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    outputs: list[str] = []
    try:
        for batch_start in range(0, len(prompts), max(1, int(batch_size))):
            batch_prompts = list(prompts[batch_start : batch_start + max(1, int(batch_size))])
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max(1, int(max_new_tokens)),
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            input_width = int(inputs.input_ids.shape[1])
            for row_index in range(len(batch_prompts)):
                new_tokens = generated[row_index, input_width:]
                outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = old_padding_side

    return outputs
