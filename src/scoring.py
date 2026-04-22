from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn.functional as F

from .models import CausalLMWrapper, PromptBuildResult


@dataclass(frozen=True)
class CandidateScore:
    candidate_text: str
    total_logprob: float
    mean_logprob: float
    num_candidate_tokens: int
    prefix_token_count: int
    candidate_token_ids: List[int]
    candidate_tokens: List[str]
    candidate_token_logprobs: List[float]
    prompt_format: str
    rendered_prefix: str


@dataclass(frozen=True)
class PairwisePreference:
    score_a: CandidateScore
    score_b: CandidateScore
    total_logprob_diff_a_minus_b: float
    mean_logprob_diff_a_minus_b: float


def _pad_sequences(sequences: Sequence[Sequence[int]], pad_token_id: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    output = torch.full((len(sequences), max_len), fill_value=pad_token_id, dtype=torch.long)
    for row_idx, seq in enumerate(sequences):
        output[row_idx, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return output


def score_candidates_batched(
    model_wrapper: CausalLMWrapper,
    prefixes: Sequence[str],
    candidates: Sequence[str],
) -> List[CandidateScore]:
    if len(prefixes) != len(candidates):
        raise ValueError("prefixes and candidates must have equal length")
    if not prefixes:
        return []

    prompt_objects: List[PromptBuildResult] = [
        model_wrapper.build_prompt_ids(prefix_text=prefix, candidate_text=candidate)
        for prefix, candidate in zip(prefixes, candidates)
    ]

    full_ids_list = [payload.full_ids for payload in prompt_objects]
    prefix_ids_list = [payload.prefix_ids for payload in prompt_objects]

    pad_token_id = model_wrapper.tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must expose pad_token_id for batched scoring.")

    input_ids = _pad_sequences(full_ids_list, pad_token_id=pad_token_id).to(model_wrapper.device)
    attention_mask = (input_ids != pad_token_id).long().to(model_wrapper.device)

    with torch.no_grad():
        logits = model_wrapper.model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Shift for next-token probabilities.
    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    shifted_attention = attention_mask[:, 1:]
    token_logprobs = F.log_softmax(shifted_logits, dim=-1)
    gathered_logprobs = torch.gather(token_logprobs, dim=2, index=shifted_labels.unsqueeze(-1)).squeeze(-1)

    scores: List[CandidateScore] = []
    for idx, payload in enumerate(prompt_objects):
        full_ids = full_ids_list[idx]
        prefix_ids = prefix_ids_list[idx]
        prefix_len = len(prefix_ids)
        full_len = len(full_ids)

        if prefix_len < 1:
            raise ValueError(
                "Prefix token length is zero; cannot compute conditional candidate logprob. "
                "Use non-empty prompts or enable BOS controls."
            )
        if prefix_len >= full_len:
            raise ValueError(
                "Candidate contributed no tokens. "
                f"prefix_len={prefix_len} full_len={full_len}"
            )

        # In shifted arrays, index i corresponds to token position i+1 in the unshifted input.
        shift_start = prefix_len - 1
        shift_end = full_len - 1

        row_logprobs = gathered_logprobs[idx, shift_start:shift_end]
        row_attention = shifted_attention[idx, shift_start:shift_end]

        # Safety: remove any padding spillover.
        row_logprobs = row_logprobs[row_attention.bool()]

        candidate_token_ids = full_ids[prefix_len:full_len]
        if len(candidate_token_ids) != int(row_logprobs.numel()):
            raise ValueError(
                "Candidate token count mismatch while scoring. "
                f"candidate_tokens={len(candidate_token_ids)} logprob_tokens={int(row_logprobs.numel())}"
            )

        total_logprob = float(row_logprobs.sum().item())
        num_tokens = int(row_logprobs.numel())
        mean_logprob = total_logprob / float(num_tokens)

        scores.append(
            CandidateScore(
                candidate_text=candidates[idx],
                total_logprob=total_logprob,
                mean_logprob=mean_logprob,
                num_candidate_tokens=num_tokens,
                prefix_token_count=prefix_len,
                candidate_token_ids=[int(token_id) for token_id in candidate_token_ids],
                candidate_tokens=model_wrapper.tokenizer.convert_ids_to_tokens(candidate_token_ids),
                candidate_token_logprobs=[float(value) for value in row_logprobs.tolist()],
                prompt_format=payload.prompt_format,
                rendered_prefix=payload.rendered_prefix,
            )
        )

    return scores


def score_pairwise_preference(
    model_wrapper: CausalLMWrapper,
    prefix: str,
    candidate_a: str,
    candidate_b: str,
) -> PairwisePreference:
    scores = score_candidates_batched(
        model_wrapper=model_wrapper,
        prefixes=[prefix, prefix],
        candidates=[candidate_a, candidate_b],
    )
    score_a = scores[0]
    score_b = scores[1]
    return PairwisePreference(
        score_a=score_a,
        score_b=score_b,
        total_logprob_diff_a_minus_b=score_a.total_logprob - score_b.total_logprob,
        mean_logprob_diff_a_minus_b=score_a.mean_logprob - score_b.mean_logprob,
    )
