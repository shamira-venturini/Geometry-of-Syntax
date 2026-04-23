import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch

from production_priming_common import (
    CORE_FILLER_SENTENCES,
    JABBERWOCKY_FILLER_SENTENCES,
    LEXICALLY_CONTROLLED_CORE_CSV,
    REPO_ROOT,
    batched_choice_detailed_scores,
    get_device,
    lexical_overlap_audit,
    load_causal_lm_and_tokenizer,
    normalize_transitive_frame,
    prompt_condition_order,
    resolve_prime_sentence,
    sample_condition_frames,
    write_common_outputs,
)


DEFAULT_CORE_CSV = LEXICALLY_CONTROLLED_CORE_CSV
DEFAULT_OUTPUT_DIR = REPO_ROOT / "behavioral_results" / "processing_experiment_1b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 1b: controlled processing-style structural priming with unified metrics."
    )
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CORE_CSV)
    parser.add_argument(
        "--prime-csv",
        type=Path,
        default=None,
        help="Prime source. If omitted, input-csv is used (domain-matched condition).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Torch dtype for model loading: auto, float32, float16, or bfloat16.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer/model from local Hugging Face cache only.",
    )
    parser.add_argument(
        "--prime-conditions",
        nargs="+",
        default=["active", "passive", "no_prime", "filler"],
        help="Subset of active passive no_prime filler.",
    )
    parser.add_argument(
        "--filler-domain",
        choices=("auto", "core", "jabberwocky"),
        default="auto",
        help="Filler pool to use. auto infers from input/prime CSV path.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--condition-label",
        default="processing_1b",
        help="Tag saved into metadata for run provenance.",
    )
    return parser.parse_args()


def build_prompt_groups(
    target_frame: pd.DataFrame,
    prime_frame: pd.DataFrame,
    tokenizer,
    prime_conditions: List[str],
    seed: int,
    filler_sentences: List[str],
) -> Tuple[List[Tuple[str, int, List[str], List[int]]], List[Dict[str, object]]]:
    prompt_groups: List[Tuple[str, int, List[str], List[int]]] = []
    row_metadata: List[Dict[str, object]] = []

    for item_index, target_row in target_frame.iterrows():
        prime_row = prime_frame.loc[item_index]
        active_target = str(target_row["ta"]).strip()
        passive_target = str(target_row["tp"]).strip()
        candidates = [f" {active_target}", f" {passive_target}"]
        candidate_lengths = [
            len(tokenizer(text, add_special_tokens=False)["input_ids"])
            for text in candidates
        ]

        for prime_condition in prime_conditions:
            prime_sentence = resolve_prime_sentence(
                prime_condition=prime_condition,
                prime_row=prime_row,
                item_index=item_index,
                filler_seed=seed,
                filler_sentences=filler_sentences,
            )
            if prime_sentence:
                prompt = prime_sentence.strip() + " "
            else:
                # Empty-context no-prime baseline; scorer handles prompt_len=0 safely.
                prompt = ""
            prompt_token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

            prompt_groups.append(
                (
                    prompt,
                    prompt_token_count,
                    candidates,
                    candidate_lengths,
                )
            )
            row_metadata.append(
                {
                    "item_index": item_index,
                    "prompt_template": "processing_teacher_forced",
                    "prime_condition": prime_condition,
                    "prime_structure": prime_condition,
                    "prime_sentence": prime_sentence or "",
                    "target_active": active_target,
                    "target_passive": passive_target,
                    "prompt": prompt,
                    "prompt_text": prompt,
                    "prompt_token_count": prompt_token_count,
                    "choice_target": "full_sentence_processing",
                }
            )

    return prompt_groups, row_metadata


def infer_filler_domain(input_csv: Path, prime_csv: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    probe = f"{input_csv.name} {prime_csv.name}".lower()
    return "jabberwocky" if "jabberwocky" in probe else "core"


def _token_debug_json(values: Sequence[object]) -> str:
    return json.dumps(list(values), ensure_ascii=True)


def _safe_token_value(values: Sequence[float], index: int) -> float:
    if not values:
        return float("nan")
    if index < 0:
        index = len(values) + index
    if index < 0 or index >= len(values):
        return float("nan")
    return float(values[index])


def _shared_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    limit = min(len(a), len(b))
    count = 0
    while count < limit and int(a[count]) == int(b[count]):
        count += 1
    return count


def _location_metrics(
    *,
    token_ids_a: Sequence[int],
    token_ids_b: Sequence[int],
    token_logprobs_a: Sequence[float],
    token_logprobs_b: Sequence[float],
) -> Dict[str, object]:
    shared_prefix = _shared_prefix_len(token_ids_a, token_ids_b)
    divergence_index = shared_prefix if shared_prefix < min(len(token_ids_a), len(token_ids_b)) else -1
    aligned_length = min(len(token_logprobs_a), len(token_logprobs_b))
    aligned_diffs = [
        float(token_logprobs_a[idx] - token_logprobs_b[idx])
        for idx in range(aligned_length)
    ]

    if aligned_diffs:
        aligned_mean = float(sum(aligned_diffs) / len(aligned_diffs))
        aligned_first = float(aligned_diffs[0])
        aligned_last = float(aligned_diffs[-1])
    else:
        aligned_mean = float("nan")
        aligned_first = float("nan")
        aligned_last = float("nan")

    if divergence_index >= 0:
        divergence_a = _safe_token_value(token_logprobs_a, divergence_index)
        divergence_b = _safe_token_value(token_logprobs_b, divergence_index)
        divergence_diff = divergence_a - divergence_b
    else:
        divergence_a = float("nan")
        divergence_b = float("nan")
        divergence_diff = float("nan")

    return {
        "shared_prefix_token_count": int(shared_prefix),
        "divergence_token_index": int(divergence_index),
        "candidate_a_first_token_logprob": _safe_token_value(token_logprobs_a, 0),
        "candidate_b_first_token_logprob": _safe_token_value(token_logprobs_b, 0),
        "candidate_a_second_token_logprob": _safe_token_value(token_logprobs_a, 1),
        "candidate_b_second_token_logprob": _safe_token_value(token_logprobs_b, 1),
        "candidate_a_last_token_logprob": _safe_token_value(token_logprobs_a, -1),
        "candidate_b_last_token_logprob": _safe_token_value(token_logprobs_b, -1),
        "candidate_a_divergence_token_logprob": divergence_a,
        "candidate_b_divergence_token_logprob": divergence_b,
        "preference_first_token": _safe_token_value(token_logprobs_a, 0) - _safe_token_value(token_logprobs_b, 0),
        "preference_second_token": _safe_token_value(token_logprobs_a, 1) - _safe_token_value(token_logprobs_b, 1),
        "preference_last_token": _safe_token_value(token_logprobs_a, -1) - _safe_token_value(token_logprobs_b, -1),
        "preference_divergence_token": divergence_diff,
        "preference_aligned_mean": aligned_mean,
        "preference_aligned_first": aligned_first,
        "preference_aligned_last": aligned_last,
        "preference_tokenwise_aligned_diffs": _token_debug_json(aligned_diffs),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    input_csv = args.input_csv.resolve()
    prime_csv = args.prime_csv.resolve() if args.prime_csv else input_csv
    target_frame = normalize_transitive_frame(pd.read_csv(input_csv))
    prime_frame = normalize_transitive_frame(pd.read_csv(prime_csv))
    target_frame, prime_frame, prime_alignment_mode = sample_condition_frames(
        target_frame=target_frame,
        prime_frame=prime_frame,
        max_items=args.max_items,
        seed=args.seed,
    )
    overlap_audit = lexical_overlap_audit(target_frame=target_frame, prime_frame=prime_frame)

    device = get_device(args.device)
    tokenizer, model, resolved_dtype = load_causal_lm_and_tokenizer(
        model_name=args.model_name,
        device=device,
        local_files_only=args.local_files_only,
        torch_dtype_name=args.torch_dtype,
    )

    prime_conditions = prompt_condition_order(args.prime_conditions)
    filler_domain = infer_filler_domain(input_csv=input_csv, prime_csv=prime_csv, requested=args.filler_domain)
    filler_sentences = (
        JABBERWOCKY_FILLER_SENTENCES if filler_domain == "jabberwocky" else CORE_FILLER_SENTENCES
    )
    prompt_groups, row_metadata = build_prompt_groups(
        target_frame=target_frame,
        prime_frame=prime_frame,
        tokenizer=tokenizer,
        prime_conditions=prime_conditions,
        seed=args.seed,
        filler_sentences=filler_sentences,
    )
    batched_scores = batched_choice_detailed_scores(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt_groups=prompt_groups,
        batch_size=args.batch_size,
    )

    rows: List[Dict[str, object]] = []
    for metadata, candidate_scores in zip(row_metadata, batched_scores):
        active_score, passive_score = candidate_scores
        active_sum = float(active_score["total_logprob"])
        passive_sum = float(passive_score["total_logprob"])
        active_len = int(active_score["token_count"])
        passive_len = int(passive_score["token_count"])
        active_mean = active_sum / max(1, active_len)
        passive_mean = passive_sum / max(1, passive_len)
        chosen_structure = "passive" if passive_mean > active_mean else "active"

        row: Dict[str, object] = {
            **metadata,
            "model_name": args.model_name,
            "model_condition": args.condition_label,
            "task": "processing_experiment_1b",
            "task_short": "processing_experiment_1b",
            "task_family": "processing_like_disambiguation",
            "prompt_format_used": "plain_text",
            "question_template_used": "",
            "target_voice": "active_vs_passive",
            "target_sentence_used": "",
            "lexicality_condition": "nonce" if filler_domain == "jabberwocky" else "real",
            "candidate_a_label": "active_completion",
            "candidate_a_text": str(active_score["candidate_text"]),
            "candidate_a_total_logprob": active_sum,
            "candidate_a_mean_logprob": float(active_score["mean_logprob"]),
            "candidate_a_token_count": active_len,
            "candidate_a_token_ids": _token_debug_json(active_score["candidate_token_ids"]),
            "candidate_a_tokens": _token_debug_json(active_score["candidate_tokens"]),
            "candidate_a_token_logprobs": _token_debug_json(active_score["candidate_token_logprobs"]),
            "candidate_b_label": "passive_completion",
            "candidate_b_text": str(passive_score["candidate_text"]),
            "candidate_b_total_logprob": passive_sum,
            "candidate_b_mean_logprob": float(passive_score["mean_logprob"]),
            "candidate_b_token_count": passive_len,
            "candidate_b_token_ids": _token_debug_json(passive_score["candidate_token_ids"]),
            "candidate_b_tokens": _token_debug_json(passive_score["candidate_tokens"]),
            "candidate_b_token_logprobs": _token_debug_json(passive_score["candidate_token_logprobs"]),
            "preference_total": active_sum - passive_sum,
            "preference_mean": active_mean - passive_mean,
            "active_minus_passive_logprob_total": active_sum - passive_sum,
            "active_minus_passive_logprob_mean": active_mean - passive_mean,
            "active_choice_logprob": active_mean,
            "passive_choice_logprob": passive_mean,
            "passive_minus_active_logprob": passive_mean - active_mean,
            "active_choice_logprob_sum": active_sum,
            "passive_choice_logprob_sum": passive_sum,
            "passive_minus_active_logprob_sum": passive_sum - active_sum,
            "active_target_token_count": active_len,
            "passive_target_token_count": passive_len,
            "chosen_structure": chosen_structure,
            "passive_choice_indicator": 1.0 if chosen_structure == "passive" else 0.0,
        }
        row.update(
            _location_metrics(
                token_ids_a=[int(value) for value in active_score["candidate_token_ids"]],
                token_ids_b=[int(value) for value in passive_score["candidate_token_ids"]],
                token_logprobs_a=[float(value) for value in active_score["candidate_token_logprobs"]],
                token_logprobs_b=[float(value) for value in passive_score["candidate_token_logprobs"]],
            )
        )
        rows.append(row)

    results = pd.DataFrame(rows)
    metadata = {
        "condition_label": args.condition_label,
        "model_name": args.model_name,
        "input_csv": str(input_csv),
        "prime_csv": str(prime_csv),
        "max_items": None if args.max_items is None else int(args.max_items),
        "batch_size": int(args.batch_size),
        "prime_conditions": prime_conditions,
        "seed": int(args.seed),
        "device": device,
        "torch_dtype": str(resolved_dtype) if resolved_dtype is not None else "default",
        "local_files_only": bool(args.local_files_only),
        "n_rows": int(len(results)),
        "n_items": int(len(target_frame)),
        "prime_alignment_mode": prime_alignment_mode,
        "lexical_overlap_audit": overlap_audit,
        "filler_sentence_count": len(filler_sentences),
        "filler_domain": filler_domain,
        "task_type": "processing_experiment_1b",
        "primary_score": "mean_target_logprob_difference",
        "secondary_score": "sum_target_logprob_difference",
    }
    write_common_outputs(
        frame=results,
        output_dir=output_dir,
        title="Experiment 1b: Controlled Processing Structural Priming",
        prime_condition_ordering=prime_conditions,
        extra_metadata=metadata,
    )

    # Save an explicit processing-oriented view for convenience.
    processing_summary = results.groupby(["prime_condition"], as_index=False).agg(
        n_items=("item_index", "count"),
        mean_delta_mean=("passive_minus_active_logprob", "mean"),
        mean_delta_sum=("passive_minus_active_logprob_sum", "mean"),
        passive_choice_rate=("passive_choice_indicator", "mean"),
    )
    processing_summary.to_csv(output_dir / "processing_summary.csv", index=False)
    (output_dir / "processing_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
