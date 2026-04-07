import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from production_priming_common import (
    CORE_FILLER_SENTENCES,
    JABBERWOCKY_FILLER_SENTENCES,
    REPO_ROOT,
    batched_choice_log_probs,
    get_device,
    normalize_transitive_frame,
    prompt_condition_order,
    resolve_prime_sentence,
    sample_condition_frames,
    write_common_outputs,
)


DEFAULT_CORE_CSV = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_constrained_counterbalanced.csv"
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
        "--local-files-only",
        action="store_true",
        help="Load tokenizer/model from local Hugging Face cache only.",
    )
    parser.add_argument(
        "--prime-conditions",
        nargs="+",
        default=["active", "passive", "no_prime_eos", "no_prime_empty", "filler"],
        help="Subset of active passive no_prime_eos no_prime_empty filler (no_prime aliases to no_prime_eos).",
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
            elif prime_condition == "no_prime_empty":
                # True empty-context baseline; batched scorer handles prompt_len=0 safely.
                prompt = ""
            else:
                # Boundary-cued baseline.
                prompt = f"{tokenizer.eos_token} "

            prompt_groups.append(
                (
                    prompt,
                    len(tokenizer(prompt, add_special_tokens=False)["input_ids"]),
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
                    "choice_target": "full_sentence_processing",
                }
            )

    return prompt_groups, row_metadata


def infer_filler_domain(input_csv: Path, prime_csv: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    probe = f"{input_csv.name} {prime_csv.name}".lower()
    return "jabberwocky" if "jabberwocky" in probe else "core"


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

    device = get_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

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
    batched_scores = batched_choice_log_probs(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt_groups=prompt_groups,
        batch_size=args.batch_size,
    )

    rows: List[Dict[str, object]] = []
    for metadata, candidate_log_probs in zip(row_metadata, batched_scores):
        active_sum, passive_sum = candidate_log_probs
        active_len = len(tokenizer(" " + metadata["target_active"], add_special_tokens=False)["input_ids"])
        passive_len = len(tokenizer(" " + metadata["target_passive"], add_special_tokens=False)["input_ids"])
        active_mean = active_sum / max(1, active_len)
        passive_mean = passive_sum / max(1, passive_len)
        chosen_structure = "passive" if passive_mean > active_mean else "active"
        rows.append(
            {
                **metadata,
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
        )

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
        "local_files_only": bool(args.local_files_only),
        "n_rows": int(len(results)),
        "n_items": int(len(target_frame)),
        "prime_alignment_mode": prime_alignment_mode,
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
