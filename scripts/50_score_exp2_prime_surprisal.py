#!/usr/bin/env python3
"""Score Experiment 2 prime sentences under their own event contexts.

This is a lightweight scoring pass for IFE-style analyses. It does not rerun
Experiment 2 generation. Instead, it reconstructs the prime demonstration in
the Exp2 prompt:

    prime event description + Mary answered, "

and scores the active/passive prime answer sentence itself.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from production_priming_common import (  # noqa: E402
    batched_choice_detailed_scores,
    get_device,
    load_causal_lm_and_tokenizer,
)


DEFAULT_PROMPT_CSV = (
    REPO_ROOT
    / "behavioral_results/generated_materials/experiment-2/prompts"
    / "experiment_2_core_demo_prompts_lexically_controlled.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "behavioral_results/experiment-2/exp2_prime_surprisal"
PRIME_TYPES = ("active", "passive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--model-condition", default=None)
    parser.add_argument("--prompt-csv", type=Path, default=DEFAULT_PROMPT_CSV)
    parser.add_argument("--dataset-label", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-items", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def infer_dataset_label(path: Path) -> str:
    name = path.name.lower()
    if "core_targets_jabberwocky_primes" in name:
        return "core_targets_jabberwocky_primes"
    if "jabberwocky" in name:
        return "jabberwocky"
    if "core" in name:
        return "core"
    return "unspecified"


def normalize_lexicality(dataset_label: str) -> str:
    if dataset_label == "core":
        return "core"
    if dataset_label == "jabberwocky":
        return "jabberwocky"
    if dataset_label == "core_targets_jabberwocky_primes":
        return "mixed"
    return dataset_label


def extract_first_mary_answer(prompt: str) -> tuple[str, str]:
    marker = 'Mary answered, "'
    start = prompt.find(marker)
    if start < 0:
        raise ValueError("Could not find first Mary answered marker in prompt.")
    answer_start = start + len(marker)
    answer_end = prompt.find('"', answer_start)
    if answer_end < 0:
        raise ValueError("Could not find closing quote for first Mary answered sentence.")
    context = prompt[:answer_start]
    answer = prompt[answer_start:answer_end]
    if not answer:
        raise ValueError("Extracted empty prime answer.")
    return context, answer


def safe_json(values: object) -> str:
    return json.dumps(list(values), ensure_ascii=True)


def build_prime_rows(frame: pd.DataFrame, dataset_label: str, max_items: int | None) -> pd.DataFrame:
    if max_items is not None:
        frame = frame.loc[frame["item_index"].astype(int) < int(max_items)].copy()
    rows: List[Dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        for prime_type in PRIME_TYPES:
            prompt_column = f"prompt_{prime_type}"
            if prompt_column not in row:
                raise ValueError(f"Missing prompt column: {prompt_column}")
            context, answer = extract_first_mary_answer(str(row[prompt_column]))
            declared_sentence = str(row.get(f"prime_{prime_type}_sentence", "")).strip()
            normalized_declared = re.sub(r"\s+", " ", declared_sentence).strip(" .").lower()
            normalized_answer = re.sub(r"\s+", " ", answer).strip(" .").lower()
            rows.append(
                {
                    "item_index": int(row["item_index"]),
                    "dataset": dataset_label,
                    "lexicality_condition": normalize_lexicality(dataset_label),
                    "prime_type": prime_type,
                    "prime_condition": prime_type,
                    "prompt_column": prompt_column,
                    "prime_context": context,
                    "prime_sentence": answer,
                    "declared_prime_sentence": declared_sentence,
                    "prime_sentence_matches_column": normalized_declared == normalized_answer,
                    "target_active": row.get("target_active", ""),
                    "target_passive": row.get("target_passive", ""),
                    "event_style": row.get("event_style", ""),
                    "role_style": row.get("role_style", ""),
                    "quote_style": row.get("quote_style", ""),
                    "role_order": row.get("role_order", ""),
                    "target_verb_cue": row.get("target_verb_cue", ""),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No prime rows were constructed.")
    return out


def score_prime_rows(
    *,
    frame: pd.DataFrame,
    tokenizer,
    model,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    prompt_groups = []
    metadata_rows: List[Dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        context = str(row["prime_context"])
        continuation = str(row["prime_sentence"])
        prompt_token_count = len(tokenizer(context, add_special_tokens=False)["input_ids"])
        continuation_token_count = len(tokenizer(continuation, add_special_tokens=False)["input_ids"])
        prompt_groups.append((context, prompt_token_count, [continuation], [continuation_token_count]))
        metadata_rows.append(row)

    print(
        f"Scoring {len(prompt_groups)} Exp2 prime sentences "
        f"(batch_size={batch_size}).",
        flush=True,
    )
    detailed_scores = batched_choice_detailed_scores(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt_groups=prompt_groups,
        batch_size=max(1, int(batch_size)),
        progress_label="Exp2 prime surprisal",
    )

    output_rows: List[Dict[str, object]] = []
    for metadata, group in zip(metadata_rows, detailed_scores):
        score = group[0]
        logprob_sum = float(score["total_logprob"])
        logprob_mean = float(score["mean_logprob"])
        output_rows.append(
            {
                **metadata,
                "prime_logprob_sum": logprob_sum,
                "prime_logprob_mean": logprob_mean,
                "prime_surprisal_sum": -logprob_sum,
                "prime_surprisal_mean": -logprob_mean,
                "prime_token_count": int(score["token_count"]),
                "prime_token_ids_json": safe_json(score["candidate_token_ids"]),
                "prime_tokens_json": safe_json(score["candidate_tokens"]),
                "prime_token_logprobs_json": safe_json(score["candidate_token_logprobs"]),
            }
        )
    return pd.DataFrame(output_rows)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    prompt_csv = args.prompt_csv.resolve()
    dataset_label = args.dataset_label or infer_dataset_label(prompt_csv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(prompt_csv).fillna("")
    prime_rows = build_prime_rows(frame, dataset_label=dataset_label, max_items=args.max_items)

    device = get_device(args.device)
    tokenizer, model, resolved_dtype = load_causal_lm_and_tokenizer(
        model_name=args.model_name,
        device=device,
        local_files_only=args.local_files_only,
        torch_dtype_name=args.torch_dtype,
    )

    scored = score_prime_rows(
        frame=prime_rows,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    scored["model_name"] = args.model_name
    scored["model_condition"] = args.model_condition or args.model_name
    scored["prompt_format_used"] = "plain_text"

    item_path = output_dir / "prime_surprisal_item_level.csv"
    summary_path = output_dir / "prime_surprisal_summary.csv"
    metadata_path = output_dir / "run_metadata_exp2_prime_surprisal.json"

    scored.to_csv(item_path, index=False)
    summary = (
        scored.groupby(["dataset", "lexicality_condition", "prime_type"], as_index=False)
        .agg(
            n_items=("item_index", "count"),
            prime_surprisal_mean=("prime_surprisal_mean", "mean"),
            prime_surprisal_sum=("prime_surprisal_sum", "mean"),
            prime_token_count=("prime_token_count", "mean"),
            match_rate=("prime_sentence_matches_column", "mean"),
        )
    )
    summary.to_csv(summary_path, index=False)
    metadata = {
        "model_name": args.model_name,
        "model_condition": args.model_condition or args.model_name,
        "prompt_csv": str(prompt_csv),
        "dataset_label": dataset_label,
        "max_items": args.max_items,
        "batch_size": args.batch_size,
        "device": device,
        "torch_dtype": str(resolved_dtype),
        "n_rows": int(len(scored)),
        "output_item_level": str(item_path),
        "output_summary": str(summary_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {item_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
