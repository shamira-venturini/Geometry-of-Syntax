#!/usr/bin/env python3
"""Compute Sinclair-style target PE for Experiment 4 prime-target pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
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
    one_sample_summary,
)


DEFAULT_PROMPT_CSV = (
    REPO_ROOT
    / "behavioral_results/generated_materials/experiment-4/complex_np"
    / "experiment_4_complex_np_core_role_recovery_prompts.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "behavioral_results/experiment-4/exp4_sinclair_target_pe"
PRIME_ORDER = ["active", "passive", "filler", "no_prime"]
TARGET_VOICE_ORDER = ["active", "passive"]
METRIC_COLUMNS = [
    "pe_active_target_logprob_sum_same_minus_other",
    "pe_passive_target_logprob_sum_same_minus_other",
    "pe_logprob_sum_imbalance_passive_minus_active",
    "pe_active_target_logprob_mean_same_minus_other",
    "pe_passive_target_logprob_mean_same_minus_other",
    "pe_logprob_mean_imbalance_passive_minus_active",
    "active_target_shift_active_minus_no_prime_sum",
    "passive_target_shift_passive_minus_no_prime_sum",
    "active_target_shift_active_minus_filler_sum",
    "passive_target_shift_passive_minus_filler_sum",
    "baseline_no_prime_passive_minus_active_logprob_sum",
    "baseline_filler_passive_minus_active_logprob_sum",
    "active_target_shift_active_minus_no_prime_mean",
    "passive_target_shift_passive_minus_no_prime_mean",
    "active_target_shift_active_minus_filler_mean",
    "passive_target_shift_passive_minus_filler_mean",
    "baseline_no_prime_passive_minus_active_logprob_mean",
    "baseline_filler_passive_minus_active_logprob_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--model-condition", default=None)
    parser.add_argument("--prompt-csv", type=Path, default=DEFAULT_PROMPT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-items",
        type=int,
        default=2048,
        help="Maximum base item_index value to keep. Keeps all selected variants for selected items.",
    )
    parser.add_argument("--prime-conditions", nargs="+", default=PRIME_ORDER)
    parser.add_argument("--target-voices", nargs="+", default=TARGET_VOICE_ORDER)
    parser.add_argument(
        "--complexity-conditions",
        nargs="+",
        default=["agent_complex", "patient_complex", "both_complex"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def _validate_subset(frame: pd.DataFrame, column: str, requested: Sequence[str]) -> List[str]:
    values = [str(value) for value in requested if str(value).strip()]
    if not values:
        raise ValueError(f"At least one {column} value must be selected.")
    available = set(frame[column].astype(str))
    invalid = sorted(set(values).difference(available))
    if invalid:
        raise ValueError(f"Invalid {column} values {invalid}; available={sorted(available)}")
    return values


def load_prime_target_rows(args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_csv(args.prompt_csv.resolve()).fillna("")
    if args.max_items is not None:
        frame = frame.loc[frame["item_index"].astype(int) < int(args.max_items)].copy()

    prime_conditions = _validate_subset(frame, "prime_condition", args.prime_conditions)
    target_voices = _validate_subset(frame, "target_voice", args.target_voices)
    complexity_conditions = _validate_subset(frame, "complexity_condition", args.complexity_conditions)

    filtered = frame.loc[
        frame["prime_condition"].astype(str).isin(prime_conditions)
        & frame["target_voice"].astype(str).isin(target_voices)
        & frame["complexity_condition"].astype(str).isin(complexity_conditions)
    ].copy()
    if filtered.empty:
        raise ValueError("Prompt selection produced zero rows.")

    dedupe_columns = [
        "item_index",
        "complexity_condition",
        "source_label",
        "lexicality_condition",
        "target_cell",
        "prime_cell",
        "prime_condition",
        "target_voice",
        "prime_text",
        "target_sentence",
        "target_active_complex",
        "target_passive_complex",
    ]
    keep_columns = [column for column in dedupe_columns if column in filtered.columns]
    deduped = filtered[keep_columns].drop_duplicates().reset_index(drop=True)
    if deduped.empty:
        raise ValueError("Deduplicated prime-target selection produced zero rows.")
    return deduped


def score_target_rows(
    *,
    frame: pd.DataFrame,
    tokenizer,
    model,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    prompt_groups = []
    metadata_rows: List[Dict[str, object]] = []
    for row in frame.itertuples(index=False):
        prime_text = str(getattr(row, "prime_text", "") or "").strip()
        target_sentence = str(getattr(row, "target_sentence")).strip()
        prompt = f"{prime_text} " if prime_text else ""
        continuation = f" {target_sentence}"
        prompt_token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        continuation_token_count = len(tokenizer(continuation, add_special_tokens=False)["input_ids"])
        prompt_groups.append((prompt, prompt_token_count, [continuation], [continuation_token_count]))
        metadata_rows.append(row._asdict())

    print(
        f"Experiment 4 Sinclair PE selected {len(prompt_groups)} unique prime-target rows "
        f"(batch_size={batch_size}).",
        flush=True,
    )
    detailed_scores = batched_choice_detailed_scores(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt_groups=prompt_groups,
        batch_size=max(1, int(batch_size)),
        progress_label="Exp4 Sinclair target PE",
    )

    output_rows: List[Dict[str, object]] = []
    for metadata, group in zip(metadata_rows, detailed_scores):
        score = group[0]
        output_rows.append(
            {
                **metadata,
                "target_logprob_sum": float(score["total_logprob"]),
                "target_logprob_mean": float(score["mean_logprob"]),
                "target_token_count_scored": int(score["token_count"]),
                "target_token_ids_json": json.dumps(score["candidate_token_ids"]),
                "target_tokens_json": json.dumps(score["candidate_tokens"]),
                "target_token_logprobs_json": json.dumps(score["candidate_token_logprobs"]),
            }
        )
    return pd.DataFrame(output_rows)


def _pivot_scores(score_frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    index_columns = ["item_index", "complexity_condition", "lexicality_condition"]
    optional_columns = ["target_cell", "target_active_complex", "target_passive_complex"]
    index_columns.extend([column for column in optional_columns if column in score_frame.columns])
    collapsed = (
        score_frame.groupby(index_columns + ["target_voice", "prime_condition"], as_index=False)
        .agg({value_column: "mean"})
    )
    pivot = collapsed.pivot_table(
        index=index_columns,
        columns=["target_voice", "prime_condition"],
        values=value_column,
        aggfunc="mean",
    )
    pivot.columns = [
        f"logp_{target_voice}_target_given_{prime_condition}_prime_{value_column.replace('target_logprob_', '')}"
        for target_voice, prime_condition in pivot.columns
    ]
    return pivot.reset_index()


def build_item_pe_table(score_frame: pd.DataFrame) -> pd.DataFrame:
    sum_table = _pivot_scores(score_frame, "target_logprob_sum")
    mean_table = _pivot_scores(score_frame, "target_logprob_mean")
    key_columns = [
        column
        for column in [
            "item_index",
            "complexity_condition",
            "lexicality_condition",
            "target_cell",
            "target_active_complex",
            "target_passive_complex",
        ]
        if column in sum_table.columns and column in mean_table.columns
    ]
    item = sum_table.merge(mean_table, on=key_columns, how="inner")

    def col(voice: str, prime: str, metric: str) -> str:
        return f"logp_{voice}_target_given_{prime}_prime_{metric}"

    required = [
        col("active", "active", "sum"),
        col("active", "passive", "sum"),
        col("passive", "active", "sum"),
        col("passive", "passive", "sum"),
        col("active", "active", "mean"),
        col("active", "passive", "mean"),
        col("passive", "active", "mean"),
        col("passive", "passive", "mean"),
    ]
    missing = [column for column in required if column not in item.columns]
    if missing:
        raise ValueError(f"Cannot compute Sinclair PE; missing columns: {missing}")

    for metric in ["sum", "mean"]:
        item[f"pe_active_target_logprob_{metric}_same_minus_other"] = (
            item[col("active", "active", metric)] - item[col("active", "passive", metric)]
        )
        item[f"pe_passive_target_logprob_{metric}_same_minus_other"] = (
            item[col("passive", "passive", metric)] - item[col("passive", "active", metric)]
        )
        item[f"pe_logprob_{metric}_imbalance_passive_minus_active"] = (
            item[f"pe_passive_target_logprob_{metric}_same_minus_other"]
            - item[f"pe_active_target_logprob_{metric}_same_minus_other"]
        )

        if col("active", "no_prime", metric) in item.columns:
            item[f"active_target_shift_active_minus_no_prime_{metric}"] = (
                item[col("active", "active", metric)] - item[col("active", "no_prime", metric)]
            )
            item[f"passive_target_shift_passive_minus_no_prime_{metric}"] = (
                item[col("passive", "passive", metric)] - item[col("passive", "no_prime", metric)]
            )
            item[f"baseline_no_prime_passive_minus_active_logprob_{metric}"] = (
                item[col("passive", "no_prime", metric)] - item[col("active", "no_prime", metric)]
            )

        if col("active", "filler", metric) in item.columns:
            item[f"active_target_shift_active_minus_filler_{metric}"] = (
                item[col("active", "active", metric)] - item[col("active", "filler", metric)]
            )
            item[f"passive_target_shift_passive_minus_filler_{metric}"] = (
                item[col("passive", "passive", metric)] - item[col("passive", "filler", metric)]
            )
            item[f"baseline_filler_passive_minus_active_logprob_{metric}"] = (
                item[col("passive", "filler", metric)] - item[col("active", "filler", metric)]
            )

    return item.replace([np.inf, -np.inf], np.nan)


def summarize_item_pe(item_frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grouping_specs = [
        ("overall", ["lexicality_condition"]),
        ("by_complexity", ["lexicality_condition", "complexity_condition"]),
    ]
    rows: List[Dict[str, object]] = []
    for group_label, group_columns in grouping_specs:
        present_columns = [column for column in group_columns if column in item_frame.columns]
        grouped = item_frame.groupby(present_columns, dropna=False) if present_columns else [((), item_frame)]
        for group_values, subset in grouped:
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            base: Dict[str, object] = {"summary_group": group_label}
            for column, value in zip(present_columns, group_values):
                base[column] = value
            for metric in METRIC_COLUMNS:
                if metric not in subset.columns:
                    continue
                stats = one_sample_summary(subset[metric].to_numpy(dtype=float), rng=rng)
                rows.append(
                    {
                        **base,
                        "metric": metric,
                        "n_items": stats["n_items"],
                        "mean": stats["mean"],
                        "sd": stats["sd"],
                        "ci95_low": stats["bootstrap_ci95_low"],
                        "ci95_high": stats["bootstrap_ci95_high"],
                        "t_p_two_sided": stats["t_p_two_sided"],
                        "perm_p_two_sided": stats["perm_p_two_sided"],
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_rows = load_prime_target_rows(args)
    device = get_device(args.device)
    if str(args.device or "").strip().lower().startswith("cuda") and not str(device).startswith("cuda"):
        raise RuntimeError(
            "CUDA was requested for Exp4 Sinclair PE scoring, but no CUDA device is available. "
            "Switch the Colab runtime to GPU and rerun the setup/model-loading cells."
        )
    print(f"Resolved Exp4 Sinclair PE device: {device}", flush=True)
    if str(device).startswith("cuda"):
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}", flush=True)
    tokenizer, model, resolved_dtype = load_causal_lm_and_tokenizer(
        model_name=args.model_name,
        device=device,
        local_files_only=args.local_files_only,
        torch_dtype_name=args.torch_dtype,
    )
    print(f"Model first parameter device: {next(model.parameters()).device}", flush=True)

    score_frame = score_target_rows(
        frame=target_rows,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    item_frame = build_item_pe_table(score_frame)
    summary_frame = summarize_item_pe(item_frame, seed=int(args.seed))

    score_frame.to_csv(output_dir / "target_logprob_scores_exp4.csv", index=False)
    item_frame.to_csv(output_dir / "sinclair_pe_item_level_exp4.csv", index=False)
    summary_frame.to_csv(output_dir / "sinclair_pe_summary_exp4.csv", index=False)
    metadata = {
        "model_name": args.model_name,
        "model_condition": args.model_condition or output_dir.name,
        "prompt_csv": str(args.prompt_csv.resolve()),
        "output_dir": str(output_dir),
        "n_prime_target_rows": int(len(score_frame)),
        "n_item_pe_rows": int(len(item_frame)),
        "max_items": int(args.max_items) if args.max_items is not None else None,
        "prime_conditions": list(args.prime_conditions),
        "target_voices": list(args.target_voices),
        "complexity_conditions": list(args.complexity_conditions),
        "batch_size": int(args.batch_size),
        "device": device,
        "torch_dtype": str(resolved_dtype) if resolved_dtype is not None else "default",
        "local_files_only": bool(args.local_files_only),
        "seed": int(args.seed),
        "scoring_note": "Scores target_sentence immediately after prime_text; question and answer scaffold are excluded. No-prime rows score continuation token 2 onward.",
    }
    (output_dir / "run_metadata_exp4_sinclair_pe.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved Experiment 4 Sinclair-style target PE outputs to {output_dir}")


if __name__ == "__main__":
    main()
