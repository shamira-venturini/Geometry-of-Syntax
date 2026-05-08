#!/usr/bin/env python3
"""Build/apply one deduplicated manual-review file for Experiment 2 generations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "behavioral_results"
DEFAULT_REVIEW_CSV = REPO_ROOT / "behavioral_results/experiment-2/exp2_master_generation_review.csv"
DEFAULT_APPLY_SUMMARY = REPO_ROOT / "behavioral_results/experiment-2/exp2_master_generation_review_apply_summary.csv"

KEY_COLUMNS = ["greedy_answer_first_sentence_normalized"]
REVIEW_COLUMNS = [
    "generation_class_detailed",
    "generation_structure_reason",
    "argument_structure_inferred",
    "role_frame_inferred",
    "argument_inference_note",
    "generation_class_strict",
    "generation_class_lax",
]
PROMPT_ORDER = ["prompt_active", "prompt_filler", "prompt_no_prime", "prompt_passive"]
PRIME_ORDER = ["active", "filler", "no_prime", "passive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Create a deduplicated master review CSV.")
    build.add_argument(
        "--results-root",
        action="append",
        type=Path,
        default=None,
        help="Root to search recursively. Can be passed more than once.",
    )
    build.add_argument("--output-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    build.add_argument(
        "--human-csv",
        type=Path,
        default=None,
        help="Optional compact review CSV with only human-editable columns.",
    )
    build.add_argument(
        "--include-reviewed",
        action="store_true",
        help="Include rows that are already non-other. Default keeps only rows needing review.",
    )
    build.add_argument(
        "--path-contains",
        action="append",
        default=["experiment-2"],
        help="Only include item_generations.csv paths containing this string. Repeat for OR matching.",
    )
    build.add_argument(
        "--xlsx",
        type=Path,
        default=None,
        help="Optional compact Excel review workbook.",
    )

    apply = subparsers.add_parser("apply", help="Apply a completed master review CSV to item files.")
    apply.add_argument(
        "--results-root",
        action="append",
        type=Path,
        default=None,
        help="Root to search recursively. Can be passed more than once.",
    )
    apply.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    apply.add_argument("--summary-csv", type=Path, default=DEFAULT_APPLY_SUMMARY)
    apply.add_argument(
        "--path-contains",
        action="append",
        default=["experiment-2"],
        help="Only include item_generations.csv paths containing this string. Repeat for OR matching.",
    )
    apply.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any row needing review has no matching master-review key.",
    )
    apply.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be updated without writing reviewed outputs.",
    )
    return parser.parse_args()


def search_roots(values: Sequence[Path] | None) -> list[Path]:
    roots = list(values or [DEFAULT_RESULTS_ROOT])
    return [root.resolve() for root in roots]


def item_generation_paths(roots: Sequence[Path], path_contains: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("item_generations.csv"):
            text = str(path)
            if path_contains and not any(token in text for token in path_contains):
                continue
            paths.append(path.resolve())
    return sorted(set(paths))


def normalize_text(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def review_key_from_values(generated_normalized: object) -> str:
    payload = normalize_text(generated_normalized)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def add_review_key(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot create review key; missing columns: {missing}")
    out = frame.copy()
    out["review_key"] = [
        review_key_from_values(generated)
        for generated in out["greedy_answer_first_sentence_normalized"]
    ]
    return out


def needs_review_mask(frame: pd.DataFrame) -> pd.Series:
    def as_bool_series(series: pd.Series) -> pd.Series:
        if series.dtype == bool:
            return series.fillna(False)
        return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

    strict = (
        as_bool_series(frame["review_remaining_strict"])
        if "review_remaining_strict" in frame.columns
        else frame["generation_class_strict"].astype(str).eq("other")
    )
    lax = (
        as_bool_series(frame["review_remaining_lax"])
        if "review_remaining_lax" in frame.columns
        else frame["generation_class_lax"].astype(str).eq("other")
    )
    return strict | lax


def infer_model_run(path: Path) -> tuple[str, str]:
    parts = path.parts
    model_run = ""
    dataset = path.parent.name
    for part in parts:
        if part.startswith("colab_"):
            model_run = part
            break
    if not model_run:
        model_run = path.parents[2].name if len(path.parents) >= 3 else "unknown_model_run"
    return model_run, dataset


def read_item_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(KEY_COLUMNS + REVIEW_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame = add_review_key(frame)
    model_run, dataset = infer_model_run(path)
    frame["source_item_path"] = str(path)
    frame["source_model_run"] = model_run
    frame["source_dataset"] = dataset
    return frame


def compact_join(values: Iterable[object], limit: int = 12) -> str:
    unique = []
    seen = set()
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            unique.append(text)
        if len(unique) >= limit:
            break
    return ";".join(unique)


def build_master_review(args: argparse.Namespace) -> None:
    paths = item_generation_paths(
        roots=search_roots(args.results_root),
        path_contains=args.path_contains,
    )
    if not paths:
        raise FileNotFoundError("No item_generations.csv files found.")

    frames = []
    for path in paths:
        frame = read_item_file(path)
        if not args.include_reviewed:
            frame = frame.loc[needs_review_mask(frame)].copy()
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise ValueError("No rows requiring review were found.")

    all_rows = pd.concat(frames, ignore_index=True)
    grouping = ["review_key"] + KEY_COLUMNS
    first_cols = [
        "greedy_answer_first_sentence",
        "target_active",
        "target_passive",
        "generation_class",
        "generation_class_detailed",
        "generation_voice_auto",
        "generation_structure_reason",
        "argument_structure_inferred",
        "role_frame_inferred",
        "argument_inference_note",
        "generation_class_strict",
        "generation_class_lax",
    ]
    aggregations = {
        column: (column, "first")
        for column in first_cols
        if column in all_rows.columns
    }
    master = (
        all_rows.groupby(grouping, dropna=False)
        .agg(
            occurrence_count=("review_key", "size"),
            model_runs=("source_model_run", compact_join),
            datasets=("source_dataset", compact_join),
            prompt_columns=("prompt_column", compact_join),
            prime_conditions=("prime_condition", compact_join),
            item_indices=("item_index", compact_join),
            target_active_examples=("target_active", compact_join),
            target_passive_examples=("target_passive", compact_join),
            source_paths=("source_item_path", compact_join),
            **aggregations,
        )
        .reset_index()
    )

    for column in REVIEW_COLUMNS:
        if column in master.columns:
            master[f"reviewed_{column}"] = master[column]
        else:
            master[f"reviewed_{column}"] = ""

    master["review_notes"] = ""
    preferred_order = [
        "review_key",
        "greedy_answer_first_sentence_normalized",
        "greedy_answer_first_sentence",
        "occurrence_count",
        "generation_class_strict",
        "generation_class_lax",
        "generation_class_detailed",
        "generation_voice_auto",
        "reviewed_generation_class_strict",
        "reviewed_generation_class_lax",
        "reviewed_generation_class_detailed",
        "review_notes",
        "model_runs",
        "datasets",
        "prompt_columns",
        "prime_conditions",
        "item_indices",
        "target_active_examples",
        "target_passive_examples",
        "generation_structure_reason",
        "argument_structure_inferred",
        "role_frame_inferred",
        "argument_inference_note",
        "reviewed_generation_structure_reason",
        "reviewed_argument_structure_inferred",
        "reviewed_role_frame_inferred",
        "reviewed_argument_inference_note",
        "target_active",
        "target_passive",
        "source_paths",
    ]
    ordered_columns = [column for column in preferred_order if column in master.columns]
    ordered_columns.extend([column for column in master.columns if column not in ordered_columns])
    master = master[ordered_columns]
    sort_cols = [column for column in ["datasets", "occurrence_count", "greedy_answer_first_sentence_normalized"] if column in master.columns]
    master = master.sort_values(sort_cols, ascending=[True, False, True][: len(sort_cols)])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.output_csv, index=False)

    human_columns = [
        "review_key",
        "greedy_answer_first_sentence_normalized",
        "greedy_answer_first_sentence",
        "occurrence_count",
        "model_runs",
        "datasets",
        "prompt_columns",
        "prime_conditions",
        "generation_class_strict",
        "generation_class_lax",
        "generation_class_detailed",
        "generation_voice_auto",
        "reviewed_generation_class_strict",
        "reviewed_generation_class_lax",
        "reviewed_generation_class_detailed",
        "review_notes",
    ]
    human = master[[column for column in human_columns if column in master.columns]].copy()
    human_csv = args.human_csv
    if human_csv is None:
        human_csv = args.output_csv.with_name(args.output_csv.stem + "_HUMAN_REVIEW.csv")
    human_csv.parent.mkdir(parents=True, exist_ok=True)
    human.to_csv(human_csv, index=False)

    if args.xlsx is not None:
        args.xlsx.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(args.xlsx, engine="openpyxl") as writer:
            human.to_excel(writer, index=False, sheet_name="review")
            worksheet = writer.sheets["review"]
            worksheet.freeze_panes = "A2"
            widths = {
                "A": 18,
                "B": 55,
                "C": 55,
                "D": 16,
                "E": 35,
                "F": 35,
                "G": 28,
                "H": 28,
                "I": 22,
                "J": 22,
                "K": 34,
                "L": 20,
                "M": 26,
                "N": 26,
                "O": 36,
                "P": 36,
            }
            for column, width in widths.items():
                worksheet.column_dimensions[column].width = width

    summary = {
        "item_generation_files_scanned": len(paths),
        "rows_requiring_review": int(len(all_rows)),
        "deduplicated_review_rows": int(len(master)),
        "deduplication_key": "greedy_answer_first_sentence_normalized",
        "output_csv": str(args.output_csv.resolve()),
        "human_csv": str(human_csv.resolve()),
        "xlsx": str(args.xlsx.resolve()) if args.xlsx else None,
    }
    summary_path = args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def summarize_counts(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    counts = (
        frame.groupby(["prompt_column", "prime_condition", label_column], as_index=False)
        .agg(n_items=("item_index", "count"))
    )
    totals = (
        frame.groupby(["prompt_column", "prime_condition"], as_index=False)
        .agg(total_items=("item_index", "count"))
    )
    summary = counts.merge(totals, on=["prompt_column", "prime_condition"], how="left")
    summary["share"] = summary["n_items"] / summary["total_items"]
    return sort_prompt_prime(summary, label_column)


def sort_prompt_prime(frame: pd.DataFrame, label_column: str | None = None) -> pd.DataFrame:
    out = frame.copy()
    out["_prompt_order"] = out["prompt_column"].map({v: i for i, v in enumerate(PROMPT_ORDER)}).fillna(99)
    out["_prime_order"] = out["prime_condition"].map({v: i for i, v in enumerate(PRIME_ORDER)}).fillna(99)
    sort_cols = ["_prompt_order", "_prime_order"]
    if label_column and label_column in out.columns:
        sort_cols.append(label_column)
    out = out.sort_values(sort_cols)
    return out.drop(columns=["_prompt_order", "_prime_order"])


def summarize_binary_rates(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["is_active"] = annotated[label_column].astype(str).eq("active")
    annotated["is_passive"] = annotated[label_column].astype(str).eq("passive")
    annotated["is_other"] = annotated[label_column].astype(str).eq("other")
    annotated["is_congruent"] = (
        ((annotated["prime_condition"] == "active") & annotated["is_active"])
        | ((annotated["prime_condition"] == "passive") & annotated["is_passive"])
    )
    summary = (
        annotated.groupby(["prompt_column", "prime_condition"], as_index=False)
        .agg(
            n_items=("item_index", "count"),
            active_rate=("is_active", "mean"),
            passive_rate=("is_passive", "mean"),
            other_rate=("is_other", "mean"),
            congruent_rate=("is_congruent", "mean"),
            master_reviewed_rate=("master_reviewed", "mean"),
        )
    )
    return sort_prompt_prime(summary)


def summarize_binary_rates_by_role_order(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    if "role_order" not in frame.columns:
        return pd.DataFrame()
    annotated = frame.copy()
    annotated["is_active"] = annotated[label_column].astype(str).eq("active")
    annotated["is_passive"] = annotated[label_column].astype(str).eq("passive")
    annotated["is_other"] = annotated[label_column].astype(str).eq("other")
    annotated["is_congruent"] = (
        ((annotated["prime_condition"] == "active") & annotated["is_active"])
        | ((annotated["prime_condition"] == "passive") & annotated["is_passive"])
    )
    summary = (
        annotated.groupby(["prompt_column", "prime_condition", "role_order"], as_index=False)
        .agg(
            n_items=("item_index", "count"),
            active_rate=("is_active", "mean"),
            passive_rate=("is_passive", "mean"),
            other_rate=("is_other", "mean"),
            congruent_rate=("is_congruent", "mean"),
            master_reviewed_rate=("master_reviewed", "mean"),
        )
    )
    return sort_prompt_prime(summary)


def write_reviewed_outputs(frame: pd.DataFrame, output_dir: Path, dry_run: bool) -> None:
    outputs = {
        "item_generations_reviewed.csv": frame,
        "generation_detailed_summary_reviewed.csv": summarize_counts(frame, "generation_class_detailed_final"),
        "generation_summary_reviewed_strict.csv": summarize_counts(frame, "generation_class_strict_final"),
        "generation_summary_reviewed_lax.csv": summarize_counts(frame, "generation_class_lax_final"),
        "generation_quality_summary_reviewed_strict.csv": summarize_binary_rates(frame, "generation_class_strict_final"),
        "generation_quality_summary_reviewed_lax.csv": summarize_binary_rates(frame, "generation_class_lax_final"),
        "argument_structure_summary_reviewed.csv": summarize_counts(frame, "argument_structure_inferred_final"),
        "role_frame_summary_reviewed.csv": summarize_counts(frame, "role_frame_inferred_final"),
    }
    by_role_strict = summarize_binary_rates_by_role_order(frame, "generation_class_strict_final")
    by_role_lax = summarize_binary_rates_by_role_order(frame, "generation_class_lax_final")
    if not by_role_strict.empty:
        outputs["generation_quality_by_role_order_reviewed_strict.csv"] = by_role_strict
    if not by_role_lax.empty:
        outputs["generation_quality_by_role_order_reviewed_lax.csv"] = by_role_lax

    if dry_run:
        return
    for filename, output in outputs.items():
        output.to_csv(output_dir / filename, index=False)


def apply_master_review(args: argparse.Namespace) -> None:
    if not args.review_csv.exists():
        raise FileNotFoundError(f"Missing master review CSV: {args.review_csv}")
    review = pd.read_csv(args.review_csv).fillna("")
    if "review_key" not in review.columns:
        review = add_review_key(review)
    required_review = ["review_key"]
    missing = [column for column in required_review if column not in review.columns]
    if missing:
        raise ValueError(f"{args.review_csv} is missing required review columns: {missing}")
    for column in REVIEW_COLUMNS:
        reviewed_column = f"reviewed_{column}"
        if reviewed_column not in review.columns:
            review[reviewed_column] = ""
    required_review = ["review_key"] + [f"reviewed_{column}" for column in REVIEW_COLUMNS]

    duplicate_keys = review["review_key"].duplicated(keep=False)
    if duplicate_keys.any():
        raise ValueError(
            "Master review contains duplicate review_key values: "
            f"{review.loc[duplicate_keys, 'review_key'].head().tolist()}"
        )

    review_subset = review[required_review].copy()
    paths = item_generation_paths(
        roots=search_roots(args.results_root),
        path_contains=args.path_contains,
    )
    if not paths:
        raise FileNotFoundError("No item_generations.csv files found.")

    summary_rows = []
    for path in paths:
        frame = read_item_file(path)
        need_mask = needs_review_mask(frame)
        merged = frame.merge(review_subset, on="review_key", how="left")
        matched_key_mask = merged["reviewed_generation_class_strict"].astype(str).str.len().gt(0)
        matched_review_mask = need_mask & matched_key_mask
        missing_needed = int((need_mask & ~matched_key_mask).sum())
        if args.strict and missing_needed:
            raise ValueError(f"{path} has {missing_needed} review-needed rows missing from master review.")

        merged["master_reviewed"] = matched_review_mask
        for column in REVIEW_COLUMNS:
            reviewed_col = f"reviewed_{column}"
            final_col = f"{column}_final"
            reviewed_values = merged[reviewed_col].where(matched_review_mask).replace("", pd.NA)
            merged[final_col] = reviewed_values.combine_first(merged[column])

        merged["review_remaining_strict_final"] = merged["generation_class_strict_final"].eq("other")
        merged["review_remaining_lax_final"] = merged["generation_class_lax_final"].eq("other")

        output_dir = path.parent
        write_reviewed_outputs(merged, output_dir=output_dir, dry_run=args.dry_run)

        model_run, dataset = infer_model_run(path)
        summary_rows.append(
            {
                "item_generations_path": str(path),
                "model_run": model_run,
                "dataset": dataset,
                "n_rows": int(len(merged)),
                "n_review_needed_original": int(need_mask.sum()),
                "n_master_review_matches": int(matched_review_mask.sum()),
                "n_review_needed_missing_from_master": missing_needed,
                "n_strict_other_final": int(merged["review_remaining_strict_final"].sum()),
                "n_lax_other_final": int(merged["review_remaining_lax_final"].sum()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    if not args.dry_run:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_csv, index=False)
    print(summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build_master_review(args)
    elif args.command == "apply":
        apply_master_review(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
