import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


BASE_DIR = Path("behavioral_results/experiment-2/llama323binstruct/experiment-2_generation_audit_lexically_controlled")
DATASETS: Dict[str, Dict[str, Path]] = {
    "core": {
        "items": BASE_DIR / "core" / "item_generations.csv",
        "review": BASE_DIR / "core_unclassified_for_manual_review.csv",
        "output_dir": BASE_DIR / "core",
    },
    "jabberwocky": {
        "items": BASE_DIR / "jabberwocky" / "item_generations.csv",
        "review": BASE_DIR / "jabberwocky_unclassified_for_manual_review.csv",
        "output_dir": BASE_DIR / "jabberwocky",
    },
}
MERGE_KEYS = [
    "item_index",
    "prompt_column",
    "prime_condition",
    "target_active",
    "target_passive",
    "greedy_answer_first_sentence_normalized",
]
OVERRIDE_COLUMNS = [
    "generation_class_detailed",
    "generation_structure_reason",
    "argument_structure_inferred",
    "role_frame_inferred",
    "argument_inference_note",
    "generation_class_strict",
    "generation_class_lax",
]
PROMPT_ORDER = ["prompt_active", "prompt_filler", "prompt_no_prime", "prompt_passive"]
PROMPT_ORDER_MAP = {name: index for index, name in enumerate(PROMPT_ORDER)}
PROMPT_ALIAS_MAP = {
    "prompt_no_demo": "prompt_no_prime",
}
PRIME_ORDER = ["active", "filler", "no_prime", "passive"]
PRIME_ORDER_MAP = {name: index for index, name in enumerate(PRIME_ORDER)}
PRIME_ALIAS_MAP = {
    "no_demo": "no_prime",
    "no_prime_empty": "no_prime",
    "no_prime_eos": "no_prime",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay manual review labels for Experiment 2 generation audit "
            "and write reviewed summary tables."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        help="Subset of datasets to process. Defaults to all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summaries without writing reviewed output files.",
    )
    return parser.parse_args()


def sorted_frame(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["_prompt_order"] = ordered["prompt_column"].map(PROMPT_ORDER_MAP).fillna(len(PROMPT_ORDER_MAP))
    ordered["_prime_order"] = ordered["prime_condition"].map(PRIME_ORDER_MAP).fillna(len(PRIME_ORDER_MAP))
    ordered["_label"] = ordered[label_column].astype(str)
    ordered = ordered.sort_values(["_prompt_order", "_prime_order", "_label"])
    return ordered.drop(columns=["_prompt_order", "_prime_order", "_label"])


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
    return sorted_frame(summary, label_column=label_column)


def summarize_binary_rates(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["is_active"] = annotated[label_column].eq("active")
    annotated["is_passive"] = annotated[label_column].eq("passive")
    annotated["is_other"] = annotated[label_column].eq("other")
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
            manually_reviewed_rate=("manually_reviewed", "mean"),
        )
    )
    summary["_prompt_order"] = summary["prompt_column"].map(PROMPT_ORDER_MAP).fillna(len(PROMPT_ORDER_MAP))
    summary["_prime_order"] = summary["prime_condition"].map(PRIME_ORDER_MAP).fillna(len(PRIME_ORDER_MAP))
    summary = summary.sort_values(["_prompt_order", "_prime_order"]).drop(columns=["_prompt_order", "_prime_order"])
    return summary


def summarize_roles(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
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
    return sorted_frame(summary, label_column=label_column)


def require_columns(frame: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def overlay_manual_review(items_path: Path, review_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(items_path)
    require_columns(frame=frame, required=MERGE_KEYS + OVERRIDE_COLUMNS, path=items_path)
    frame = frame.copy()
    frame["prompt_column"] = frame["prompt_column"].replace(PROMPT_ALIAS_MAP)
    frame["prime_condition"] = frame["prime_condition"].replace(PRIME_ALIAS_MAP)

    review = pd.read_csv(review_path)
    require_columns(frame=review, required=MERGE_KEYS + OVERRIDE_COLUMNS, path=review_path)
    review = review.copy()
    review["prompt_column"] = review["prompt_column"].replace(PROMPT_ALIAS_MAP)
    review["prime_condition"] = review["prime_condition"].replace(PRIME_ALIAS_MAP)

    duplicated = review.duplicated(subset=MERGE_KEYS, keep=False)
    if duplicated.any():
        raise ValueError(
            f"{review_path} contains duplicate review rows for keys: "
            f"{review.loc[duplicated, MERGE_KEYS].head().to_dict(orient='records')}"
        )

    review_columns = MERGE_KEYS + OVERRIDE_COLUMNS
    review_subset = review.loc[:, review_columns].copy()
    review_subset["manually_reviewed"] = True

    merged = frame.merge(review_subset, on=MERGE_KEYS, how="left", suffixes=("", "_review"))
    merged["manually_reviewed"] = merged["manually_reviewed"].fillna(False)

    for column in OVERRIDE_COLUMNS:
        review_column = f"{column}_review"
        final_column = f"{column}_final"
        merged[final_column] = merged[review_column].combine_first(merged[column])

    merged["review_remaining_strict_final"] = merged["generation_class_strict_final"].eq("other")
    merged["review_remaining_lax_final"] = merged["generation_class_lax_final"].eq("other")
    return merged


def write_dataset_outputs(dataset_name: str, frame: pd.DataFrame, output_dir: Path, dry_run: bool) -> None:
    outputs = {
        "item_generations_reviewed.csv": frame,
        "generation_detailed_summary_reviewed.csv": summarize_counts(
            frame, label_column="generation_class_detailed_final"
        ),
        "generation_summary_reviewed_strict.csv": summarize_counts(
            frame, label_column="generation_class_strict_final"
        ),
        "generation_summary_reviewed_lax.csv": summarize_counts(
            frame, label_column="generation_class_lax_final"
        ),
        "generation_quality_summary_reviewed_strict.csv": summarize_binary_rates(
            frame, label_column="generation_class_strict_final"
        ),
        "generation_quality_summary_reviewed_lax.csv": summarize_binary_rates(
            frame, label_column="generation_class_lax_final"
        ),
        "argument_structure_summary_reviewed.csv": summarize_roles(
            frame, label_column="argument_structure_inferred_final"
        ),
        "role_frame_summary_reviewed.csv": summarize_roles(
            frame, label_column="role_frame_inferred_final"
        ),
    }

    print(f"\n=== {dataset_name}")
    print(outputs["generation_quality_summary_reviewed_strict.csv"].to_string(index=False))
    print()
    print(outputs["generation_quality_summary_reviewed_lax.csv"].to_string(index=False))

    if dry_run:
        return

    for filename, output in outputs.items():
        output.to_csv(output_dir / filename, index=False)


def main() -> None:
    args = parse_args()
    dataset_names = args.dataset if args.dataset else sorted(DATASETS)
    for dataset_name in dataset_names:
        paths = DATASETS[dataset_name]
        frame = overlay_manual_review(
            items_path=paths["items"].resolve(),
            review_path=paths["review"].resolve(),
        )
        write_dataset_outputs(
            dataset_name=dataset_name,
            frame=frame,
            output_dir=paths["output_dir"].resolve(),
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
