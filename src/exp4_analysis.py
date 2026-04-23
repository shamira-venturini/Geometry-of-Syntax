from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .analysis import PRIME_CONDITION_ALIASES


def _normalize_prime_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["prime_condition"] = (
        result["prime_condition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(PRIME_CONDITION_ALIASES)
    )
    return result


def _safe_mean(frame: pd.DataFrame, *, prime_condition: str | None = None, target_voice: str | None = None) -> float:
    subset = frame
    if prime_condition is not None:
        subset = subset.loc[subset["prime_condition"] == prime_condition]
    if target_voice is not None:
        subset = subset.loc[subset["target_voice"] == target_voice]
    if subset.empty:
        return float("nan")
    return float(subset["is_correct"].astype(float).mean())


def _safe_foil_rate(frame: pd.DataFrame, *, prime_condition: str, target_voice: str) -> float:
    subset = frame.loc[
        (frame["prime_condition"] == prime_condition)
        & (frame["target_voice"] == target_voice)
    ]
    if subset.empty:
        return float("nan")
    return float((subset["matched_label"] == "foil").mean())


def _aggregate_accuracy(frame: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, object] = {column: value for column, value in zip(group_cols, keys)}
        row["n_items"] = int(len(group))
        row["accuracy"] = float(group["is_correct"].astype(float).mean()) if len(group) else float("nan")
        row["correct_n"] = int((group["matched_label"] == "correct").sum())
        row["foil_n"] = int((group["matched_label"] == "foil").sum())
        row["ambiguous_n"] = int((group["matched_label"] == "ambiguous").sum())
        row["other_n"] = int((group["matched_label"] == "other").sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def summary_baseline_vs_primed(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    grouping = ["model_name", "model_condition", "lexicality_condition"]

    for keys, group in frame.groupby(grouping, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, object] = {column: value for column, value in zip(grouping, keys)}

        row["n_items"] = int(len(group))

        row["accuracy_active_prime"] = _safe_mean(group, prime_condition="active")
        row["accuracy_passive_prime"] = _safe_mean(group, prime_condition="passive")
        row["accuracy_filler"] = _safe_mean(group, prime_condition="filler")
        row["accuracy_no_prime"] = _safe_mean(group, prime_condition="no_prime")

        row["priming_active_minus_passive"] = row["accuracy_active_prime"] - row["accuracy_passive_prime"]
        row["active_minus_no_prime"] = row["accuracy_active_prime"] - row["accuracy_no_prime"]
        row["passive_minus_no_prime"] = row["accuracy_passive_prime"] - row["accuracy_no_prime"]
        row["active_minus_filler"] = row["accuracy_active_prime"] - row["accuracy_filler"]
        row["passive_minus_filler"] = row["accuracy_passive_prime"] - row["accuracy_filler"]

        # Baseline comprehension-bias diagnostics.
        row["accuracy_no_prime_active_target"] = _safe_mean(
            group,
            prime_condition="no_prime",
            target_voice="active",
        )
        row["accuracy_no_prime_passive_target"] = _safe_mean(
            group,
            prime_condition="no_prime",
            target_voice="passive",
        )
        row["baseline_no_prime_passive_minus_active"] = (
            row["accuracy_no_prime_passive_target"] - row["accuracy_no_prime_active_target"]
        )

        row["accuracy_filler_active_target"] = _safe_mean(
            group,
            prime_condition="filler",
            target_voice="active",
        )
        row["accuracy_filler_passive_target"] = _safe_mean(
            group,
            prime_condition="filler",
            target_voice="passive",
        )
        row["baseline_filler_passive_minus_active"] = (
            row["accuracy_filler_passive_target"] - row["accuracy_filler_active_target"]
        )

        # Active-leaning bias proxy: foil selection rate on passive targets.
        row["foil_rate_no_prime_passive_target"] = _safe_foil_rate(
            group,
            prime_condition="no_prime",
            target_voice="passive",
        )
        row["foil_rate_filler_passive_target"] = _safe_foil_rate(
            group,
            prime_condition="filler",
            target_voice="passive",
        )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(grouping).reset_index(drop=True)


def ceiling_diagnostics(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    grouped = _aggregate_accuracy(
        frame,
        group_cols=[
            "model_name",
            "model_condition",
            "prime_condition",
            "target_voice",
            "lexicality_condition",
        ],
    )
    grouped["is_near_ceiling"] = grouped["accuracy"] >= float(threshold)
    grouped["ceiling_threshold"] = float(threshold)
    return grouped


def run_exp4_analysis(item_level: pd.DataFrame, *, ceiling_threshold: float = 0.95) -> Dict[str, pd.DataFrame]:
    frame = _normalize_prime_conditions(item_level)

    outputs: Dict[str, pd.DataFrame] = {}
    outputs["summary_by_prime_condition"] = _aggregate_accuracy(
        frame,
        group_cols=["model_name", "model_condition", "prime_condition"],
    )
    outputs["summary_by_target_voice"] = _aggregate_accuracy(
        frame,
        group_cols=["model_name", "model_condition", "target_voice"],
    )
    outputs["summary_by_lexicality"] = _aggregate_accuracy(
        frame,
        group_cols=["model_name", "model_condition", "lexicality_condition"],
    )
    outputs["summary_baseline_vs_primed"] = summary_baseline_vs_primed(frame)
    outputs["ceiling_diagnostics"] = ceiling_diagnostics(frame, threshold=ceiling_threshold)
    return outputs
