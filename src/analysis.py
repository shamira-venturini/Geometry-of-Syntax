from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

PRIME_CONDITION_ALIASES = {
    "active_prime": "active",
    "passive_prime": "passive",
    "filler_prime": "filler",
    "no_prime_eos": "no_prime",
    "no_prime_empty": "no_prime",
    "no_demo": "no_prime",
    "none": "no_prime",
}


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float


DEFAULT_PREFERENCE_MEASURES: Sequence[str] = (
    "preference_total",
    "preference_mean",
    "preference_first_token",
    "preference_second_token",
    "preference_last_token",
    "preference_divergence_token",
    "preference_aligned_mean",
    "preference_aligned_first",
    "preference_aligned_last",
)


def bootstrap_mean_ci(
    values: Sequence[float],
    n_resamples: int = 5000,
    ci: float = 95.0,
    seed: int = 13,
) -> BootstrapResult:
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return BootstrapResult(mean=float("nan"), ci_low=float("nan"), ci_high=float("nan"))

    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample = rng.choice(data, size=data.size, replace=True)
        means[index] = float(np.mean(sample))

    alpha = (100.0 - ci) / 2.0
    lower = float(np.percentile(means, alpha))
    upper = float(np.percentile(means, 100.0 - alpha))
    return BootstrapResult(mean=float(np.mean(data)), ci_low=lower, ci_high=upper)


def _describe_group(
    frame: pd.DataFrame,
    value_columns: Sequence[str],
) -> Dict[str, float]:
    payload: Dict[str, float] = {"n_rows": float(len(frame))}
    for value_column in value_columns:
        values = frame[value_column].astype(float)
        payload[f"{value_column}_mean"] = float(values.mean())
        payload[f"{value_column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        payload[f"{value_column}_se"] = (
            float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
        )
    return payload


def summarize(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    value_columns: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for keys, group_frame in frame.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: key for column, key in zip(group_cols, keys)}
        row.update(_describe_group(group_frame, value_columns=value_columns))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)


def bootstrap_summary(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    value_column: str,
    n_resamples: int,
    ci: float,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for keys, group_frame in frame.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        result = bootstrap_mean_ci(
            values=group_frame[value_column].astype(float).tolist(),
            n_resamples=n_resamples,
            ci=ci,
            seed=seed,
        )
        row = {column: key for column, key in zip(group_cols, keys)}
        row[f"{value_column}_mean"] = result.mean
        row[f"{value_column}_ci_low"] = result.ci_low
        row[f"{value_column}_ci_high"] = result.ci_high
        row["n_rows"] = int(len(group_frame))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)


def _safe_mean(frame: pd.DataFrame, condition: str, value_col: str) -> float:
    subset = frame.loc[frame["prime_condition"] == condition, value_col]
    if subset.empty:
        return float("nan")
    return float(subset.astype(float).mean())


def normalize_prime_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "prime_condition" in result.columns:
        result["prime_condition"] = (
            result["prime_condition"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(PRIME_CONDITION_ALIASES)
        )
    return result


def compute_priming_effects(
    frame: pd.DataFrame,
    preference_col: str = "preference_total",
) -> pd.DataFrame:
    grouping = ["model_name", "model_condition", "prompt_format_used", "task", "lexicality_condition"]
    rows: List[Dict[str, float]] = []

    for keys, group_frame in frame.groupby(grouping, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: key for column, key in zip(grouping, keys)}

        row["active_mean"] = _safe_mean(group_frame, "active", preference_col)
        row["passive_mean"] = _safe_mean(group_frame, "passive", preference_col)
        row["filler_mean"] = _safe_mean(group_frame, "filler", preference_col)
        row["no_prime_mean"] = _safe_mean(group_frame, "no_prime", preference_col)

        row["priming_effect_active_minus_passive"] = (
            row["active_mean"] - row["passive_mean"]
        )
        row["active_minus_no_prime"] = row["active_mean"] - row["no_prime_mean"]
        row["passive_minus_no_prime"] = row["passive_mean"] - row["no_prime_mean"]
        row["active_minus_filler"] = row["active_mean"] - row["filler_mean"]
        row["passive_minus_filler"] = row["passive_mean"] - row["filler_mean"]

        rows.append(row)

    return pd.DataFrame(rows).sort_values(grouping).reset_index(drop=True)


def paired_condition_tests(
    frame: pd.DataFrame,
    preference_col: str = "preference_total",
) -> pd.DataFrame:
    grouping = ["model_name", "model_condition", "prompt_format_used", "task", "lexicality_condition"]
    condition_pairs = [
        ("active", "passive"),
        ("active", "no_prime"),
        ("passive", "no_prime"),
        ("active", "filler"),
        ("passive", "filler"),
    ]

    rows: List[Dict[str, float]] = []
    for keys, group_frame in frame.groupby(grouping, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base_row = {column: key for column, key in zip(grouping, keys)}

        pivot = (
            group_frame.groupby(["pairing_key", "prime_condition"], as_index=False)[preference_col]
            .mean()
            .pivot(index="pairing_key", columns="prime_condition", values=preference_col)
        )

        for condition_a, condition_b in condition_pairs:
            row = dict(base_row)
            row["condition_a"] = condition_a
            row["condition_b"] = condition_b

            if condition_a not in pivot.columns or condition_b not in pivot.columns:
                row.update(
                    {
                        "n_pairs": 0,
                        "mean_diff_a_minus_b": float("nan"),
                        "t_stat": float("nan"),
                        "p_value": float("nan"),
                    }
                )
                rows.append(row)
                continue

            paired = pivot[[condition_a, condition_b]].dropna()
            if paired.empty:
                row.update(
                    {
                        "n_pairs": 0,
                        "mean_diff_a_minus_b": float("nan"),
                        "t_stat": float("nan"),
                        "p_value": float("nan"),
                    }
                )
                rows.append(row)
                continue

            diffs = paired[condition_a].astype(float) - paired[condition_b].astype(float)
            if len(paired) < 2:
                t_stat = float("nan")
                p_value = float("nan")
            else:
                t_stat, p_value = stats.ttest_rel(
                    paired[condition_a].astype(float),
                    paired[condition_b].astype(float),
                    nan_policy="omit",
                )
                t_stat = float(t_stat)
                p_value = float(p_value)

            row.update(
                {
                    "n_pairs": int(len(paired)),
                    "mean_diff_a_minus_b": float(diffs.mean()),
                    "t_stat": t_stat,
                    "p_value": p_value,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows).sort_values(grouping + ["condition_a", "condition_b"]).reset_index(drop=True)


def run_analysis(
    item_level: pd.DataFrame,
    n_bootstrap: int,
    bootstrap_ci: float,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    item_level = normalize_prime_labels(item_level)
    result: Dict[str, pd.DataFrame] = {}

    value_columns = ["preference_total", "preference_mean"]
    available_measures = [
        column
        for column in DEFAULT_PREFERENCE_MEASURES
        if column in item_level.columns
    ]

    result["summary_by_model_condition"] = summarize(
        frame=item_level,
        group_cols=["model_name", "model_condition", "prompt_format_used", "task"],
        value_columns=value_columns,
    )
    result["summary_by_prime_condition"] = summarize(
        frame=item_level,
        group_cols=["model_name", "model_condition", "prompt_format_used", "task", "prime_condition"],
        value_columns=value_columns,
    )
    result["summary_by_lexicality"] = summarize(
        frame=item_level,
        group_cols=["model_name", "model_condition", "prompt_format_used", "task", "lexicality_condition"],
        value_columns=value_columns,
    )
    result["summary_by_task"] = summarize(
        frame=item_level,
        group_cols=["model_name", "prompt_format_used", "task"],
        value_columns=value_columns,
    )

    result["bootstrap_by_prime_condition"] = bootstrap_summary(
        frame=item_level,
        group_cols=["model_name", "model_condition", "prompt_format_used", "task", "prime_condition"],
        value_column="preference_total",
        n_resamples=n_bootstrap,
        ci=bootstrap_ci,
        seed=seed,
    )

    result["priming_effects_relative_to_baseline"] = compute_priming_effects(
        frame=item_level,
        preference_col="preference_total",
    )

    result["paired_condition_tests"] = paired_condition_tests(
        frame=item_level,
        preference_col="preference_total",
    )

    if available_measures:
        result["summary_by_measure_prime_condition"] = summarize_by_measure_prime_condition(
            frame=item_level,
            measure_columns=available_measures,
        )
        result["priming_effects_all_measures"] = priming_effects_all_measures(
            frame=item_level,
            measure_columns=available_measures,
        )
        result["paired_condition_tests_all_measures"] = paired_condition_tests_all_measures(
            frame=item_level,
            measure_columns=available_measures,
        )

    token_position_summary, token_position_preference_summary = token_position_summaries(item_level=item_level)
    result["token_position_summary"] = token_position_summary
    result["token_position_preference_summary"] = token_position_preference_summary

    return result


def summarize_by_measure_prime_condition(
    frame: pd.DataFrame,
    measure_columns: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    group_cols = [
        "model_name",
        "model_condition",
        "prompt_format_used",
        "task",
        "prime_condition",
        "lexicality_condition",
    ]
    for measure in measure_columns:
        for keys, group_frame in frame.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            values = group_frame[measure].astype(float).dropna()
            row = {column: key for column, key in zip(group_cols, keys)}
            row["measure"] = measure
            row["n_rows"] = int(len(values))
            row["mean"] = float(values.mean()) if not values.empty else float("nan")
            row["std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            row["se"] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
            rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols + ["measure"]).reset_index(drop=True)


def priming_effects_all_measures(
    frame: pd.DataFrame,
    measure_columns: Sequence[str],
) -> pd.DataFrame:
    tables: List[pd.DataFrame] = []
    for measure in measure_columns:
        table = compute_priming_effects(frame=frame, preference_col=measure).copy()
        table.insert(0, "measure", measure)
        tables.append(table)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def paired_condition_tests_all_measures(
    frame: pd.DataFrame,
    measure_columns: Sequence[str],
) -> pd.DataFrame:
    tables: List[pd.DataFrame] = []
    for measure in measure_columns:
        table = paired_condition_tests(frame=frame, preference_col=measure).copy()
        table.insert(0, "measure", measure)
        tables.append(table)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def _decode_list_field(value: object) -> List[float]:
    if isinstance(value, list):
        return [float(entry) for entry in value]
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if not isinstance(value, str):
        return []
    stripped = value.strip()
    if not stripped:
        return []
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    output: List[float] = []
    for entry in payload:
        try:
            output.append(float(entry))
        except Exception:
            continue
    return output


def token_position_summaries(item_level: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    token_rows: List[Dict[str, float]] = []
    preference_rows: List[Dict[str, float]] = []
    group_base_cols = [
        "model_name",
        "model_condition",
        "prompt_format_used",
        "task",
        "prime_condition",
        "lexicality_condition",
    ]

    for row in item_level.to_dict(orient="records"):
        base = {column: row.get(column, "") for column in group_base_cols}
        a_logprobs = _decode_list_field(row.get("candidate_a_token_logprobs"))
        b_logprobs = _decode_list_field(row.get("candidate_b_token_logprobs"))
        a_label = str(row.get("candidate_a_label", "candidate_a"))
        b_label = str(row.get("candidate_b_label", "candidate_b"))

        for idx, value in enumerate(a_logprobs):
            token_rows.append(
                {
                    **base,
                    "candidate_label": a_label,
                    "token_position": int(idx),
                    "token_logprob": float(value),
                }
            )
        for idx, value in enumerate(b_logprobs):
            token_rows.append(
                {
                    **base,
                    "candidate_label": b_label,
                    "token_position": int(idx),
                    "token_logprob": float(value),
                }
            )

        aligned = min(len(a_logprobs), len(b_logprobs))
        for idx in range(aligned):
            preference_rows.append(
                {
                    **base,
                    "token_position": int(idx),
                    "preference_token_logprob": float(a_logprobs[idx] - b_logprobs[idx]),
                }
            )

    if token_rows:
        token_frame = pd.DataFrame(token_rows)
        token_summary = (
            token_frame.groupby(group_base_cols + ["candidate_label", "token_position"], as_index=False)
            .agg(
                token_logprob_mean=("token_logprob", "mean"),
                token_logprob_std=("token_logprob", "std"),
                n_tokens=("token_logprob", "size"),
            )
            .sort_values(group_base_cols + ["candidate_label", "token_position"])
            .reset_index(drop=True)
        )
    else:
        token_summary = pd.DataFrame(
            columns=group_base_cols + ["candidate_label", "token_position", "token_logprob_mean", "token_logprob_std", "n_tokens"]
        )

    if preference_rows:
        preference_frame = pd.DataFrame(preference_rows)
        preference_summary = (
            preference_frame.groupby(group_base_cols + ["token_position"], as_index=False)
            .agg(
                preference_token_logprob_mean=("preference_token_logprob", "mean"),
                preference_token_logprob_std=("preference_token_logprob", "std"),
                n_tokens=("preference_token_logprob", "size"),
            )
            .sort_values(group_base_cols + ["token_position"])
            .reset_index(drop=True)
        )
    else:
        preference_summary = pd.DataFrame(
            columns=group_base_cols + ["token_position", "preference_token_logprob_mean", "preference_token_logprob_std", "n_tokens"]
        )

    return token_summary, preference_summary
