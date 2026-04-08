import argparse
from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "behavioral_results"
    / "experiment-1"
    / "experiment-1a"
    / "transitive_token_profiles"
    / "transitive_item_level_scores.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "behavioral_results" / "experiment-1" / "experiment-1a" / "transitive_token_profiles" / "stats"
)
PRIMARY_METRIC = "sentence_pe_mean"
SECONDARY_METRICS = [
    "sentence_pe",
    "post_divergence_pe_mean",
    "critical_word_pe_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired tests and confound-aware models for transitive priming outputs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-lmm",
        dest="run_lmm",
        action="store_true",
        default=True,
        help="Run per-condition linear mixed effects models.",
    )
    parser.add_argument(
        "--skip-lmm",
        dest="run_lmm",
        action="store_false",
        help="Skip mixed-effects models.",
    )
    return parser.parse_args()


def zscore(series: pd.Series) -> pd.Series:
    sd = float(series.std(ddof=0))
    if sd == 0.0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / sd


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    headers = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def permutation_p_values(
    diffs: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
    batch_size: int = 1000,
) -> Dict[str, float]:
    observed = float(diffs.mean())
    n = diffs.size
    if n == 0:
        return {"p_two_sided": np.nan, "p_greater_than_zero": np.nan}

    extreme_two_sided = 1
    extreme_greater = 1
    done = 0
    while done < n_permutations:
        current = min(batch_size, n_permutations - done)
        signs = rng.integers(0, 2, size=(current, n), dtype=np.int8) * 2 - 1
        perm_means = (signs * diffs).mean(axis=1)
        extreme_two_sided += int(np.sum(np.abs(perm_means) >= abs(observed)))
        extreme_greater += int(np.sum(perm_means >= observed))
        done += current

    denom = n_permutations + 1
    return {
        "p_two_sided": extreme_two_sided / denom,
        "p_greater_than_zero": extreme_greater / denom,
    }


def bootstrap_mean_ci(
    diffs: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    batch_size: int = 1000,
) -> Dict[str, float]:
    n = diffs.size
    if n == 0:
        return {"ci_low": np.nan, "ci_high": np.nan}

    means: List[np.ndarray] = []
    done = 0
    while done < n_bootstrap:
        current = min(batch_size, n_bootstrap - done)
        idx = rng.integers(0, n, size=(current, n))
        sample_means = diffs[idx].mean(axis=1)
        means.append(sample_means)
        done += current

    boot = np.concatenate(means)
    low = float(np.quantile(boot, alpha / 2))
    high = float(np.quantile(boot, 1 - alpha / 2))
    return {"ci_low": low, "ci_high": high}


def paired_stats(
    diffs: np.ndarray,
    n_permutations: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    mean_diff = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if diffs.size > 1 else np.nan
    dz = mean_diff / sd if sd and sd > 0 else np.nan
    t_stat, t_p = stats.ttest_1samp(diffs, popmean=0.0)
    perm = permutation_p_values(diffs, n_permutations, rng)
    boot = bootstrap_mean_ci(diffs, n_bootstrap, rng)
    return {
        "n_items": int(diffs.size),
        "mean_diff": mean_diff,
        "sd_diff": sd,
        "effect_size_dz": dz,
        "t_stat": float(t_stat),
        "t_p_two_sided": float(t_p),
        "perm_p_two_sided": perm["p_two_sided"],
        "perm_p_greater_than_zero": perm["p_greater_than_zero"],
        "bootstrap_ci95_low": boot["ci_low"],
        "bootstrap_ci95_high": boot["ci_high"],
    }


def prepare_wide(frame: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "condition",
        "item_index",
        "target_structure",
        "sentence_pe_mean",
        "sentence_pe",
        "post_divergence_pe_mean",
        "critical_word_pe_mean",
        "target_length",
        "critical_word_token_count",
    ]
    missing = [col for col in keep_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    subset = frame[keep_cols].copy()
    counts = subset.groupby(["condition", "item_index"])["target_structure"].nunique()
    invalid = counts[counts != 2]
    if not invalid.empty:
        raise ValueError(f"Expected exactly two structures per item; found violations for {len(invalid)} items.")

    wide = subset.pivot(
        index=["condition", "item_index"],
        columns="target_structure",
        values=[
            "sentence_pe_mean",
            "sentence_pe",
            "post_divergence_pe_mean",
            "critical_word_pe_mean",
            "target_length",
            "critical_word_token_count",
        ],
    )
    wide.columns = [f"{metric}_{structure}" for metric, structure in wide.columns]
    wide = wide.reset_index()
    return wide


def run_primary_and_secondary_tests(
    wide: pd.DataFrame,
    n_permutations: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    metrics = [PRIMARY_METRIC] + SECONDARY_METRICS
    rows: List[Dict[str, float]] = []
    for condition, cond_df in wide.groupby("condition"):
        for metric in metrics:
            diffs = (
                cond_df[f"{metric}_passive"].to_numpy(dtype=float)
                - cond_df[f"{metric}_active"].to_numpy(dtype=float)
            )
            stats_row = paired_stats(
                diffs=diffs,
                n_permutations=n_permutations,
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            stats_row["condition"] = condition
            stats_row["metric"] = metric
            rows.append(stats_row)
    return pd.DataFrame(rows)


def run_delta_regression(wide: pd.DataFrame) -> pd.DataFrame:
    delta = pd.DataFrame(
        {
            "condition": wide["condition"],
            "item_index": wide["item_index"],
            "delta_sentence_pe_mean": wide["sentence_pe_mean_passive"] - wide["sentence_pe_mean_active"],
            "delta_target_length": wide["target_length_passive"] - wide["target_length_active"],
            "delta_critical_word_token_count": (
                wide["critical_word_token_count_passive"] - wide["critical_word_token_count_active"]
            ),
        }
    )
    delta["z_delta_target_length"] = zscore(delta["delta_target_length"])
    delta["z_delta_critical_word_token_count"] = zscore(delta["delta_critical_word_token_count"])

    model = smf.ols(
        "delta_sentence_pe_mean ~ C(condition) + z_delta_target_length + z_delta_critical_word_token_count",
        data=delta,
    ).fit(cov_type="HC3")

    coef = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    coef["model"] = "delta_ols_hc3"
    return coef


def run_lmm_per_condition(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for condition, cond_df in frame.groupby("condition"):
        model_df = cond_df.copy()
        model_df["target_structure"] = pd.Categorical(
            model_df["target_structure"],
            categories=["active", "passive"],
            ordered=True,
        )
        model_df["z_target_length"] = zscore(model_df["target_length"])
        model_df["z_critical_word_token_count"] = zscore(model_df["critical_word_token_count"])

        target_term = "C(target_structure)[T.passive]"
        try:
            model = smf.mixedlm(
                "sentence_pe_mean ~ C(target_structure) + z_target_length + z_critical_word_token_count",
                data=model_df,
                groups=model_df["item_index"],
            )
            result = None
            for method in ("lbfgs", "cg", "powell"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    candidate = model.fit(reml=False, method=method, maxiter=500, disp=False)
                result = candidate
                if bool(candidate.converged):
                    break
            assert result is not None
            params = result.params
            bse = result.bse
            pvalues = result.pvalues
            rows.append(
                {
                    "condition": condition,
                    "term": target_term,
                    "coef": float(params.get(target_term, np.nan)),
                    "std_err": float(bse.get(target_term, np.nan)),
                    "p_value": float(pvalues.get(target_term, np.nan)),
                    "aic": float(result.aic),
                    "bic": float(result.bic),
                    "converged": bool(result.converged),
                    "n_obs": int(result.nobs),
                    "model": "lmm_random_intercept",
                    "error": "",
                }
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            rows.append(
                {
                    "condition": condition,
                    "term": target_term,
                    "coef": np.nan,
                    "std_err": np.nan,
                    "p_value": np.nan,
                    "aic": np.nan,
                    "bic": np.nan,
                    "converged": False,
                    "n_obs": int(len(model_df)),
                    "model": "lmm_random_intercept",
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def write_markdown_report(
    output_path: Path,
    paired_df: pd.DataFrame,
    delta_coef: pd.DataFrame,
    lmm_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Transitive Statistical Analysis")
    lines.append("")
    lines.append("## Primary and Secondary Paired Effects")
    lines.append("")
    lines.append(
        markdown_table(
            paired_df[
                [
                    "condition",
                    "metric",
                    "n_items",
                    "mean_diff",
                    "bootstrap_ci95_low",
                    "bootstrap_ci95_high",
                    "perm_p_greater_than_zero",
                    "effect_size_dz",
                ]
            ]
        )
    )
    lines.append("")
    lines.append("## Confound-Aware Delta Regression (HC3)")
    lines.append("")
    lines.append(markdown_table(delta_coef))
    lines.append("")
    lines.append("## Per-Condition LMM Robustness")
    lines.append("")
    if lmm_df.empty:
        lines.append("LMM was skipped.")
    else:
        lines.append(markdown_table(lmm_df))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Primary effect is passive minus active on sentence_pe_mean.")
    lines.append("- Permutation p-values are sign-flip tests on paired item-level differences.")
    lines.append("- Bootstrap confidence intervals are percentile 95% intervals for mean paired difference.")
    lines.append("- Delta regression controls for target length and critical-word token-count asymmetries.")
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input)
    wide = prepare_wide(frame)

    paired_df = run_primary_and_secondary_tests(
        wide=wide,
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
        rng=rng,
    ).sort_values(["metric", "condition"])

    delta_coef = run_delta_regression(wide)

    lmm_df = pd.DataFrame()
    if args.run_lmm:
        lmm_df = run_lmm_per_condition(frame)

    paired_path = args.output_dir / "paired_effects.csv"
    delta_path = args.output_dir / "delta_regression_coefficients.csv"
    lmm_path = args.output_dir / "lmm_condition_coefficients.csv"
    report_path = args.output_dir / "analysis_report.md"

    paired_df.to_csv(paired_path, index=False)
    delta_coef.to_csv(delta_path, index=False)
    lmm_df.to_csv(lmm_path, index=False)
    write_markdown_report(report_path, paired_df, delta_coef, lmm_df)

    print(f"Wrote paired effects: {paired_path}")
    print(f"Wrote delta regression coefficients: {delta_path}")
    if args.run_lmm:
        print(f"Wrote LMM coefficients: {lmm_path}")
    else:
        print("Skipped LMM coefficients.")
    print(f"Wrote markdown report: {report_path}")


if __name__ == "__main__":
    main()
