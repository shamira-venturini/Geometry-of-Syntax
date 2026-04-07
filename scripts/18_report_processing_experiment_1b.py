import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "behavioral_results" / "processing_experiment_1b_gpt2large_v2"
BASELINE_ORDER = ["no_prime_eos", "no_prime_empty", "filler", "no_prime"]
N_BOOTSTRAP = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create priming-framed summaries for processing Experiment 1b."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=["processing_1b_core_core", "processing_1b_jabberwocky_jabberwocky"],
        help="Run subdirectories inside --root.",
    )
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> List[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True)
    means = samples.mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def one_sample_stats(values: np.ndarray, rng: np.random.Generator) -> Dict[str, float]:
    if len(values) <= 1:
        return {
            "mean": float(values.mean()) if len(values) else float("nan"),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "t_stat": float("nan"),
            "p_two_sided": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    t_stat, p = stats.ttest_1samp(values, popmean=0.0)
    ci_low, ci_high = bootstrap_ci(values, rng)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "t_stat": float(t_stat),
        "p_two_sided": float(p),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def infer_condition_label(run_dir_name: str) -> str:
    lowered = run_dir_name.lower()
    if "jabberwocky" in lowered:
        return "jabberwocky_jabberwocky"
    if "core" in lowered:
        return "core_core"
    return run_dir_name


def summarize_run(run_dir: Path, condition_label: str, rng: np.random.Generator) -> List[Dict[str, object]]:
    item_scores = pd.read_csv(run_dir / "item_scores.csv")
    summary = pd.read_csv(run_dir / "summary.csv")

    passive_by_condition = {
        row["prime_condition"]: float(row["passive_choice_rate"])
        for row in summary.to_dict(orient="records")
    }
    active_by_condition = {
        key: 1.0 - value
        for key, value in passive_by_condition.items()
    }

    pivot_choice = item_scores.pivot(
        index="item_index",
        columns="prime_condition",
        values="passive_choice_indicator",
    )
    pivot_logprob = item_scores.pivot(
        index="item_index",
        columns="prime_condition",
        values="passive_minus_active_logprob",
    )

    available_conditions = set(pivot_choice.columns.tolist())
    baselines = [name for name in BASELINE_ORDER if name in available_conditions]
    if not baselines and "no_prime" in available_conditions:
        baselines = ["no_prime"]

    rows: List[Dict[str, object]] = []
    for baseline in baselines:
        if "active" not in available_conditions or "passive" not in available_conditions:
            continue

        active_priming_choice = pivot_choice[baseline].to_numpy() - pivot_choice["active"].to_numpy()
        passive_priming_choice = pivot_choice["passive"].to_numpy() - pivot_choice[baseline].to_numpy()
        imbalance_choice = passive_priming_choice - active_priming_choice

        active_priming_logprob = pivot_logprob[baseline].to_numpy() - pivot_logprob["active"].to_numpy()
        passive_priming_logprob = pivot_logprob["passive"].to_numpy() - pivot_logprob[baseline].to_numpy()
        imbalance_logprob = passive_priming_logprob - active_priming_logprob

        active_choice_stats = one_sample_stats(active_priming_choice, rng)
        passive_choice_stats = one_sample_stats(passive_priming_choice, rng)
        imbalance_choice_stats = one_sample_stats(imbalance_choice, rng)
        active_logprob_stats = one_sample_stats(active_priming_logprob, rng)
        passive_logprob_stats = one_sample_stats(passive_priming_logprob, rng)
        imbalance_logprob_stats = one_sample_stats(imbalance_logprob, rng)

        rows.append(
            {
                "condition": condition_label,
                "baseline": baseline,
                "n_items": int(len(pivot_choice)),
                "active_prime_passive_choice_rate": passive_by_condition["active"],
                "passive_prime_passive_choice_rate": passive_by_condition["passive"],
                "baseline_passive_choice_rate": passive_by_condition[baseline],
                "baseline_active_choice_rate": active_by_condition[baseline],
                "active_choice_priming": active_choice_stats["mean"],
                "active_choice_priming_ci95_low": active_choice_stats["ci95_low"],
                "active_choice_priming_ci95_high": active_choice_stats["ci95_high"],
                "active_choice_priming_p": active_choice_stats["p_two_sided"],
                "passive_choice_priming": passive_choice_stats["mean"],
                "passive_choice_priming_ci95_low": passive_choice_stats["ci95_low"],
                "passive_choice_priming_ci95_high": passive_choice_stats["ci95_high"],
                "passive_choice_priming_p": passive_choice_stats["p_two_sided"],
                "imbalance_choice_passive_minus_active": imbalance_choice_stats["mean"],
                "imbalance_choice_ci95_low": imbalance_choice_stats["ci95_low"],
                "imbalance_choice_ci95_high": imbalance_choice_stats["ci95_high"],
                "imbalance_choice_p": imbalance_choice_stats["p_two_sided"],
                "active_logprob_priming": active_logprob_stats["mean"],
                "active_logprob_priming_ci95_low": active_logprob_stats["ci95_low"],
                "active_logprob_priming_ci95_high": active_logprob_stats["ci95_high"],
                "active_logprob_priming_p": active_logprob_stats["p_two_sided"],
                "passive_logprob_priming": passive_logprob_stats["mean"],
                "passive_logprob_priming_ci95_low": passive_logprob_stats["ci95_low"],
                "passive_logprob_priming_ci95_high": passive_logprob_stats["ci95_high"],
                "passive_logprob_priming_p": passive_logprob_stats["p_two_sided"],
                "imbalance_logprob_passive_minus_active": imbalance_logprob_stats["mean"],
                "imbalance_logprob_ci95_low": imbalance_logprob_stats["ci95_low"],
                "imbalance_logprob_ci95_high": imbalance_logprob_stats["ci95_high"],
                "imbalance_logprob_p": imbalance_logprob_stats["p_two_sided"],
            }
        )

    return rows


def write_markdown(rows_df: pd.DataFrame, root: Path) -> None:
    lines = [
        "# Experiment 1b Priming-Framed Summary",
        "",
        "This table decomposes priming into active-choice and passive-choice effects relative to each baseline.",
        "",
        "```csv",
        rows_df.to_csv(index=False).strip(),
        "```",
    ]
    (root / "priming_framed_report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rng = np.random.default_rng(args.seed)

    all_rows: List[Dict[str, object]] = []
    run_meta: List[Dict[str, str]] = []
    for run_name in args.run_dirs:
        run_dir = root / run_name
        if not run_dir.exists():
            continue
        if not (run_dir / "item_scores.csv").exists():
            continue
        condition_label = infer_condition_label(run_name)
        run_meta.append({"run_dir": run_name, "condition": condition_label})
        all_rows.extend(summarize_run(run_dir=run_dir, condition_label=condition_label, rng=rng))

    if not all_rows:
        raise ValueError(f"No valid run directories found under {root}")

    results = pd.DataFrame(all_rows).sort_values(["condition", "baseline"])
    results.to_csv(root / "priming_framed_results.csv", index=False)
    write_markdown(results, root=root)
    (root / "priming_framed_manifest.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "run_dirs": run_meta,
                "n_rows": int(len(results)),
                "seed": int(args.seed),
                "bootstrap_resamples": int(N_BOOTSTRAP),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
