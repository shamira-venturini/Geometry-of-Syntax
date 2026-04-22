import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
N_BOOTSTRAP = 5000
BASELINE_ORDER = ["no_prime", "filler"]
LEGACY_PRIME_MAP = {
    "no_prime_eos": "no_prime",
    "no_prime_empty": "no_prime",
    "no_demo": "no_prime",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create priming-framed summaries for any choice-based priming runs."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--run-dirs",
        nargs="*",
        default=None,
        help="Optional subdirectories inside --root. If omitted, auto-detects directories containing item_scores.csv.",
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
            "p_two_sided": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    t_stat, p_value = stats.ttest_1samp(values, popmean=0.0)
    _ = t_stat
    ci_low, ci_high = bootstrap_ci(values, rng)
    return {
        "mean": float(values.mean()),
        "p_two_sided": float(p_value),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def discover_run_dirs(root: Path, requested: List[str] | None) -> List[Path]:
    if requested:
        return [root / run_dir for run_dir in requested]
    return sorted(
        path for path in root.iterdir()
        if path.is_dir() and (path / "item_scores.csv").exists()
    )


def summarize_run(run_dir: Path, rng: np.random.Generator) -> List[Dict[str, object]]:
    item_scores = pd.read_csv(run_dir / "item_scores.csv")
    item_scores = item_scores.copy()
    item_scores["prime_condition"] = item_scores["prime_condition"].replace(LEGACY_PRIME_MAP)
    item_scores = (
        item_scores.groupby(["item_index", "prime_condition"], as_index=False)
        .agg(
            passive_choice_indicator=("passive_choice_indicator", "mean"),
            passive_minus_active_logprob=("passive_minus_active_logprob", "mean"),
        )
    )

    passive_by_condition = (
        item_scores.groupby("prime_condition", as_index=False)
        .agg(passive_choice_rate=("passive_choice_indicator", "mean"))
        .set_index("prime_condition")["passive_choice_rate"]
        .to_dict()
    )
    active_by_condition = {
        condition: 1.0 - passive_rate
        for condition, passive_rate in passive_by_condition.items()
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

    rows: List[Dict[str, object]] = []
    for baseline in baselines:
        if "active" not in available_conditions or "passive" not in available_conditions:
            continue

        active_choice = pivot_choice[baseline].to_numpy() - pivot_choice["active"].to_numpy()
        passive_choice = pivot_choice["passive"].to_numpy() - pivot_choice[baseline].to_numpy()
        imbalance_choice = passive_choice - active_choice

        active_logprob = pivot_logprob[baseline].to_numpy() - pivot_logprob["active"].to_numpy()
        passive_logprob = pivot_logprob["passive"].to_numpy() - pivot_logprob[baseline].to_numpy()
        imbalance_logprob = passive_logprob - active_logprob

        active_choice_stats = one_sample_stats(active_choice, rng)
        passive_choice_stats = one_sample_stats(passive_choice, rng)
        imbalance_choice_stats = one_sample_stats(imbalance_choice, rng)
        active_logprob_stats = one_sample_stats(active_logprob, rng)
        passive_logprob_stats = one_sample_stats(passive_logprob, rng)
        imbalance_logprob_stats = one_sample_stats(imbalance_logprob, rng)

        rows.append(
            {
                "condition": run_dir.name,
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


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rng = np.random.default_rng(args.seed)
    run_dirs = discover_run_dirs(root, args.run_dirs)

    rows: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        rows.extend(summarize_run(run_dir, rng))

    if not rows:
        raise ValueError(f"No valid run directories found under {root}")

    results = pd.DataFrame(rows).sort_values(["condition", "baseline"])
    results.to_csv(root / "priming_framed_results.csv", index=False)
    (root / "priming_framed_manifest.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "run_dirs": [run_dir.name for run_dir in run_dirs],
                "n_rows": int(len(results)),
                "seed": int(args.seed),
                "bootstrap_resamples": int(N_BOOTSTRAP),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
