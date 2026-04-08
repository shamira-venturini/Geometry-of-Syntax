import argparse
from pathlib import Path

import pandas as pd

from production_priming_common import REPO_ROOT


DEFAULT_OUTPUT_DIR = REPO_ROOT / "behavioral_results" / "experiment-2" / "demo_prompt_wording_ablation_core"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank demo-prompt wording combinations by active-vs-passive prime separation."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    comparison = pd.read_csv(output_dir / "comparison.csv")
    stats = pd.read_csv(output_dir / "stats.csv")
    summary = pd.read_csv(output_dir / "summary.csv")

    active_passive = comparison[
        (comparison["condition_a"] == "active") & (comparison["condition_b"] == "passive")
    ][
        [
            "prompt_template",
            "passive_choice_rate_a",
            "passive_choice_rate_b",
            "passive_choice_rate_diff_b_minus_a",
            "mean_logprob_a",
            "mean_logprob_b",
            "mean_logprob_diff_b_minus_a",
        ]
    ].copy()

    passive_choice_stats = stats[
        (stats["metric"] == "passive_choice_delta")
        & (stats["condition_a"] == "active")
        & (stats["condition_b"] == "passive")
    ][["prompt_template", "t_p_two_sided", "mcnemar_p_exact"]].rename(
        columns={"t_p_two_sided": "passive_choice_t_p", "mcnemar_p_exact": "passive_choice_mcnemar_p"}
    )

    logprob_stats = stats[
        (stats["metric"] == "logprob_delta")
        & (stats["condition_a"] == "active")
        & (stats["condition_b"] == "passive")
    ][["prompt_template", "t_p_two_sided", "effect_size_dz"]].rename(
        columns={"t_p_two_sided": "logprob_t_p", "effect_size_dz": "logprob_effect_size_dz"}
    )

    no_demo_rows = summary[summary["prime_condition"] == "no_demo"][
        ["prompt_template", "passive_choice_rate", "mean_passive_minus_active_logprob"]
    ].rename(
        columns={
            "passive_choice_rate": "no_demo_passive_choice_rate",
            "mean_passive_minus_active_logprob": "no_demo_mean_logprob",
        }
    )

    ranking = active_passive.merge(passive_choice_stats, on="prompt_template", how="left")
    ranking = ranking.merge(logprob_stats, on="prompt_template", how="left")
    ranking = ranking.merge(no_demo_rows, on="prompt_template", how="left")
    ranking = ranking.sort_values(
        ["passive_choice_rate_diff_b_minus_a", "mean_logprob_diff_b_minus_a"],
        ascending=[False, False],
    )

    ranking.to_csv(output_dir / "wording_ranking.csv", index=False)
    print(ranking.to_csv(index=False).strip())


if __name__ == "__main__":
    main()
