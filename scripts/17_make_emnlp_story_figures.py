import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures" / "emnlp_story"
DEFAULT_EXP1A_SUMMARY = (
    REPO_ROOT
    / "behavioral_results"
    / "experiment-1"
    / "experiment-1a"
    / "transitive_token_profiles"
    / "transitive_item_summary.csv"
)
DEFAULT_EXP1A_PAIRED = (
    REPO_ROOT
    / "behavioral_results"
    / "experiment-1"
    / "experiment-1a"
    / "transitive_token_profiles"
    / "stats"
    / "paired_effects.csv"
)
DEFAULT_EXP1B_ROOT = (
    REPO_ROOT
    / "behavioral_results"
    / "experiment-1"
    / "experiment-1b"
    / "processing_experiment_1b_gpt2large_v1_lexical-overlap"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate EMNLP-ready methods/results figures for Experiments 1a and 1b."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--exp1a-summary", type=Path, default=DEFAULT_EXP1A_SUMMARY)
    parser.add_argument("--exp1a-paired", type=Path, default=DEFAULT_EXP1A_PAIRED)
    parser.add_argument("--exp1b-root", type=Path, default=DEFAULT_EXP1B_ROOT)
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_inputs(paths: List[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files:\n- " + "\n- ".join(missing))


def save_figure(fig: plt.Figure, output_base: Path, dpi: int) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


def ci_normal(mean: float, sd: float, n: int) -> Tuple[float, float]:
    if n <= 1:
        return mean, mean
    half = 1.96 * (sd / np.sqrt(n))
    return mean - half, mean + half


def ci_prop(p: float, n: int) -> Tuple[float, float]:
    if n <= 0:
        return p, p
    half = 1.96 * np.sqrt(max(0.0, p * (1.0 - p)) / n)
    return max(0.0, p - half), min(1.0, p + half)


def load_exp1a(summary_path: Path, paired_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(summary_path)
    paired = pd.read_csv(paired_path)
    return summary, paired


def normalize_prime_labels(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    alias_map = {
        "no_prime_eos": "no_prime",
        "no_prime_empty": "no_prime",
        "no_demo": "no_prime",
    }
    for column in ["prime_condition", "condition_a", "condition_b", "baseline"]:
        if column in normalized.columns:
            normalized[column] = normalized[column].replace(alias_map)
    return normalized


def load_exp1b(exp1b_root: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    mapping = {
        "Core-Core": exp1b_root / "processing_1b_core_core",
        "Jabberwocky-Jabberwocky": exp1b_root / "processing_1b_jabberwocky_jabberwocky",
    }
    result: Dict[str, Dict[str, pd.DataFrame]] = {}
    for label, folder in mapping.items():
        result[label] = {
            "summary": normalize_prime_labels(pd.read_csv(folder / "summary.csv")),
            "stats": normalize_prime_labels(pd.read_csv(folder / "stats.csv")),
            "items": normalize_prime_labels(pd.read_csv(folder / "item_scores.csv")),
        }
    return result


def build_priming_decomposition(
    exp1b: Dict[str, Dict[str, pd.DataFrame]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    imbalance_rows: List[Dict[str, float]] = []

    for cond_label, data in exp1b.items():
        stats = data["stats"]

        for baseline in ("no_prime", "filler"):
            active_row = stats[
                (stats["metric"] == "passive_choice_delta")
                & (stats["condition_a"] == "active")
                & (stats["condition_b"] == baseline)
            ].iloc[0]
            passive_row = stats[
                (stats["metric"] == "passive_choice_delta")
                & (stats["condition_a"] == "passive")
                & (stats["condition_b"] == baseline)
            ].iloc[0]

            # active priming toward ACTIVE
            rows.append(
                {
                    "condition": cond_label,
                    "baseline": baseline,
                    "priming_type": "active",
                    "estimate": float(active_row["mean_diff_b_minus_a"]),
                    "ci_low": float(active_row["bootstrap_ci95_low"]),
                    "ci_high": float(active_row["bootstrap_ci95_high"]),
                }
            )

            # passive priming toward PASSIVE
            rows.append(
                {
                    "condition": cond_label,
                    "baseline": baseline,
                    "priming_type": "passive",
                    "estimate": -float(passive_row["mean_diff_b_minus_a"]),
                    "ci_low": -float(passive_row["bootstrap_ci95_high"]),
                    "ci_high": -float(passive_row["bootstrap_ci95_low"]),
                }
            )

            imbalance = -float(passive_row["mean_diff_b_minus_a"]) - float(active_row["mean_diff_b_minus_a"])
            imbalance_rows.append(
                {
                    "condition": cond_label,
                    "baseline": baseline,
                    "imbalance": imbalance,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(imbalance_rows)


def make_fig1_paradigm(output_base: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax in axes:
        ax.set_axis_off()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

    # 1a panel
    axes[0].set_title("Experiment 1a (Sinclair-style processing)", loc="left", fontweight="bold")
    boxes_1a = [
        (0.7, 7.2, 8.6, 1.4, "Prime sentence (active OR passive)"),
        (0.7, 4.8, 8.6, 1.4, "Target sentence (fixed structure)"),
        (0.7, 2.4, 8.6, 1.4, "PE = logP(target|congruent) - logP(target|incongruent)"),
    ]
    for x, y, w, h, text in boxes_1a:
        axes[0].add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25", ec="#444", fc="#eef4ff"))
        axes[0].text(x + w / 2, y + h / 2, text, ha="center", va="center")
    axes[0].annotate("", xy=(5.0, 6.9), xytext=(5.0, 6.2), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[0].annotate("", xy=(5.0, 4.5), xytext=(5.0, 3.8), arrowprops=dict(arrowstyle="->", lw=1.5))

    # 1b panel
    axes[1].set_title("Experiment 1b (controlled processing)", loc="left", fontweight="bold")
    boxes_1b = [
        (0.7, 7.7, 8.6, 1.0, "Prime condition: active / passive / no_prime / filler"),
        (0.7, 5.9, 8.6, 1.0, "Score both target alternatives"),
        (0.7, 4.1, 4.1, 1.0, "Active target"),
        (5.2, 4.1, 4.1, 1.0, "Passive target"),
        (0.7, 2.2, 8.6, 1.0, "Delta = logP(passive) - logP(active)"),
    ]
    for x, y, w, h, text in boxes_1b:
        axes[1].add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25", ec="#444", fc="#eefcf2"))
        axes[1].text(x + w / 2, y + h / 2, text, ha="center", va="center")
    axes[1].annotate("", xy=(5.0, 6.8), xytext=(5.0, 6.2), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[1].annotate("", xy=(2.75, 5.0), xytext=(5.0, 5.6), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[1].annotate("", xy=(7.25, 5.0), xytext=(5.0, 5.6), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[1].annotate("", xy=(5.0, 3.1), xytext=(2.75, 4.0), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[1].annotate("", xy=(5.0, 3.1), xytext=(7.25, 4.0), arrowprops=dict(arrowstyle="->", lw=1.5))

    fig.suptitle("Paradigm Comparison", y=1.02, fontsize=14, fontweight="bold")
    save_figure(fig, output_base, dpi=dpi)


def make_fig2_exp1a_forest(summary: pd.DataFrame, paired: pd.DataFrame, output_base: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"CORE": "#2E86DE", "jabberwocky": "#E67E22"}

    y_labels: List[str] = []
    means: List[float] = []
    lows: List[float] = []
    highs: List[float] = []
    row_colors: List[str] = []

    for condition in ("CORE", "jabberwocky"):
        sub = summary[summary["condition"] == condition]
        for structure in ("active", "passive"):
            row = sub[sub["target_structure"] == structure].iloc[0]
            mean = float(row["mean_sentence_pe_mean"])
            low, high = ci_normal(mean, float(row["sd_sentence_pe_mean"]), int(row["n_items"]))
            y_labels.append(f"{condition} {structure} PE")
            means.append(mean)
            lows.append(low)
            highs.append(high)
            row_colors.append(colors[condition])

        p_row = paired[(paired["condition"] == condition) & (paired["metric"] == "sentence_pe_mean")].iloc[0]
        y_labels.append(f"{condition} passive-active")
        means.append(float(p_row["mean_diff"]))
        lows.append(float(p_row["bootstrap_ci95_low"]))
        highs.append(float(p_row["bootstrap_ci95_high"]))
        row_colors.append(colors[condition])

    y_pos = np.arange(len(y_labels))[::-1]
    for y, mean, low, high, color in zip(y_pos, means, lows, highs, row_colors):
        ax.plot([low, high], [y, y], color=color, lw=2.2)
        ax.scatter(mean, y, color=color, s=42, zorder=3)

    ax.axvline(0, color="#333", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Effect size (sentence_pe_mean)")
    ax.set_title("Experiment 1a: replication pattern and asymmetry (95% CI)")
    ax.grid(axis="x", alpha=0.2)

    save_figure(fig, output_base, dpi=dpi)


def make_fig3_exp1b_profiles(exp1b: Dict[str, Dict[str, pd.DataFrame]], output_base: Path, dpi: int) -> None:
    preferred_order = ["active", "passive", "no_prime", "filler"]
    present_conditions = set()
    for cond_data in exp1b.values():
        present_conditions.update(cond_data["summary"]["prime_condition"].astype(str).tolist())
    prime_order = [p for p in preferred_order if p in present_conditions]
    x = np.arange(len(prime_order))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    condition_order = ["Core-Core", "Jabberwocky-Jabberwocky"]
    colors = {"Core-Core": "#2E86DE", "Jabberwocky-Jabberwocky": "#E67E22"}

    for r, condition in enumerate(condition_order):
        summary = (
            exp1b[condition]["summary"]
            .groupby("prime_condition", as_index=False)
            .agg(
                n_items=("n_items", "mean"),
                passive_choice_rate=("passive_choice_rate", "mean"),
                mean_passive_minus_active_logprob=("mean_passive_minus_active_logprob", "mean"),
                sd_passive_minus_active_logprob=("sd_passive_minus_active_logprob", "mean"),
            )
        )
        row_data = {row["prime_condition"]: row for _, row in summary.iterrows()}

        passive_rates = []
        passive_low = []
        passive_high = []
        deltas = []
        delta_low = []
        delta_high = []

        for p in prime_order:
            row = row_data[p]
            n = int(row["n_items"])

            rate = float(row["passive_choice_rate"])
            low_r, high_r = ci_prop(rate, n)
            passive_rates.append(rate)
            passive_low.append(low_r)
            passive_high.append(high_r)

            delta = float(row["mean_passive_minus_active_logprob"])
            low_d, high_d = ci_normal(delta, float(row["sd_passive_minus_active_logprob"]), n)
            deltas.append(delta)
            delta_low.append(low_d)
            delta_high.append(high_d)

        ax_left = axes[r, 0]
        ax_right = axes[r, 1]

        yerr_left = np.array([np.array(passive_rates) - np.array(passive_low), np.array(passive_high) - np.array(passive_rates)])
        ax_left.errorbar(x, passive_rates, yerr=yerr_left, fmt="-o", color=colors[condition], capsize=3, lw=2)
        ax_left.set_ylim(0.0, 1.05)
        ax_left.set_ylabel("Passive-choice rate")
        ax_left.set_title(condition)
        ax_left.grid(axis="y", alpha=0.2)

        yerr_right = np.array([np.array(deltas) - np.array(delta_low), np.array(delta_high) - np.array(deltas)])
        ax_right.errorbar(x, deltas, yerr=yerr_right, fmt="-o", color=colors[condition], capsize=3, lw=2)
        ax_right.set_ylabel("Passive - Active logprob")
        ax_right.set_title(condition)
        ax_right.grid(axis="y", alpha=0.2)

    for c in range(2):
        axes[1, c].set_xticks(x)
        axes[1, c].set_xticklabels(prime_order)
        axes[0, c].set_xticks(x)
        axes[0, c].set_xticklabels(prime_order)

    fig.suptitle("Experiment 1b profiles by prime condition (95% CI)")
    fig.tight_layout()
    save_figure(fig, output_base, dpi=dpi)


def make_fig4_baseline_decomposition(
    decomposition: pd.DataFrame,
    imbalance: pd.DataFrame,
    output_base: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), gridspec_kw={"width_ratios": [1.4, 1.0]})

    # Left: active/passive priming relative to baseline
    left = axes[0]
    baseline_order = ["no_prime", "filler"]
    plot_rows: List[Tuple[str, str]] = []
    for cond in ("Core-Core", "Jabberwocky-Jabberwocky"):
        available = decomposition[decomposition["condition"] == cond]["baseline"].unique().tolist()
        for baseline in baseline_order:
            if baseline in available:
                plot_rows.append((cond, baseline))
    y_positions = np.arange(len(plot_rows))[::-1]

    for y, (cond, baseline) in zip(y_positions, plot_rows):
        for priming_type, color, offset in (
            ("active", "#2E86DE", -0.12),
            ("passive", "#E67E22", 0.12),
        ):
            row = decomposition[
                (decomposition["condition"] == cond)
                & (decomposition["baseline"] == baseline)
                & (decomposition["priming_type"] == priming_type)
            ].iloc[0]
            est = float(row["estimate"])
            low = float(row["ci_low"])
            high = float(row["ci_high"])
            y_adj = y + offset
            left.plot([low, high], [y_adj, y_adj], color=color, lw=2)
            left.scatter(est, y_adj, color=color, s=36, zorder=3)

    left.axvline(0, color="#333", linestyle="--", linewidth=1)
    left.set_yticks(y_positions)
    left.set_yticklabels([f"{c}, {b}" for c, b in plot_rows])
    left.set_xlabel("Priming magnitude (choice-rate delta)")
    left.set_title("Baseline-referenced priming")
    left.grid(axis="x", alpha=0.2)
    left.legend(
        handles=[
            plt.Line2D([0], [0], color="#2E86DE", marker="o", lw=2, label="Active priming"),
            plt.Line2D([0], [0], color="#E67E22", marker="o", lw=2, label="Passive priming"),
        ],
        loc="lower right",
    )

    # Right: imbalance passive-active priming
    right = axes[1]
    y_positions_r = np.arange(len(plot_rows))[::-1]
    for y, (cond, baseline) in zip(y_positions_r, plot_rows):
        row = imbalance[(imbalance["condition"] == cond) & (imbalance["baseline"] == baseline)].iloc[0]
        est = float(row["imbalance"])
        right.scatter(est, y, color="#8E44AD", s=46, zorder=3)
    right.axvline(0, color="#333", linestyle="--", linewidth=1)
    right.set_yticks(y_positions_r)
    right.set_yticklabels([f"{c}, {b}" for c, b in plot_rows])
    right.set_xlabel("Imbalance = Passive priming - Active priming")
    right.set_title("Priming imbalance")
    right.grid(axis="x", alpha=0.2)

    fig.suptitle("Experiment 1b decomposition and imbalance")
    fig.tight_layout()
    save_figure(fig, output_base, dpi=dpi)


def write_figure_notes(output_dir: Path) -> None:
    lines = [
        "# EMNLP Figure Set (Experiments 1a/1b)",
        "",
        "Generated by `scripts/17_make_emnlp_story_figures.py`.",
        "",
        "## Figure list",
        "- `fig1_paradigm_comparison`: schematic comparing the 1a and 1b paradigms.",
        "- `fig2_exp1a_replication_forest`: 1a active/passive PE plus passive-active contrasts with 95% CIs.",
        "- `fig3_exp1b_prime_condition_profiles`: small multiples across available prime conditions for choice and logprob metrics.",
        "- `fig4_exp1b_baseline_decomposition`: baseline-referenced active/passive priming and imbalance.",
        "",
        "## Notes",
        "- 1a active/passive CIs are normal-approximation CIs from summary SD and N.",
        "- 1a passive-active CIs are the bootstrap CIs from `paired_effects.csv`.",
        "- 1b choice-rate CIs are normal-approximation proportion CIs.",
        "- 1b logprob CIs are normal-approximation CIs from summary SD and N.",
        "- 1b decomposition CIs are taken from `stats.csv` and sign-adjusted for passive-priming framing.",
        "- `no_prime_empty`, `no_demo`, and `no_prime_eos` are normalized to `no_prime`.",
    ]
    (output_dir / "figure_notes.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    exp1b_root = args.exp1b_root.resolve()
    exp1a_summary_path = args.exp1a_summary.resolve()
    exp1a_paired_path = args.exp1a_paired.resolve()
    output_dir = args.output_dir.resolve()

    ensure_inputs(
        [
            exp1a_summary_path,
            exp1a_paired_path,
            exp1b_root / "processing_1b_core_core" / "summary.csv",
            exp1b_root / "processing_1b_core_core" / "stats.csv",
            exp1b_root / "processing_1b_core_core" / "item_scores.csv",
            exp1b_root / "processing_1b_jabberwocky_jabberwocky" / "summary.csv",
            exp1b_root / "processing_1b_jabberwocky_jabberwocky" / "stats.csv",
            exp1b_root / "processing_1b_jabberwocky_jabberwocky" / "item_scores.csv",
        ]
    )

    exp1a_summary, exp1a_paired = load_exp1a(exp1a_summary_path, exp1a_paired_path)
    exp1b = load_exp1b(exp1b_root)
    decomposition, imbalance = build_priming_decomposition(exp1b)

    make_fig1_paradigm(output_dir / "fig1_paradigm_comparison", dpi=args.dpi)
    make_fig2_exp1a_forest(exp1a_summary, exp1a_paired, output_dir / "fig2_exp1a_replication_forest", dpi=args.dpi)
    make_fig3_exp1b_profiles(exp1b, output_dir / "fig3_exp1b_prime_condition_profiles", dpi=args.dpi)
    make_fig4_baseline_decomposition(decomposition, imbalance, output_dir / "fig4_exp1b_baseline_decomposition", dpi=args.dpi)
    write_figure_notes(output_dir)

    print(f"Saved figure set to: {output_dir}")


if __name__ == "__main__":
    main()
