from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIME_ORDER = ["active", "passive", "no_prime", "filler"]


def _series_label(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["model_condition"].astype(str)
        + " | "
        + frame["task"].astype(str)
        + " | "
        + frame["prompt_format_used"].astype(str)
    )


def _group_mean(item_level: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    return (
        item_level.groupby(group_cols, as_index=False)
        .agg(preference_total_mean=("preference_total", "mean"))
        .sort_values(group_cols)
    )


def plot_preference_by_prime_condition(item_level: pd.DataFrame, output_path: Path) -> None:
    summary = _group_mean(
        item_level=item_level,
        group_cols=["model_condition", "task", "prompt_format_used", "prime_condition"],
    )
    summary = summary.copy()
    summary["series"] = _series_label(summary)

    x_labels = [condition for condition in PRIME_ORDER if condition in set(summary["prime_condition"])]
    series_list = sorted(summary["series"].unique())

    x = np.arange(len(x_labels), dtype=float)
    width = 0.8 / max(len(series_list), 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, series in enumerate(series_list):
        subset = summary.loc[summary["series"] == series]
        values = [
            float(subset.loc[subset["prime_condition"] == condition, "preference_total_mean"].mean())
            if not subset.loc[subset["prime_condition"] == condition, "preference_total_mean"].empty
            else np.nan
            for condition in x_labels
        ]
        ax.bar(x + idx * width, values, width=width, label=series)

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x + width * (len(series_list) - 1) / 2.0)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_ylabel("Mean preference_total")
    ax.set_title("Preference by Prime Condition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_preference_by_task(item_level: pd.DataFrame, output_path: Path) -> None:
    summary = _group_mean(
        item_level=item_level,
        group_cols=["model_condition", "prompt_format_used", "task"],
    )
    summary = summary.copy()
    summary["series"] = summary["model_condition"].astype(str) + " | " + summary["prompt_format_used"].astype(str)

    tasks = sorted(summary["task"].unique())
    series_list = sorted(summary["series"].unique())

    x = np.arange(len(tasks), dtype=float)
    width = 0.8 / max(len(series_list), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, series in enumerate(series_list):
        subset = summary.loc[summary["series"] == series]
        values = [
            float(subset.loc[subset["task"] == task, "preference_total_mean"].mean())
            if not subset.loc[subset["task"] == task, "preference_total_mean"].empty
            else np.nan
            for task in tasks
        ]
        ax.bar(x + idx * width, values, width=width, label=series)

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x + width * (len(series_list) - 1) / 2.0)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Mean preference_total")
    ax.set_title("Preference by Task")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_preference_by_lexicality(item_level: pd.DataFrame, output_path: Path) -> None:
    summary = _group_mean(
        item_level=item_level,
        group_cols=["model_condition", "task", "prompt_format_used", "lexicality_condition"],
    )
    summary = summary.copy()
    summary["series"] = _series_label(summary)

    lexicalities = sorted(summary["lexicality_condition"].unique())
    series_list = sorted(summary["series"].unique())

    x = np.arange(len(lexicalities), dtype=float)
    width = 0.8 / max(len(series_list), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, series in enumerate(series_list):
        subset = summary.loc[summary["series"] == series]
        values = [
            float(
                subset.loc[
                    subset["lexicality_condition"] == lexicality,
                    "preference_total_mean",
                ].mean()
            )
            if not subset.loc[subset["lexicality_condition"] == lexicality, "preference_total_mean"].empty
            else np.nan
            for lexicality in lexicalities
        ]
        ax.bar(x + idx * width, values, width=width, label=series)

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x + width * (len(series_list) - 1) / 2.0)
    ax.set_xticklabels(lexicalities)
    ax.set_ylabel("Mean preference_total")
    ax.set_title("Preference by Lexicality")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_interaction_prime_by_task(item_level: pd.DataFrame, output_path: Path) -> None:
    summary = _group_mean(
        item_level=item_level,
        group_cols=["model_condition", "task", "prime_condition"],
    )
    summary = summary.copy()

    models = sorted(summary["model_condition"].unique())
    x_labels = [condition for condition in PRIME_ORDER if condition in set(summary["prime_condition"])]
    x = np.arange(len(x_labels), dtype=float)

    fig, axes = plt.subplots(1, max(len(models), 1), figsize=(6 * max(len(models), 1), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for axis, model_condition in zip(axes, models):
        subset = summary.loc[summary["model_condition"] == model_condition]
        for task in sorted(subset["task"].unique()):
            task_subset = subset.loc[subset["task"] == task]
            values = [
                float(task_subset.loc[task_subset["prime_condition"] == condition, "preference_total_mean"].mean())
                if not task_subset.loc[task_subset["prime_condition"] == condition, "preference_total_mean"].empty
                else np.nan
                for condition in x_labels
            ]
            axis.plot(x, values, marker="o", label=task)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xticks(x)
        axis.set_xticklabels(x_labels, rotation=20, ha="right")
        axis.set_title(model_condition)
        axis.set_xlabel("Prime condition")
        axis.legend(fontsize=8)

    axes[0].set_ylabel("Mean preference_total")
    fig.suptitle("Prime x Task Interaction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_all_default_plots(item_level: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_preference_by_prime_condition(
        item_level=item_level,
        output_path=output_dir / "preference_by_prime_condition.png",
    )
    plot_preference_by_task(
        item_level=item_level,
        output_path=output_dir / "preference_by_task.png",
    )
    plot_preference_by_lexicality(
        item_level=item_level,
        output_path=output_dir / "preference_by_lexicality.png",
    )
    plot_interaction_prime_by_task(
        item_level=item_level,
        output_path=output_dir / "interaction_prime_by_task.png",
    )
