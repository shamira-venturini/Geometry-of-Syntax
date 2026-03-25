import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "behavioral_results" / "transitive_token_profiles"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "transitive_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compact markdown report from jabberwocky_transitive priming outputs."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def stderr(series: pd.Series) -> float:
    return float(series.std(ddof=1) / (len(series) ** 0.5)) if len(series) > 1 else 0.0


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [len(header) for header in headers]

    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    header_row = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_row, separator] + body)


def main() -> None:
    args = parse_args()
    item_path = args.input_dir / "transitive_item_level_scores.csv"
    token_path = args.input_dir / "transitive_token_level_scores.csv"
    word_path = args.input_dir / "transitive_word_level_scores.csv"

    items = pd.read_csv(item_path)
    tokens = pd.read_csv(token_path)
    words = pd.read_csv(word_path)

    region_summary = (
        items.groupby(["condition", "target_structure"])
        .agg(
            n_items=("item_index", "count"),
            sentence_mean=("sentence_pe", "mean"),
            sentence_se=("sentence_pe", stderr),
            sentence_token_mean=("sentence_pe_mean", "mean"),
            sentence_token_mean_se=("sentence_pe_mean", stderr),
            critical_word_index=("critical_word_index", "first"),
            critical_word_mean=("critical_word_pe", "mean"),
            critical_word_se=("critical_word_pe", stderr),
            critical_word_token_mean=("critical_word_pe_mean", "mean"),
            critical_word_token_mean_se=("critical_word_pe_mean", stderr),
            post_divergence_mean=("post_divergence_pe", "mean"),
            post_divergence_se=("post_divergence_pe", stderr),
            post_divergence_token_mean=("post_divergence_pe_mean", "mean"),
            post_divergence_token_mean_se=("post_divergence_pe_mean", stderr),
            structure_region_mean=("structure_region_pe", "mean"),
            structure_region_se=("structure_region_pe", stderr),
            structure_region_token_mean=("structure_region_pe_mean", "mean"),
            structure_region_token_mean_se=("structure_region_pe_mean", stderr),
        )
        .reset_index()
    )

    critical_word_summary = (
        items.groupby(["condition", "target_structure"])
        .agg(
            n_items=("item_index", "count"),
            critical_word_index=("critical_word_index", "first"),
            mean_word_pe=("critical_word_pe", "mean"),
            word_pe_se=("critical_word_pe", stderr),
            mean_word_pe_mean=("critical_word_pe_mean", "mean"),
            word_pe_mean_se=("critical_word_pe_mean", stderr),
            mean_token_count=("critical_word_token_count", "mean"),
        )
        .reset_index()
    )

    token_summary = (
        tokens.groupby(["condition", "target_structure", "token_index"])
        .agg(
            mean_token_pe=("token_pe", "mean"),
            token_pe_se=("token_pe", stderr),
            share_post_divergence=("is_post_divergence", "mean"),
        )
        .reset_index()
    )

    word_summary = (
        words.groupby(["condition", "target_structure", "word_index"])
        .agg(
            n_items=("item_index", "count"),
            mean_word_pe=("word_pe", "mean"),
            word_pe_se=("word_pe", stderr),
            mean_word_pe_mean=("word_pe_mean", "mean"),
            word_pe_mean_se=("word_pe_mean", stderr),
            mean_token_count=("token_count", "mean"),
            critical_word_rate=("is_critical_word", "mean"),
        )
        .reset_index()
    )

    lines = [
        "# Transitive Priming Report",
        "",
        "## Region-level summary",
        "",
        markdown_table(region_summary),
        "",
        "## Critical-word summary",
        "",
        markdown_table(critical_word_summary),
        "",
        "## Token-level summary",
        "",
        markdown_table(token_summary),
        "",
        "## Word-level summary",
        "",
        markdown_table(word_summary),
        "",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
