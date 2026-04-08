import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from production_priming_common import REPO_ROOT, lexical_overlap_audit, normalize_transitive_frame


DEFAULT_SOURCE = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_bpe_filtered.csv"
DEFAULT_OUTPUT = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_bpe_filtered_2080.csv"
DEFAULT_SUMMARY = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_bpe_filtered_2080_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a fixed-size Jabberwocky transitive subset for controlled experiments "
            "instead of relying on runtime sampling."
        )
    )
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--n-items", type=int, default=2080)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_csv = args.source_csv.resolve()
    source_frame = normalize_transitive_frame(pd.read_csv(source_csv))
    if args.n_items > len(source_frame):
        raise ValueError(f"Requested {args.n_items} rows from a {len(source_frame)}-row source corpus.")

    rng = np.random.default_rng(args.seed)
    sampled_indices = rng.choice(len(source_frame), size=args.n_items, replace=False)
    sampled_frame = source_frame.iloc[sampled_indices].reset_index(drop=True)
    overlap_audit = lexical_overlap_audit(target_frame=sampled_frame, prime_frame=sampled_frame)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled_frame.to_csv(args.output_csv, index=False)

    summary = {
        "source_csv": str(source_csv),
        "output_csv": str(args.output_csv.resolve()),
        "summary_json": str(args.summary_json.resolve()),
        "n_items": int(args.n_items),
        "seed": int(args.seed),
        "sampling": "without_replacement",
        "sampled_source_row_indices_preview": [int(index) for index in sampled_indices[:25]],
        "within_row_lexical_audit": overlap_audit,
        "notes": [
            "This file freezes the Jabberwocky prime pool at the same size as the strict controlled CORE target set.",
            "Within-row lexical audit compares pa/pp against ta/tp for the sampled rows.",
        ],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {len(sampled_frame)} Jabberwocky rows to {args.output_csv}")
    print(f"Summary written to {args.summary_json}")


if __name__ == "__main__":
    main()
