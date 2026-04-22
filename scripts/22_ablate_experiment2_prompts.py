import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPLETION_SCRIPT = REPO_ROOT / "scripts" / "12_counterbalanced_completion_choice_experiment.py"
CORE_CSV = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_constrained_counterbalanced_lexically_controlled.csv"
JABBERWOCKY_CSV_2080 = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_bpe_filtered_2080.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "behavioral_results" / "experiment-2" / "prompt_ablation"

PROMPT_CONFIGS: List[Dict[str, str]] = [
    {"prompt_template": "role_labeled", "sentence_stub": "Sentence: the", "label": "baseline_role_labeled"},
    {"prompt_template": "cue_list", "sentence_stub": "The", "label": "cue_list_the"},
    {"prompt_template": "another_event", "sentence_stub": "The", "label": "another_event_the"},
    {"prompt_template": "same_kind_event", "sentence_stub": "The", "label": "same_kind_event_the"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small prompt-family ablation for Experiment 2a completion-choice prompts."
    )
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--which",
        choices=("core", "jabberwocky", "both"),
        default="core",
        help="Which prime condition family to test against the counterbalanced core targets.",
    )
    parser.add_argument(
        "--role-order",
        choices=("fixed", "shuffle"),
        default="fixed",
        help="Hold role cue order fixed by default so the ablation isolates prompt family.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer from local Hugging Face cache only.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def run_command(command: List[str], cwd: Path) -> None:
    print("")
    print("Running:")
    print(" ".join(command))
    sys.stdout.flush()
    subprocess.run(command, cwd=cwd, check=True)


def prime_configs(which: str) -> List[Dict[str, object]]:
    configs = [
        {
            "name": "core_primes",
            "input_csv": CORE_CSV,
            "prime_csv": CORE_CSV,
            "filler_domain": "core",
            "which_key": "core",
        },
        {
            "name": "jabberwocky_primes",
            "input_csv": CORE_CSV,
            "prime_csv": JABBERWOCKY_CSV_2080,
            "filler_domain": "jabberwocky",
            "which_key": "jabberwocky",
        },
    ]
    if which == "both":
        return configs
    return [cfg for cfg in configs if cfg["which_key"] == which]


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "model_name": args.model_name,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_items": args.max_items,
        "seed": args.seed,
        "role_order": args.role_order,
        "which": args.which,
        "prompt_configs": PROMPT_CONFIGS,
        "runs": [],
    }

    for condition_cfg in prime_configs(args.which):
        for prompt_cfg in PROMPT_CONFIGS:
            run_label = f"{condition_cfg['name']}__{prompt_cfg['label']}"
            output_dir = output_root / run_label
            output_dir.mkdir(parents=True, exist_ok=True)

            command = [
                sys.executable,
                str(COMPLETION_SCRIPT),
                "--model-name",
                args.model_name,
                "--input-csv",
                str(condition_cfg["input_csv"]),
                "--prime-csv",
                str(condition_cfg["prime_csv"]),
                "--output-dir",
                str(output_dir),
                "--max-items",
                str(args.max_items),
                "--batch-size",
                str(args.batch_size),
                "--prompt-template",
                str(prompt_cfg["prompt_template"]),
                "--sentence-stub",
                str(prompt_cfg["sentence_stub"]),
                "--role-order",
                args.role_order,
                "--filler-domain",
                str(condition_cfg["filler_domain"]),
                "--seed",
                str(args.seed),
                "--prime-conditions",
                "active",
                "passive",
                "no_prime",
                "filler",
            ]
            if args.device:
                command.extend(["--device", args.device])
            if args.local_files_only:
                command.append("--local-files-only")

            run_command(command, cwd=REPO_ROOT)
            manifest["runs"].append(
                {
                    "label": run_label,
                    "condition": condition_cfg["name"],
                    "prompt_template": prompt_cfg["prompt_template"],
                    "sentence_stub": prompt_cfg["sentence_stub"],
                    "output_dir": str(output_dir),
                }
            )

    summary_rows: List[Dict[str, object]] = []
    for run in manifest["runs"]:
        summary = pd.read_csv(Path(run["output_dir"]) / "summary.csv")
        stats = pd.read_csv(Path(run["output_dir"]) / "stats.csv")

        active_rate = float(summary.loc[summary["prime_condition"] == "active", "passive_choice_rate"].iloc[0])
        passive_rate = float(summary.loc[summary["prime_condition"] == "passive", "passive_choice_rate"].iloc[0])
        filler_rate = float(summary.loc[summary["prime_condition"] == "filler", "passive_choice_rate"].iloc[0])
        no_prime_rate = float(summary.loc[summary["prime_condition"] == "no_prime", "passive_choice_rate"].iloc[0])

        active_vs_passive = stats[
            (stats["metric"] == "passive_choice_delta")
            & (stats["condition_a"] == "active")
            & (stats["condition_b"] == "passive")
        ].iloc[0]
        active_vs_filler = stats[
            (stats["metric"] == "passive_choice_delta")
            & (stats["condition_a"] == "active")
            & (stats["condition_b"] == "filler")
        ].iloc[0]
        passive_vs_filler = stats[
            (stats["metric"] == "passive_choice_delta")
            & (stats["condition_a"] == "passive")
            & (stats["condition_b"] == "filler")
        ].iloc[0]

        summary_rows.append(
            {
                "condition": run["condition"],
                "prompt_template": run["prompt_template"],
                "sentence_stub": run["sentence_stub"],
                "passive_choice_active": active_rate,
                "passive_choice_passive": passive_rate,
                "passive_choice_filler": filler_rate,
                "passive_choice_no_prime": no_prime_rate,
                "prime_contrast": float(active_vs_passive["mean_diff_b_minus_a"]),
                "prime_contrast_p": float(active_vs_passive["t_p_two_sided"]),
                "active_priming_vs_filler": float(active_vs_filler["mean_diff_b_minus_a"]),
                "active_priming_vs_filler_p": float(active_vs_filler["t_p_two_sided"]),
                "passive_priming_vs_filler": float(-passive_vs_filler["mean_diff_b_minus_a"]),
                "passive_priming_vs_filler_p": float(passive_vs_filler["t_p_two_sided"]),
            }
        )

    ablation_summary = pd.DataFrame(summary_rows).sort_values(["condition", "prompt_template"])
    ablation_summary.to_csv(output_root / "prompt_ablation_summary.csv", index=False)
    (output_root / "prompt_ablation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("")
    print(f"Saved prompt ablation summary to {output_root / 'prompt_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
