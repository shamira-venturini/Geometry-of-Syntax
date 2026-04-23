import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMING_SCRIPT = REPO_ROOT / "scripts" / "2_transitive_token_priming.py"
REPORT_SCRIPT = REPO_ROOT / "scripts" / "3_summarize_transitive_priming.py"
STATS_SCRIPT = REPO_ROOT / "scripts" / "5_analyze_transitive_statistics.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Colab-friendly runner for Experiment 1a token-level transitive priming."
    )
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Torch dtype for model loading in the priming step: auto, float32, float16, or bfloat16.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-items", type=int, default=15000)
    parser.add_argument(
        "--preset",
        choices=("paper_main", "primelm_core", "primelm_recency", "primelm_cumulative", "primelm_semsim"),
        default="paper_main",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "behavioral_results" / "experiment-1" / "experiment-1a" / "transitive_token_profiles",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip scripts/3_summarize_transitive_priming.py",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip scripts/5_analyze_transitive_statistics.py",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer/model from local Hugging Face cache only.",
    )
    return parser.parse_args()


def run_command(command: List[str], cwd: Path) -> None:
    print("")
    print("Running:")
    print(" ".join(command))
    sys.stdout.flush()
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    commands: Dict[str, List[str]] = {}

    priming_command = [
        sys.executable,
        str(PRIMING_SCRIPT),
        "--model-name",
        args.model_name,
        "--output-dir",
        str(output_root),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--torch-dtype",
        args.torch_dtype,
        "--preset",
        args.preset,
        "--max-items",
        str(args.max_items),
    ]
    if args.local_files_only:
        priming_command.append("--local-files-only")
    commands["priming"] = priming_command
    run_command(priming_command, cwd=REPO_ROOT)

    if not args.skip_report:
        report_command = [
            sys.executable,
            str(REPORT_SCRIPT),
            "--input-dir",
            str(output_root),
            "--output",
            str(output_root / "transitive_report.md"),
        ]
        commands["report"] = report_command
        run_command(report_command, cwd=REPO_ROOT)

    if not args.skip_stats:
        stats_command = [
            sys.executable,
            str(STATS_SCRIPT),
            "--input",
            str(output_root / "transitive_item_level_scores.csv"),
            "--output-dir",
            str(output_root / "stats"),
            "--seed",
            str(args.seed),
        ]
        commands["stats"] = stats_command
        run_command(stats_command, cwd=REPO_ROOT)

    manifest = {
        "runner": str(Path(__file__).resolve()),
        "model_name": args.model_name,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "batch_size": args.batch_size,
        "max_items": args.max_items,
        "preset": args.preset,
        "seed": args.seed,
        "local_files_only": bool(args.local_files_only),
        "skip_report": bool(args.skip_report),
        "skip_stats": bool(args.skip_stats),
        "output_root": str(output_root),
        "commands": commands,
    }
    manifest_path = output_root / "experiment_1a_colab_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
