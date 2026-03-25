import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "7_core_completion_choice_pilot.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the two full retained completion-choice experiments sequentially for Colab."
    )
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-items", type=int, default=15000)
    parser.add_argument(
        "--prompt-template",
        choices=("role_labeled", "word_list", "all"),
        default="role_labeled",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "behavioral_results",
        help="Directory where experiment folders will be created.",
    )
    return parser.parse_args()


def run_command(command: List[str], cwd: Path) -> None:
    print("")
    print("Running:")
    print(" ".join(command))
    sys.stdout.flush()
    subprocess.run(command, cwd=cwd, check=True)


def experiment_configs(output_root: Path) -> List[Dict[str, object]]:
    return [
        {
            "name": "core_targets_core_primes",
            "input_csv": REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_15000sampled_10-1.csv",
            "prime_csv": None,
            "output_dir": output_root / "core_completion_choice_full_role_labeled_gpt2large",
        },
        {
            "name": "core_targets_jabberwocky_primes",
            "input_csv": REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_15000sampled_10-1.csv",
            "prime_csv": REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_bpe_filtered.csv",
            "output_dir": output_root / "core_targets_jabber_primes_completion_choice_full_role_labeled_gpt2large",
        },
    ]


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "model_name": args.model_name,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_items": args.max_items,
        "prompt_template": args.prompt_template,
        "seed": args.seed,
        "experiments": [],
    }

    for config in experiment_configs(output_root):
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(RUNNER_SCRIPT),
            "--model-name",
            args.model_name,
            "--input-csv",
            str(config["input_csv"]),
            "--output-dir",
            str(output_dir),
            "--max-items",
            str(args.max_items),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--prompt-template",
            args.prompt_template,
            "--seed",
            str(args.seed),
        ]
        if config["prime_csv"] is not None:
            command.extend(["--prime-csv", str(config["prime_csv"])])

        print(f"=== {config['name']} ===")
        run_command(command, cwd=REPO_ROOT)

        manifest["experiments"].append(
            {
                "name": config["name"],
                "input_csv": str(config["input_csv"]),
                "prime_csv": str(config["prime_csv"]) if config["prime_csv"] is not None else None,
                "output_dir": str(output_dir),
            }
        )

    manifest_path = output_root / "completion_choice_full_runs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("")
    print(f"Saved manifest to {manifest_path}")
    print("Finished both completion-choice experiments.")


if __name__ == "__main__":
    main()
