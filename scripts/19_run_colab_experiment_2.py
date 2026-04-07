import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "14_run_counterbalanced_production_experiments.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Colab-friendly wrapper for Experiment 2 production runs."
    )
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-items", type=int, default=2080)
    parser.add_argument(
        "--prompt-template",
        choices=("role_labeled", "word_list", "all"),
        default="role_labeled",
    )
    parser.add_argument(
        "--prime-conditions",
        nargs="+",
        default=["active", "passive", "no_prime_eos", "filler"],
        help="Subset of active passive no_prime_eos no_prime_empty filler.",
    )
    parser.add_argument(
        "--role-order",
        choices=("fixed", "shuffle"),
        default="shuffle",
    )
    parser.add_argument(
        "--which",
        choices=("completion", "generation", "both"),
        default="both",
        help="completion = Experiment 2a, generation = Experiment 2b, both = run both.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "behavioral_results" / "experiment_2_colab",
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

    command = [
        sys.executable,
        str(RUNNER_SCRIPT),
        "--model-name",
        args.model_name,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--max-items",
        str(args.max_items),
        "--prompt-template",
        args.prompt_template,
        "--role-order",
        args.role_order,
        "--which",
        args.which,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seed",
        str(args.seed),
        "--output-root",
        str(output_root),
        "--prime-conditions",
        *args.prime_conditions,
    ]

    run_command(command, cwd=REPO_ROOT)

    manifest: Dict[str, object] = {
        "runner": str(RUNNER_SCRIPT),
        "model_name": args.model_name,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_items": args.max_items,
        "prompt_template": args.prompt_template,
        "prime_conditions": args.prime_conditions,
        "role_order": args.role_order,
        "which": args.which,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "output_root": str(output_root),
    }
    manifest_path = output_root / "experiment_2_colab_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
