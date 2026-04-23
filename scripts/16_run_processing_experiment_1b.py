import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "15_counterbalanced_processing_experiment_1b.py"
CORE_COUNTERBALANCED = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_constrained_counterbalanced.csv"
CORE_COUNTERBALANCED_LEXICALLY_CONTROLLED = (
    REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_constrained_counterbalanced_lexically_controlled.csv"
)
JABBERWOCKY_2048 = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_bpe_filtered_2048.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 1b conditions with domain-matched processing corpora."
    )
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Torch dtype for model loading: auto, float32, float16, or bfloat16.",
    )
    parser.add_argument(
        "--prime-conditions",
        nargs="+",
        default=["active", "passive", "no_prime", "filler"],
        help="Subset of active passive no_prime filler.",
    )
    parser.add_argument(
        "--which",
        choices=("core", "jabberwocky", "both"),
        default="both",
    )
    parser.add_argument(
        "--core-prime-mode",
        choices=("lexically_controlled",),
        default="lexically_controlled",
        help="Strict Sinclair-controlled core mode.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer/model from local Hugging Face cache only.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "behavioral_results")
    return parser.parse_args()


def run_command(command: List[str], cwd: Path) -> None:
    print("")
    print("Running:")
    print(" ".join(command))
    sys.stdout.flush()
    subprocess.run(command, cwd=cwd, check=True)


def condition_configs(output_root: Path, core_prime_mode: str) -> List[Dict[str, object]]:
    if core_prime_mode != "lexically_controlled":
        raise ValueError(
            f"Unsupported core-prime-mode '{core_prime_mode}'. Only 'lexically_controlled' is allowed."
        )
    core_csv = CORE_COUNTERBALANCED_LEXICALLY_CONTROLLED
    core_output = output_root / "processing_1b_core_core_lexically_controlled"
    condition_label = "processing_1b_core_core_lexically_controlled"

    return [
        {
            "name": "core_primes_core_targets",
            "input_csv": core_csv,
            "prime_csv": core_csv,
            "output_dir": core_output,
            "condition_label": condition_label,
            "which_key": "core",
            "filler_domain": "core",
        },
        {
            "name": "jabberwocky_primes_jabberwocky_targets",
            "input_csv": JABBERWOCKY_2048,
            "prime_csv": JABBERWOCKY_2048,
            "output_dir": output_root / "processing_1b_jabberwocky_jabberwocky",
            "condition_label": "processing_1b_jabberwocky_jabberwocky",
            "which_key": "jabberwocky",
            "filler_domain": "jabberwocky",
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
        "torch_dtype": args.torch_dtype,
        "max_items": args.max_items,
        "prime_conditions": args.prime_conditions,
        "which": args.which,
        "core_prime_mode": args.core_prime_mode,
        "local_files_only": bool(args.local_files_only),
        "seed": args.seed,
        "runs": [],
    }

    for cfg in condition_configs(output_root, core_prime_mode=args.core_prime_mode):
        if args.which != "both" and cfg["which_key"] != args.which:
            continue

        print(f"=== {cfg['name']} ===")
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(RUNNER_SCRIPT),
            "--model-name",
            args.model_name,
            "--input-csv",
            str(cfg["input_csv"]),
            "--prime-csv",
            str(cfg["prime_csv"]),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--condition-label",
            str(cfg["condition_label"]),
            "--prime-conditions",
            *args.prime_conditions,
            "--filler-domain",
            str(cfg["filler_domain"]),
        ]
        if args.device:
            command.extend(["--device", args.device])
        if args.torch_dtype:
            command.extend(["--torch-dtype", args.torch_dtype])
        if args.max_items is not None:
            command.extend(["--max-items", str(args.max_items)])
        if args.local_files_only:
            command.append("--local-files-only")

        run_command(command, cwd=REPO_ROOT)
        manifest["runs"].append(
            {
                "name": cfg["name"],
                "input_csv": str(cfg["input_csv"]),
                "prime_csv": str(cfg["prime_csv"]),
                "output_dir": str(output_dir),
                "condition_label": cfg["condition_label"],
            }
        )

    manifest_path = output_root / "processing_experiment_1b_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
