import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
STRICT_CORE = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_constrained_counterbalanced_lexically_controlled.csv"
LEXICAL_OVERLAP_CORE = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_constrained_counterbalanced.csv"
JABBERWOCKY_2080 = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_bpe_filtered_2080.csv"
SCRIPT = REPO_ROOT / "scripts" / "24_demo_prompt_completion_experiment.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the demonstration-based Experiment 2 completion-choice variant."
    )
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-items", type=int, default=2080)
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Torch dtype for model loading: auto, float32, float16, or bfloat16.",
    )
    parser.add_argument(
        "--prime-conditions",
        nargs="+",
        default=["active", "passive", "no_demo", "filler"],
    )
    parser.add_argument(
        "--quote-style",
        choices=("mary_answered", "said_mary"),
        default="mary_answered",
    )
    parser.add_argument(
        "--event-style",
        choices=("there_was_event", "involving_event"),
        default="there_was_event",
    )
    parser.add_argument(
        "--role-style",
        choices=("responsible_affected", "did_to"),
        default="responsible_affected",
    )
    parser.add_argument(
        "--which",
        choices=("core", "jabberwocky", "both"),
        default="both",
    )
    parser.add_argument(
        "--core-prime-mode",
        choices=("lexically_controlled", "lexical_overlap"),
        default="lexically_controlled",
        help="Use the repaired core corpus or the older lexical-overlap version for comparison.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer/model from the local Hugging Face cache only.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "behavioral_results" / "experiment-2" / "demo_prompt_completion",
    )
    return parser.parse_args()


def run_command(command: List[str], cwd: Path) -> None:
    print("")
    print("Running:")
    print(" ".join(command))
    sys.stdout.flush()
    subprocess.run(command, cwd=cwd, check=True)


def experiment_configs(output_root: Path, core_prime_mode: str) -> List[Dict[str, object]]:
    if core_prime_mode == "lexically_controlled":
        core_csv = STRICT_CORE
        core_output = output_root / "core_demo_primes_lexically_controlled"
    else:
        core_csv = LEXICAL_OVERLAP_CORE
        core_output = output_root / "core_demo_primes_lexical_overlap"

    return [
        {
            "key": "core",
            "name": "core_demo_primes_counterbalanced_core_production",
            "input_csv": core_csv,
            "prime_csv": core_csv,
            "filler_domain": "core",
            "output_dir": core_output,
        },
        {
            "key": "jabberwocky",
            "name": "jabberwocky_demo_primes_counterbalanced_core_production",
            "input_csv": core_csv,
            "prime_csv": JABBERWOCKY_2080,
            "filler_domain": "jabberwocky",
            "output_dir": output_root / "jabberwocky_demo_primes",
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
        "prime_conditions": list(args.prime_conditions),
        "quote_style": args.quote_style,
        "event_style": args.event_style,
        "role_style": args.role_style,
        "core_prime_mode": args.core_prime_mode,
        "which": args.which,
        "seed": int(args.seed),
        "local_files_only": bool(args.local_files_only),
        "experiments": [],
    }

    allowed = {"core", "jabberwocky"} if args.which == "both" else {args.which}
    for config in experiment_configs(output_root, core_prime_mode=args.core_prime_mode):
        if config["key"] not in allowed:
            continue

        print(f"=== {config['name']} ===")
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(SCRIPT),
            "--model-name",
            args.model_name,
            "--input-csv",
            str(config["input_csv"]),
            "--prime-csv",
            str(config["prime_csv"]),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            str(args.batch_size),
            "--max-items",
            str(args.max_items),
            "--filler-domain",
            str(config["filler_domain"]),
            "--quote-style",
            args.quote_style,
            "--event-style",
            args.event_style,
            "--role-style",
            args.role_style,
            "--seed",
            str(args.seed),
            "--prime-conditions",
            *args.prime_conditions,
        ]
        if args.device:
            command.extend(["--device", args.device])
        if args.torch_dtype:
            command.extend(["--torch-dtype", args.torch_dtype])
        if args.local_files_only:
            command.append("--local-files-only")
        run_command(command, cwd=REPO_ROOT)

        manifest["experiments"].append(
            {
                "name": config["name"],
                "input_csv": str(config["input_csv"]),
                "prime_csv": str(config["prime_csv"]),
                "filler_domain": str(config["filler_domain"]),
                "output_dir": str(output_dir),
            }
        )

    manifest_path = output_root / "demo_prompt_completion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
