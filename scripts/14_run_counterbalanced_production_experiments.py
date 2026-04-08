import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
COUNTERBALANCED_CORE = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_constrained_counterbalanced.csv"
COUNTERBALANCED_CORE_LEXICALLY_CONTROLLED = (
    REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_constrained_counterbalanced_lexically_controlled.csv"
)
JABBERWOCKY_PRIMES = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_bpe_filtered.csv"
COMPLETION_SCRIPT = REPO_ROOT / "scripts" / "12_counterbalanced_completion_choice_experiment.py"
GENERATION_SCRIPT = REPO_ROOT / "scripts" / "13_counterbalanced_generation_experiment.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the counterbalanced controlled-pilot and generation-style production experiments."
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
    )
    parser.add_argument(
        "--prime-set",
        choices=("core", "jabberwocky", "both"),
        default="both",
        help="Which prime-source pairing(s) to run: core primes, jabberwocky primes, or both.",
    )
    parser.add_argument(
        "--core-prime-mode",
        choices=("lexically_controlled", "lexical_overlap"),
        default="lexically_controlled",
        help="Use the repaired core prime-target corpus or the older lexical-overlap version for comparison.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer/model from the local Hugging Face cache only.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "behavioral_results")
    return parser.parse_args()


def run_command(command: List[str], cwd: Path) -> None:
    print("")
    print("Running:")
    print(" ".join(command))
    sys.stdout.flush()
    subprocess.run(command, cwd=cwd, check=True)


def experiment_configs(output_root: Path, core_prime_mode: str) -> List[Dict[str, object]]:
    if core_prime_mode == "lexically_controlled":
        core_csv = COUNTERBALANCED_CORE_LEXICALLY_CONTROLLED
        core_completion_output = output_root / "counterbalanced_completion_core_primes_lexically_controlled"
        core_generation_output = output_root / "counterbalanced_generation_core_primes_lexically_controlled"
    else:
        core_csv = COUNTERBALANCED_CORE
        core_completion_output = output_root / "counterbalanced_completion_core_primes_lexical_overlap"
        core_generation_output = output_root / "counterbalanced_generation_core_primes_lexical_overlap"

    return [
        {
            "key": "core",
            "name": "core_primes_counterbalanced_core_production",
            "input_csv": core_csv,
            "prime_csv": core_csv,
            "completion_output": core_completion_output,
            "generation_output": core_generation_output,
            "filler_domain": "core",
        },
        {
            "key": "jabberwocky",
            "name": "jabberwocky_primes_counterbalanced_core_production",
            "input_csv": core_csv,
            "prime_csv": JABBERWOCKY_PRIMES,
            "completion_output": output_root / "counterbalanced_completion_jabberwocky_primes",
            "generation_output": output_root / "counterbalanced_generation_jabberwocky_primes",
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
        "prompt_template": args.prompt_template,
        "prime_conditions": args.prime_conditions,
        "prime_set": args.prime_set,
        "role_order": args.role_order,
        "core_prime_mode": args.core_prime_mode,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "which": args.which,
        "local_files_only": bool(args.local_files_only),
        "experiments": [],
    }

    allowed_prime_sets = {"core", "jabberwocky"} if args.prime_set == "both" else {args.prime_set}
    for config in experiment_configs(output_root, core_prime_mode=args.core_prime_mode):
        if config["key"] not in allowed_prime_sets:
            continue
        print(f"=== {config['name']} ===")
        experiment_entry = {
            "name": config["name"],
            "input_csv": str(config["input_csv"]),
            "prime_csv": str(config["prime_csv"]),
        }

        if args.which in {"completion", "both"}:
            completion_output = Path(config["completion_output"])
            completion_output.mkdir(parents=True, exist_ok=True)
            completion_command = [
                sys.executable,
                str(COMPLETION_SCRIPT),
                "--model-name",
                args.model_name,
                "--input-csv",
                str(config["input_csv"]),
                "--prime-csv",
                str(config["prime_csv"]),
                "--output-dir",
                str(completion_output),
                "--batch-size",
                str(args.batch_size),
                "--prompt-template",
                args.prompt_template,
                "--role-order",
                args.role_order,
                "--seed",
                str(args.seed),
                "--filler-domain",
                str(config["filler_domain"]),
            ]
            if args.device:
                completion_command.extend(["--device", args.device])
            if args.torch_dtype:
                completion_command.extend(["--torch-dtype", args.torch_dtype])
            if args.max_items is not None:
                completion_command.extend(["--max-items", str(args.max_items)])
            if args.local_files_only:
                completion_command.append("--local-files-only")
            completion_command.extend(["--prime-conditions", *args.prime_conditions])
            run_command(completion_command, cwd=REPO_ROOT)
            experiment_entry["completion_output_dir"] = str(completion_output)

        if args.which in {"generation", "both"}:
            generation_output = Path(config["generation_output"])
            generation_output.mkdir(parents=True, exist_ok=True)
            generation_command = [
                sys.executable,
                str(GENERATION_SCRIPT),
                "--model-name",
                args.model_name,
                "--input-csv",
                str(config["input_csv"]),
                "--prime-csv",
                str(config["prime_csv"]),
                "--output-dir",
                str(generation_output),
                "--batch-size",
                str(max(1, args.batch_size // 2)),
                "--prompt-template",
                args.prompt_template,
                "--role-order",
                args.role_order,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--seed",
                str(args.seed),
                "--filler-domain",
                str(config["filler_domain"]),
            ]
            if args.device:
                generation_command.extend(["--device", args.device])
            if args.torch_dtype:
                generation_command.extend(["--torch-dtype", args.torch_dtype])
            if args.max_items is not None:
                generation_command.extend(["--max-items", str(args.max_items)])
            if args.local_files_only:
                generation_command.append("--local-files-only")
            generation_command.extend(["--prime-conditions", *args.prime_conditions])
            run_command(generation_command, cwd=REPO_ROOT)
            experiment_entry["generation_output_dir"] = str(generation_output)

        manifest["experiments"].append(experiment_entry)

    manifest_path = output_root / "counterbalanced_production_experiments_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
