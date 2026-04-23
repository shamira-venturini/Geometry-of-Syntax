from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve()
    raise SystemExit(
        f"{script_path.name} is retired. "
        "Experiment 1a now runs via scripts/10_run_colab_experiment_1a.py "
        "(token-level transitive priming)."
    )


if __name__ == "__main__":
    main()
