from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve()
    raise SystemExit(
        f"{script_path.name} is retired. Use scripts/10_run_colab_experiment_1a.py "
        "or scripts/16_run_processing_experiment_1b.py for current controlled runs."
    )


if __name__ == "__main__":
    main()
