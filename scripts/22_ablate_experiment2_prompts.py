from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve()
    raise SystemExit(
        f"{script_path.name} is retired because it ablated the retired Experiment 2a "
        "completion-choice prompts."
    )


if __name__ == "__main__":
    main()
