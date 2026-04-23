from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve()
    raise SystemExit(
        f"{script_path.name} is retired because Experiment 2a/2b were retired. "
        "Use scripts/29_demo_prompt_generation_audit.py for the definitive Experiment 2 "
        "generation-audit path."
    )


if __name__ == "__main__":
    main()
