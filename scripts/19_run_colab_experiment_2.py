from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve()
    raise SystemExit(
        f"{script_path.name} is retired because Experiment 2a/2b were retired. "
        "Run the current Experiment 2 generation-audit flow from "
        "notebooks/colab_llama32_all_experiments.ipynb or call "
        "scripts/29_demo_prompt_generation_audit.py directly."
    )


if __name__ == "__main__":
    main()
