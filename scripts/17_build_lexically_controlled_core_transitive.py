from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve()
    raise SystemExit(
        f"{script_path.name} is retired because it builds the older lexically "
        "controlled core corpus. Current experiments use "
        "corpora/transitive/CORE_transitive_strict_4cell_counterbalanced.csv."
    )


if __name__ == "__main__":
    main()
