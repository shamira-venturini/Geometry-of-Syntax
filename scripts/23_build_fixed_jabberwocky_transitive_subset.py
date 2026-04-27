from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve()
    raise SystemExit(
        f"{script_path.name} is retired because it rebuilds the old BPE-filtered "
        "Jabberwocky subset. Use scripts/42_build_gpt2_monosyllabic_jabberwocky_strict_4cell.py "
        "for the corrected strict 4-cell Jabberwocky corpus."
    )


if __name__ == "__main__":
    main()
