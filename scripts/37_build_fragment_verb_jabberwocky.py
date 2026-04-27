import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_matched_strict_4cell.csv"
DEFAULT_CORE = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_strict_4cell_counterbalanced.csv"
DEFAULT_OUTPUT = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_fragment_verbs_strict_4cell.csv"
DEFAULT_SUMMARY = (
    REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_fragment_verbs_strict_4cell_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnostic Jabberwocky ablation that preserves nonce nouns "
            "but replaces nonce verb forms with standalone inflectional fragments."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--core-csv", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--tokenizer-model", default="gpt2-large")
    return parser.parse_args()


def read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip().str.lower()
    expected = ["pa", "pp", "ta", "tp"]
    missing = set(expected).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return frame[expected].copy()


def active_parts(sentence: str) -> Tuple[str, str, str, str, str]:
    tokens = sentence.strip().split()
    if len(tokens) != 6 or tokens[-1] != ".":
        raise ValueError(f"Unexpected active sentence format: {sentence}")
    return tokens[0], tokens[1], tokens[2], tokens[3], tokens[4]


def passive_parts(sentence: str) -> Tuple[str, str, str, str, str, str]:
    tokens = sentence.strip().split()
    if len(tokens) != 8 or tokens[4] != "by" or tokens[-1] != ".":
        raise ValueError(f"Unexpected passive sentence format: {sentence}")
    return tokens[0], tokens[1], tokens[2], tokens[3], tokens[5], tokens[6]


def active_fragment(active_sentence: str, passive_sentence: str) -> str:
    det_agent, agent, _, det_patient, patient = active_parts(active_sentence)
    _, _, aux, _, _, _ = passive_parts(passive_sentence)
    if aux == "is":
        fragment = "s"
    elif aux == "was":
        fragment = "ed"
    else:
        raise ValueError(f"Unsupported passive auxiliary for tense inference: {aux}")
    return f"{det_agent} {agent} {fragment} {det_patient} {patient} ."


def passive_fragment(passive_sentence: str) -> str:
    det_patient, patient, aux, _, det_agent, agent = passive_parts(passive_sentence)
    return f"{det_patient} {patient} {aux} ed by {det_agent} {agent} ."


def build_fragment_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "pa": active_fragment(str(row.pa), str(row.pp)),
                "pp": passive_fragment(str(row.pp)),
                "ta": active_fragment(str(row.ta), str(row.tp)),
                "tp": passive_fragment(str(row.tp)),
            }
        )
    return pd.DataFrame(rows, columns=["pa", "pp", "ta", "tp"])


def det_family(det: str) -> str:
    if det == "the":
        return "def"
    if det in {"a", "an"}:
        return "indef"
    return "unknown"


def cell_from_active_passive(active: str, passive: str) -> str:
    det_a, _, _, det_p, _ = active_parts(active)
    _, _, aux, _, _, _ = passive_parts(passive)
    tense = {"is": "present", "was": "past"}.get(aux, "unknown")
    family = det_family(det_a)
    if det_family(det_p) != family:
        return f"mixed_{tense}"
    return f"{family}_{tense}"


def row_constraint_audit(frame: pd.DataFrame) -> Dict[str, int]:
    same_aux_rows = 0
    shared_noun_rows = 0
    same_active_fragment_rows = 0
    same_det_family_rows = 0

    for row in frame.itertuples(index=False):
        pa_det_a, pa_agent, pa_fragment, _, pa_patient = active_parts(str(row.pa))
        ta_det_a, ta_agent, ta_fragment, _, ta_patient = active_parts(str(row.ta))
        _, _, pp_aux, _, _, _ = passive_parts(str(row.pp))
        _, _, tp_aux, _, _, _ = passive_parts(str(row.tp))

        same_aux_rows += int(pp_aux == tp_aux)
        shared_noun_rows += int(bool({pa_agent, pa_patient} & {ta_agent, ta_patient}))
        same_active_fragment_rows += int(pa_fragment == ta_fragment)
        same_det_family_rows += int(det_family(pa_det_a) == det_family(ta_det_a))

    return {
        "same_aux_rows": same_aux_rows,
        "shared_noun_rows": shared_noun_rows,
        "same_active_fragment_rows": same_active_fragment_rows,
        "same_det_family_rows": same_det_family_rows,
    }


def safe_stats(values: Sequence[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name, local_files_only=True), None
    except Exception as exc:
        return None, str(exc)


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(" " + str(text), add_special_tokens=False))


def tokenizer_audit(core: pd.DataFrame, fragment: pd.DataFrame, tokenizer, model_name: str) -> Dict[str, object]:
    if tokenizer is None:
        return {"status": "skipped", "model": model_name}

    result = {}
    for column in ["pa", "pp", "ta", "tp"]:
        core_counts = [token_count(tokenizer, text) for text in core[column]]
        fragment_counts = [token_count(tokenizer, text) for text in fragment[column]]
        diffs = [fragment_value - core_value for core_value, fragment_value in zip(core_counts, fragment_counts)]
        result[column] = {
            "core": safe_stats(core_counts),
            "fragment": safe_stats(fragment_counts),
            "fragment_minus_core": safe_stats(diffs),
            "exact_match_rows": int(sum(diff == 0 for diff in diffs)),
            "nonzero_diff_rows": int(sum(diff != 0 for diff in diffs)),
            "negative_diff_rows": int(sum(diff < 0 for diff in diffs)),
            "positive_diff_rows": int(sum(diff > 0 for diff in diffs)),
        }
    return {"status": "ok", "model": model_name, "columns": result}


def fail_if_bad(summary: Mapping[str, object]) -> None:
    audit = summary["row_constraint_audit"]
    for key in ["same_aux_rows", "shared_noun_rows", "same_det_family_rows"]:
        if int(audit[key]) != 0:
            raise RuntimeError(f"Fragment-verb audit failed: {key}={audit[key]}")


def main() -> None:
    args = parse_args()
    source = read_frame(args.input_csv)
    core = read_frame(args.core_csv)
    fragment = build_fragment_frame(source)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fragment.to_csv(args.output_csv, index=False)

    tokenizer, tokenizer_error = load_tokenizer(args.tokenizer_model)
    target_cells = Counter(cell_from_active_passive(row.ta, row.tp) for row in fragment.itertuples(index=False))
    prime_cells = Counter(cell_from_active_passive(row.pa, row.pp) for row in fragment.itertuples(index=False))
    summary = {
        "source_jabberwocky_csv": str(args.input_csv.resolve()),
        "source_core_csv": str(args.core_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "summary_json": str(args.summary_json.resolve()),
        "row_count": int(len(fragment)),
        "manipulation": "nonce nouns retained; verb slots replaced by standalone inflectional fragments s/ed",
        "target_cell_counts": dict(sorted(target_cells.items())),
        "prime_cell_counts": dict(sorted(prime_cells.items())),
        "row_constraint_audit": row_constraint_audit(fragment),
        "tokenizer_audit": tokenizer_audit(core, fragment, tokenizer, args.tokenizer_model),
        "tokenizer_load_error": tokenizer_error,
        "notes": [
            "This is a diagnostic ablation, not an English-like Jabberwocky corpus.",
            "Active present verbs are replaced with standalone 's'.",
            "Active past verbs and passive participles are replaced with standalone 'ed'.",
            "The manipulation tests whether suffix-like tokenizer fragments can carry structural priming cues.",
        ],
    }
    fail_if_bad(summary)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Saved fragment-verb Jabberwocky corpus to {args.output_csv}")
    print(f"Summary: {args.summary_json}")


if __name__ == "__main__":
    main()
