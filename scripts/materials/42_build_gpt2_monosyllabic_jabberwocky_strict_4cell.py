#!/usr/bin/env python3
"""Build the GPT-2-large one-token-noun Jabberwocky strict 4-cell corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORE = (
    REPO_ROOT
    / "materials/corpora/CORE_transitive_strict_4cell_counterbalanced.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "materials/corpora/jabberwocky_transitive_monosyllabic_strict_4cell-counterbalanced.csv"
)
DEFAULT_LEXICON = (
    REPO_ROOT
    / "materials/vocabulary_lists/jabberwocky_lexicon_monosyllabic.json"
)
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "materials/metadata/jabberwocky_transitive_monosyllabic_strict_4cell-counterbalanced_summary.json"
)


def portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)

TEXT_COLUMNS = ("pa", "pp", "ta", "tp")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")
TOKENIZER_COMPATIBILITY_EXCLUDES = {"rul"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mirror the strict CORE 4-cell transitive template with GPT-2-large "
            "single-token Jabberwocky nouns and standalone s/ed verb fragments."
        )
    )
    parser.add_argument("--core-csv", type=Path, default=DEFAULT_CORE)
    parser.add_argument(
        "--lexicon-json",
        type=Path,
        default=DEFAULT_LEXICON,
        help="Frozen curated noun and verb-fragment inventory.",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--tokenizer-model", default="gpt2-large")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only an already-cached tokenizer instead of allowing a download.",
    )
    return parser.parse_args()


def read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip().str.lower()
    missing = set(TEXT_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return frame[list(TEXT_COLUMNS)].copy()


def active_parts(sentence: str) -> tuple[str, str, str, str, str]:
    tokens = sentence.strip().split()
    if len(tokens) != 6 or tokens[-1] != ".":
        raise ValueError(f"Unexpected active sentence format: {sentence}")
    return tokens[0].lower(), tokens[1].lower(), tokens[2].lower(), tokens[3].lower(), tokens[4].lower()


def passive_parts(sentence: str) -> tuple[str, str, str, str, str, str]:
    tokens = sentence.strip().split()
    if len(tokens) != 8 or tokens[4].lower() != "by" or tokens[-1] != ".":
        raise ValueError(f"Unexpected passive sentence format: {sentence}")
    return (
        tokens[0].lower(),
        tokens[1].lower(),
        tokens[2].lower(),
        tokens[3].lower(),
        tokens[5].lower(),
        tokens[6].lower(),
    )


def det_family(det: str) -> str:
    if det == "the":
        return "def"
    if det in {"a", "an"}:
        return "indef"
    return "unknown"


def nonce_det(source_det: str, noun: str) -> str:
    if source_det not in {"the", "a", "an"}:
        raise ValueError(f"Unexpected source determiner: {source_det}")
    if source_det == "the":
        return "the"
    if noun[:1].lower() not in CONSONANTS:
        return "an"
    return "a"


def active_fragment_from_template(active_sentence: str, passive_sentence: str, agent: str, patient: str) -> str:
    source_agent_det, _, _, source_patient_det, _ = active_parts(active_sentence)
    _, _, aux, _, _, _ = passive_parts(passive_sentence)
    if aux == "is":
        fragment = "s"
    elif aux == "was":
        fragment = "ed"
    else:
        raise ValueError(f"Unexpected auxiliary: {aux}")
    agent_det = nonce_det(source_agent_det, agent)
    patient_det = nonce_det(source_patient_det, patient)
    return f"{agent_det} {agent} {fragment} {patient_det} {patient} ."


def passive_fragment_from_template(passive_sentence: str, agent: str, patient: str) -> str:
    source_patient_det, _, aux, _, source_agent_det, _ = passive_parts(passive_sentence)
    patient_det = nonce_det(source_patient_det, patient)
    agent_det = nonce_det(source_agent_det, agent)
    return f"{patient_det} {patient} {aux} ed by {agent_det} {agent} ."


def load_nouns(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    nouns = [str(noun).strip().lower() for noun in payload.get("nouns", [])]
    if len(nouns) < 4 or len(nouns) != len(set(nouns)) or any(not noun for noun in nouns):
        raise ValueError("Lexicon must contain at least four unique, non-empty nouns.")
    excluded = sorted(set(nouns) & TOKENIZER_COMPATIBILITY_EXCLUDES)
    if excluded:
        raise ValueError(f"Lexicon contains tokenizer-incompatible nouns: {excluded}")
    return nouns


def build_frame(core: pd.DataFrame, nouns: Sequence[str]) -> pd.DataFrame:
    rows = []
    n = len(nouns)
    for index, row in enumerate(core.itertuples(index=False)):
        # Four separated offsets guarantee no within-row prime/target noun overlap
        # as long as the noun list has at least four unique entries.
        pa_agent = nouns[index % n]
        pa_patient = nouns[(index + 1) % n]
        ta_agent = nouns[(index + 2) % n]
        ta_patient = nouns[(index + 3) % n]
        rows.append(
            {
                "pa": active_fragment_from_template(str(row.pa), str(row.pp), pa_agent, pa_patient),
                "pp": passive_fragment_from_template(str(row.pp), pa_agent, pa_patient),
                "ta": active_fragment_from_template(str(row.ta), str(row.tp), ta_agent, ta_patient),
                "tp": passive_fragment_from_template(str(row.tp), ta_agent, ta_patient),
            }
        )
    return pd.DataFrame(rows, columns=list(TEXT_COLUMNS))


def cell_from_active_passive(active: str, passive: str) -> str:
    det_agent, _, _, det_patient, _ = active_parts(active)
    _, _, aux, _, _, _ = passive_parts(passive)
    tense = {"is": "present", "was": "past"}.get(aux, "unknown")
    family = det_family(det_agent)
    if det_family(det_patient) != family:
        return f"mixed_{tense}"
    return f"{family}_{tense}"


def row_constraint_audit(frame: pd.DataFrame) -> dict[str, int]:
    audit = {
        "unknown_determiner_rows": 0,
        "bad_indefinite_article_uses": 0,
        "same_active_fragment_rows": 0,
        "same_aux_rows": 0,
        "same_det_family_rows": 0,
        "shared_noun_rows": 0,
    }
    for row in frame.itertuples(index=False):
        pa_det_a, pa_agent, pa_fragment, pa_det_p, pa_patient = active_parts(str(row.pa))
        ta_det_a, ta_agent, ta_fragment, ta_det_p, ta_patient = active_parts(str(row.ta))
        _, _, pp_aux, _, _, _ = passive_parts(str(row.pp))
        _, _, tp_aux, _, _, _ = passive_parts(str(row.tp))

        audit["same_active_fragment_rows"] += int(pa_fragment == ta_fragment)
        audit["same_aux_rows"] += int(pp_aux == tp_aux)
        audit["same_det_family_rows"] += int(det_family(pa_det_a) == det_family(ta_det_a))
        audit["shared_noun_rows"] += int(bool({pa_agent, pa_patient} & {ta_agent, ta_patient}))
        for det, noun in (
            (pa_det_a, pa_agent),
            (pa_det_p, pa_patient),
            (ta_det_a, ta_agent),
            (ta_det_p, ta_patient),
        ):
            audit["unknown_determiner_rows"] += int(det_family(det) == "unknown")
            audit["bad_indefinite_article_uses"] += int(
                (det == "a" and noun[:1] not in CONSONANTS)
                or (det == "an" and noun[:1] in CONSONANTS)
            )
    return audit


def safe_stats(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(" " + str(text), add_special_tokens=False))


def tokenization_audit(core: pd.DataFrame, jabber: pd.DataFrame, tokenizer) -> dict[str, object]:
    columns = {}
    for column in TEXT_COLUMNS:
        core_counts = [token_count(tokenizer, text) for text in core[column]]
        jabber_counts = [token_count(tokenizer, text) for text in jabber[column]]
        diffs = [jabber_count - core_count for core_count, jabber_count in zip(core_counts, jabber_counts)]
        columns[column] = {
            "core": safe_stats(core_counts),
            "jabber": safe_stats(jabber_counts),
            "jabber_minus_core": safe_stats(diffs),
            "exact_match_rows": int(sum(diff == 0 for diff in diffs)),
            "nonzero_diff_rows": int(sum(diff != 0 for diff in diffs)),
            "negative_diff_rows": int(sum(diff < 0 for diff in diffs)),
            "positive_diff_rows": int(sum(diff > 0 for diff in diffs)),
        }
    return {"status": "ok", "columns": columns}


def noun_tokenization_audit(nouns: Sequence[str], tokenizer, model_name: str) -> dict[str, object]:
    rows = []
    for noun in nouns:
        ids = tokenizer.encode(" " + noun, add_special_tokens=False)
        rows.append(
            {
                "noun": noun,
                "token_count": len(ids),
                "token_ids": ids,
                "tokens": tokenizer.convert_ids_to_tokens(ids),
            }
        )
    bad = [row for row in rows if row["token_count"] != 1]
    return {
        "model": model_name,
        "status": "ok" if not bad else "failed",
        "noun_count": len(rows),
        "all_one_token": not bad,
        "bad_preview": bad[:10],
        "preview": rows[:20],
    }


def fail_if_bad(summary: dict[str, object]) -> None:
    audit = summary["row_constraint_audit"]
    for key, value in audit.items():
        if int(value) != 0:
            raise RuntimeError(f"Strict 4-cell audit failed: {key}={value}")
    if not summary["noun_tokenization"]["all_one_token"]:
        raise RuntimeError("Noun tokenization audit failed.")


def main() -> None:
    args = parse_args()
    core = read_frame(args.core_csv)
    nouns = load_nouns(args.lexicon_json)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        local_files_only=args.local_files_only,
    )
    jabber = build_frame(core, nouns)

    target_cells = Counter(cell_from_active_passive(row.ta, row.tp) for row in jabber.itertuples(index=False))
    prime_cells = Counter(cell_from_active_passive(row.pa, row.pp) for row in jabber.itertuples(index=False))

    summary = {
        "source_core_csv": portable_path(args.core_csv),
        "output_csv": portable_path(args.output_csv),
        "lexicon_json": portable_path(args.lexicon_json),
        "summary_json": portable_path(args.summary_json),
        "tokenizer_model": args.tokenizer_model,
        "row_count": int(len(jabber)),
        "manipulation": "Curated GPT-2 one-token monosyllabic nonce nouns plus standalone inflectional verb fragments s/ed",
        "nouns": list(nouns),
        "target_cell_counts": dict(sorted(target_cells.items())),
        "prime_cell_counts": dict(sorted(prime_cells.items())),
        "row_constraint_audit": row_constraint_audit(jabber),
        "noun_tokenization": noun_tokenization_audit(nouns, tokenizer, args.tokenizer_model),
        "sentence_tokenization": tokenization_audit(core, jabber, tokenizer),
        "notes": [
            "Corpus mirrors the strict CORE 4-cell row structure.",
            "Nonce determiners preserve CORE determiner family while using phonologically appropriate a/an allomorphy.",
            "Prime and target have opposite determiner family and opposite tense/auxiliary in every row.",
            "Prime and target share no noun within row.",
            "Active present verbs are standalone 's'; active past and passive participles are standalone 'ed'.",
        ],
    }
    fail_if_bad(summary)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    jabber.to_csv(args.output_csv, index=False)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Saved {args.output_csv}")
    print(f"Summary: {args.summary_json}")


if __name__ == "__main__":
    main()
