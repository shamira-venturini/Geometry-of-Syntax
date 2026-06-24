import argparse
from pathlib import Path
import re

import pandas as pd


DET = {"a", "an", "the"}
BE = {"am", "is", "are", "was", "were", "be", "been", "being"}
GET = {"get", "gets", "got", "gotten"}
AUX = BE | GET
IRREGULAR_PARTICIPLES = {
    "beat",
    "beaten",
    "bent",
    "bound",
    "bought",
    "brought",
    "built",
    "caught",
    "chosen",
    "done",
    "driven",
    "eaten",
    "fallen",
    "felt",
    "found",
    "given",
    "gone",
    "heard",
    "held",
    "hit",
    "hurt",
    "kept",
    "known",
    "left",
    "lost",
    "made",
    "met",
    "paid",
    "put",
    "read",
    "run",
    "said",
    "seen",
    "sent",
    "shot",
    "sold",
    "spent",
    "struck",
    "taught",
    "told",
    "understood",
    "won",
    "written",
}
COPULAR_COMPLEMENTS = {
    "a",
    "an",
    "the",
    "my",
    "your",
    "his",
    "her",
    "our",
    "their",
    "this",
    "that",
}
TOKEN_RE = re.compile(r"[A-Za-z'-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add conservative sentence-only prefill labels for obvious full active/passive clauses, "
            "leaving ambiguous rows blank."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=(
            Path("behavioral_results")
            / "experiment-2"
            / "llama323binstruct"
            / "experiment-2_generation_audit_lexically_controlled"
            / "all_other_generations_for_manual_review_merged.csv"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def is_participle(tok: str) -> bool:
    return tok in IRREGULAR_PARTICIPLES or tok.endswith(("ed", "en"))


def is_finite_active_verb(tok: str) -> bool:
    if tok in AUX:
        return False
    if tok.endswith("ing"):
        return False
    if is_participle(tok):
        return True
    if tok.endswith("s") and len(tok) > 3:
        return True
    return False


def looks_like_nominal_or_copular_followup(seq: list[str]) -> bool:
    if not seq:
        return True
    first = seq[0]
    if first in COPULAR_COMPLEMENTS:
        return True
    if first.endswith("ly"):
        return True
    return False


def classify_easy_clause(sentence: str) -> tuple[str, str]:
    text = str(sentence).strip()
    if not text:
        return "", "empty"
    if "," in text or ";" in text or ":" in text:
        return "", "multi_clause_or_punct"

    toks = tokens(text)
    if len(toks) < 3:
        return "", "too_short"
    if toks[0] not in DET:
        return "", "noncanonical_start"

    # Try to find the first verbal pivot after a short initial NP.
    pivot = None
    for i in range(1, min(len(toks), 7)):
        tok = toks[i]
        if tok in AUX or is_finite_active_verb(tok):
            pivot = i
            break
    if pivot is None:
        return "", "no_clear_verb"

    verb = toks[pivot]

    if verb in AUX:
        remainder = toks[pivot + 1 :]
        if not remainder:
            return "", "aux_no_remainder"
        if looks_like_nominal_or_copular_followup(remainder):
            return "", "copular_or_nominal"
        if remainder[0] == "being" and len(remainder) > 1 and is_participle(remainder[1]):
            return "passive", "aux_being_participle"
        if remainder[0] == "been" and len(remainder) > 1 and is_participle(remainder[1]):
            return "passive", "aux_been_participle"
        if is_participle(remainder[0]):
            return "passive", "aux_participle"
        if remainder[0].endswith("ing"):
            return "", "progressive_or_other"
        return "", "aux_ambiguous"

    if pivot >= len(toks) - 1:
        return "", "verb_final"
    return "active", "finite_lexical_verb"


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_csv = args.output_csv.resolve() if args.output_csv else input_csv.with_name(
        input_csv.stem + "_easy_prefill.csv"
    )

    frame = pd.read_csv(input_csv)
    labels_and_reasons = frame["greedy_answer_first_sentence"].map(classify_easy_clause)
    frame["assistant_easy_full_clause_label"] = labels_and_reasons.map(lambda pair: pair[0])
    frame["assistant_easy_full_clause_reason"] = labels_and_reasons.map(lambda pair: pair[1])

    if "manual_structure_label_strict" in frame.columns:
        strict = frame["manual_structure_label_strict"].fillna("")
        frame["assistant_easy_suggested_if_blank_strict"] = [
            label if manual == "" else ""
            for label, manual in zip(frame["assistant_easy_full_clause_label"], strict)
        ]
    if "manual_structure_label_lenient" in frame.columns:
        lenient = frame["manual_structure_label_lenient"].fillna("")
        frame["assistant_easy_suggested_if_blank_lenient"] = [
            label if manual == "" else ""
            for label, manual in zip(frame["assistant_easy_full_clause_label"], lenient)
        ]

    priority = [
        "domain",
        "item_index",
        "prompt_column",
        "prime_condition",
        "greedy_answer_first_sentence",
        "assistant_easy_full_clause_label",
        "assistant_easy_full_clause_reason",
        "assistant_easy_suggested_if_blank_strict",
        "manual_structure_label_strict",
        "review_remaining_strict",
        "assistant_easy_suggested_if_blank_lenient",
        "manual_structure_label_lenient",
        "review_remaining_lenient",
        "assistant_structure_label",
        "notes",
        "notes2",
        "prompt leak",
        "manual_notes",
        "generation_class",
        "prompt",
        "greedy_completion_raw",
    ]
    ordered = [col for col in priority if col in frame.columns]
    ordered.extend(col for col in frame.columns if col not in ordered)
    frame = frame[ordered]
    frame.to_csv(output_csv, index=False)

    counts = frame["assistant_easy_full_clause_label"].fillna("").value_counts(dropna=False)
    print(f"Wrote {output_csv}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
