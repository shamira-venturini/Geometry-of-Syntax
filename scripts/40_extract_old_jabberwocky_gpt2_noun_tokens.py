#!/usr/bin/env python3
"""Extract GPT-2-large single-token noun candidates from the old Jabberwocky vocabulary."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
VOCAB_JSON = REPO_ROOT / "corpora/transitive/vocabulary_lists/jabberwocky_transitive_strict_vocabulary.json"
OUT_CSV = REPO_ROOT / "corpora/transitive/vocabulary_lists/jabberwocky_old_gpt2_single_token_noun_candidates.csv"
OUT_SUMMARY = (
    REPO_ROOT
    / "corpora/transitive/vocabulary_lists/jabberwocky_old_gpt2_single_token_noun_candidates_summary.json"
)

MODEL_NAME = "gpt2-large"
FUNCTION_OR_MORPHEME_LIKE = {
    "a",
    "an",
    "as",
    "be",
    "by",
    "do",
    "ed",
    "es",
    "go",
    "he",
    "if",
    "in",
    "ing",
    "is",
    "it",
    "me",
    "my",
    "no",
    "of",
    "on",
    "or",
    "s",
    "so",
    "the",
    "to",
    "up",
    "us",
    "was",
    "we",
}


def load_dictionary() -> set[str]:
    path = Path("/usr/share/dict/words")
    if not path.exists():
        return set()
    return {line.strip().lower() for line in path.open() if line.strip().isalpha()}


def load_primelm_vocabulary() -> set[str]:
    vocab = set()
    for path in [
        REPO_ROOT / "PrimeLM/vocabulary_lists/nounlist_usf_freq.csv",
        REPO_ROOT / "PrimeLM/vocabulary_lists/verblist_T_usf_freq.csv",
        REPO_ROOT / "PrimeLM/vocabulary_lists/verblist_DT_usf_freq.csv",
        REPO_ROOT / "PrimeLM/vocabulary_lists/adjective_voc_list_USF.csv",
    ]:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                for value in row.values():
                    if value:
                        token = value.strip().lower()
                        if token.isalpha():
                            vocab.add(token)
    return vocab


def old_surface_forms(payload: dict[str, object]) -> Iterable[tuple[str, str, str]]:
    for noun in payload["nouns"]:
        yield "noun", str(noun), str(noun)
    for stem in payload["verb_stems"]:
        stem = str(stem)
        yield "verb_stem", stem, stem
        yield "verb_present", stem, f"{stem}s"
        yield "verb_past_participle", stem, f"{stem}ed"


def main() -> None:
    payload = json.loads(VOCAB_JSON.read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    english_words = load_dictionary()
    primelm_vocab = load_primelm_vocabulary()

    sources_by_piece: dict[str, list[str]] = defaultdict(list)
    token_id_by_piece: dict[str, int] = {}
    token_string_by_piece: dict[str, str] = {}

    for source_category, source_base, surface in old_surface_forms(payload):
        ids = tokenizer.encode(" " + surface, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        for token_id, token_string in zip(ids, tokens):
            piece = token_string.replace("Ġ", "").lower()
            if not piece.isalpha():
                continue
            standalone_ids = tokenizer.encode(" " + piece, add_special_tokens=False)
            standalone_tokens = tokenizer.convert_ids_to_tokens(standalone_ids)
            if len(standalone_ids) != 1:
                continue
            if standalone_tokens[0].replace("Ġ", "").lower() != piece:
                continue
            sources_by_piece[piece].append(f"{source_category}:{source_base}:{surface}")
            token_id_by_piece[piece] = int(standalone_ids[0])
            token_string_by_piece[piece] = standalone_tokens[0]

    rows = []
    for piece in sorted(sources_by_piece):
        reasons = []
        if len(piece) < 3:
            reasons.append("too_short")
        if piece in FUNCTION_OR_MORPHEME_LIKE:
            reasons.append("function_or_morpheme_like")
        if piece in english_words:
            reasons.append("exact_dictionary_word")
        if piece in primelm_vocab:
            reasons.append("exact_primelm_word")
        if piece.endswith(("ed", "ing", "s")):
            reasons.append("suffix_like")

        if not reasons:
            recommendation = "strict_candidate"
        elif set(reasons).issubset({"exact_dictionary_word"}):
            recommendation = "review_dictionary_only"
        else:
            recommendation = "reject"

        rows.append(
            {
                "candidate": piece,
                "gpt2_token_id": token_id_by_piece[piece],
                "gpt2_token": token_string_by_piece[piece],
                "length_chars": len(piece),
                "recommendation": recommendation,
                "risk_flags": ";".join(reasons),
                "n_old_sources": len(sources_by_piece[piece]),
                "old_sources_preview": ";".join(sources_by_piece[piece][:8]),
            }
        )

    frame = pd.DataFrame(rows).sort_values(
        ["recommendation", "length_chars", "candidate"],
        ascending=[True, False, True],
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_CSV, index=False)

    summary = {
        "model": MODEL_NAME,
        "source_vocab": str(VOCAB_JSON.relative_to(REPO_ROOT)),
        "output_csv": str(OUT_CSV.relative_to(REPO_ROOT)),
        "n_candidates_total": int(len(frame)),
        "recommendation_counts": {
            key: int(value)
            for key, value in frame["recommendation"].value_counts().sort_index().items()
        },
        "strict_candidates": frame.loc[
            frame["recommendation"].eq("strict_candidate"), "candidate"
        ].tolist(),
        "notes": [
            "Candidates are surface strings whose leading-space GPT-2-large encoding is exactly one token.",
            "They are extracted only from token pieces observed in the old Jabberwocky vocabulary/forms.",
            "The strict filter rejects exact dictionary/PrimeLM words, function-like pieces, suffix-like pieces, and pieces shorter than 3 characters.",
        ],
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
