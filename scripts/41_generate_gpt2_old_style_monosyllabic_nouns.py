#!/usr/bin/env python3
"""Generate GPT-2-large single-token, old-style monosyllabic Jabberwocky noun candidates."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable

import pandas as pd
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_VOCAB_JSON = REPO_ROOT / "corpora/transitive/vocabulary_lists/jabberwocky_transitive_strict_vocabulary.json"
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT / "corpora/transitive/vocabulary_lists/jabberwocky_gpt2_old_style_monosyllabic_noun_candidates.csv"
)
DEFAULT_OUTPUT_JSON = (
    REPO_ROOT / "corpora/transitive/vocabulary_lists/jabberwocky_gpt2_old_style_monosyllabic_noun_candidates_summary.json"
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
    # Common abbreviations/internet forms that are poor nonce nouns despite
    # not always appearing in /usr/share/dict/words.
    "def",
    "lol",
    "mom",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate old-style CVC/CCVC/CVCC nonce noun candidates and retain forms "
            "that are exactly one leading-space GPT-2-large token."
        )
    )
    parser.add_argument("--old-vocab-json", type=Path, default=OLD_VOCAB_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--target-count", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


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


def levenshtein(a: str, b: str, max_distance: int) -> int | None:
    if abs(len(a) - len(b)) > max_distance:
        return None
    previous = list(range(len(b) + 1))
    for index_a, char_a in enumerate(a, start=1):
        current = [index_a]
        row_min = index_a
        for index_b, char_b in enumerate(b, start=1):
            value = min(
                previous[index_b] + 1,
                current[index_b - 1] + 1,
                previous[index_b - 1] + int(char_a != char_b),
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return None
        previous = current
    distance = previous[-1]
    return distance if distance <= max_distance else None


def too_close_to_lexicon(candidate: str, lexicon: set[str], max_distance: int = 1) -> bool:
    for word in lexicon:
        if abs(len(candidate) - len(word)) > max_distance:
            continue
        if candidate[:1] != word[:1] and candidate[-1:] != word[-1:]:
            continue
        if levenshtein(candidate, word, max_distance=max_distance) is not None:
            return True
    return False


def old_style_units(old_vocab: dict[str, object]) -> tuple[list[str], list[str], list[str], list[str]]:
    words = list(old_vocab["nouns"]) + list(old_vocab["verb_stems"])
    onsets = sorted({word[:3] for word in words})
    middles = sorted({word[3:6] for word in words})
    codas = sorted({word[6:9] for word in words})

    # Also allow onset/coda fragments that GPT-2 already extracts from the old vocabulary.
    short_onsets = sorted({unit[:2] for unit in onsets} | {unit for unit in onsets if len(unit) == 3})
    short_codas = sorted({unit[-2:] for unit in codas} | {unit for unit in codas if len(unit) == 3})
    return onsets, middles, codas, sorted(set(short_onsets + short_codas))


def candidate_stream(old_vocab: dict[str, object], rng: random.Random) -> Iterable[tuple[str, str]]:
    onsets, middles, codas, fragments = old_style_units(old_vocab)
    seen: set[str] = set()

    templates = []
    for onset in onsets:
        for coda in codas:
            templates.append((onset + coda[-2:], "old_onset_plus_coda_tail"))
    for onset in onsets:
        for middle in middles:
            templates.append((onset[:2] + middle[-1], "old_onset_head_plus_middle_tail"))
    for middle in middles:
        for coda in codas:
            templates.append((middle[:1] + coda[-2:], "old_middle_head_plus_coda_tail"))
    for fragment in fragments:
        templates.append((fragment, "observed_old_fragment"))

    rng.shuffle(templates)
    for candidate, source_template in templates:
        candidate = candidate.lower()
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate, source_template


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    old_vocab = json.loads(args.old_vocab_json.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    english_words = load_dictionary()
    primelm_vocab = load_primelm_vocabulary()
    lexicon = english_words | primelm_vocab

    rows = []
    for candidate, source_template in candidate_stream(old_vocab, rng):
        if len(rows) >= args.max_candidates:
            break
        if not candidate.isalpha() or not 3 <= len(candidate) <= 5:
            continue

        ids = tokenizer.encode(" " + candidate, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        risk_flags = []
        if len(ids) != 1:
            risk_flags.append("not_one_gpt2_token")
        if tokens and tokens[0].replace("Ġ", "").lower() != candidate:
            risk_flags.append("token_surface_mismatch")
        if candidate in FUNCTION_OR_MORPHEME_LIKE:
            risk_flags.append("function_or_morpheme_like")
        if candidate in english_words:
            risk_flags.append("exact_dictionary_word")
        if candidate in primelm_vocab:
            risk_flags.append("exact_primelm_word")
        if candidate.endswith(("ed", "ing", "s")):
            risk_flags.append("suffix_like")
        review_flags = []
        if too_close_to_lexicon(candidate, lexicon, max_distance=1):
            review_flags.append("edit_distance_1_from_lexicon")

        hard_reject = [
            flag
            for flag in risk_flags
            if flag
            in {
                "not_one_gpt2_token",
                "token_surface_mismatch",
                "function_or_morpheme_like",
                "exact_dictionary_word",
                "exact_primelm_word",
                "suffix_like",
            }
        ]
        if hard_reject:
            recommendation = "reject"
        elif review_flags:
            recommendation = "review_candidate"
        else:
            recommendation = "strict_candidate"

        rows.append(
            {
                "candidate": candidate,
                "source_template": source_template,
                "gpt2_token_count": len(ids),
                "gpt2_token_id": int(ids[0]) if len(ids) == 1 else "",
                "gpt2_token": tokens[0] if len(tokens) == 1 else " ".join(tokens),
                "recommendation": recommendation,
                "risk_flags": ";".join(risk_flags + review_flags),
            }
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["recommendation", "source_template", "candidate"],
        ascending=[False, True, True],
    )
    selected = frame.loc[
        frame["recommendation"].isin(["strict_candidate", "review_candidate"])
    ].head(args.target_count)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)

    summary = {
        "model": args.model_name,
        "source_vocab": str(args.old_vocab_json.relative_to(REPO_ROOT)),
        "output_csv": str(args.output_csv.relative_to(REPO_ROOT)),
        "n_generated_candidates": int(len(frame)),
        "n_strict_candidates": int(frame["recommendation"].eq("strict_candidate").sum()),
        "n_review_candidates": int(frame["recommendation"].eq("review_candidate").sum()),
        "target_count": int(args.target_count),
        "selected_candidates": selected["candidate"].tolist(),
        "recommendation_counts": {
            key: int(value)
            for key, value in frame["recommendation"].value_counts().sort_index().items()
        },
        "notes": [
            "Candidates are generated from old Jabberwocky onset/middle/coda fragments.",
            "Strict candidates must be exactly one leading-space GPT-2-large token.",
            "Strict candidates reject dictionary/PrimeLM exact words, obvious function/morpheme pieces, and suffix-like strings.",
            "Edit-distance-1 lexical neighbors are retained as review candidates rather than hard-rejected because nearly all short monosyllables have a close lexical neighbor.",
        ],
    }
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
