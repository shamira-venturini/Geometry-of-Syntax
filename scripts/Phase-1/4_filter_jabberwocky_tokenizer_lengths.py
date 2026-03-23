import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Set

import pandas as pd
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOCAB_PATH = REPO_ROOT / "corpora" / "jabberwocky_transitive" / "jabberwocky_transitive_strict_vocabulary.json"
DEFAULT_OUTPUT_VOCAB = REPO_ROOT / "corpora" / "jabberwocky_transitive" / "jabberwocky_transitive_strict_bpe_filtered_vocabulary.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "behavioral_results" / "jabberwocky_tokenizer_length_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and filter a strict Jabberwocky vocabulary by GPT-2 BPE token length."
    )
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--output-vocab", type=Path, default=DEFAULT_OUTPUT_VOCAB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--noun-lengths", default="4,5")
    parser.add_argument("--verb-present-lengths", default="4,5")
    parser.add_argument("--verb-past-lengths", default="4,5")
    parser.add_argument("--min-nouns", type=int, default=40)
    parser.add_argument("--min-verb-stems", type=int, default=24)
    return parser.parse_args()


def parse_allowed_lengths(raw: str) -> Set[int]:
    return {int(piece.strip()) for piece in raw.split(",") if piece.strip()}


def token_count(tokenizer, word: str) -> int:
    return len(tokenizer.encode(" " + word, add_special_tokens=False))


def summarize_counts(values: Sequence[int]) -> str:
    counts = pd.Series(values).value_counts().sort_index()
    return ", ".join(f"{int(length)}:{int(count)}" for length, count in counts.items())


def noun_rows(tokenizer, nouns: Iterable[str]) -> List[dict]:
    rows = []
    for noun in nouns:
        rows.append(
            {
                "word": noun,
                "token_count": token_count(tokenizer, noun),
            }
        )
    return rows


def verb_rows(tokenizer, stems: Iterable[str]) -> List[dict]:
    rows = []
    for stem in stems:
        present = stem + "s"
        past = stem + "ed"
        rows.append(
            {
                "stem": stem,
                "present_form": present,
                "past_form": past,
                "stem_token_count": token_count(tokenizer, stem),
                "present_token_count": token_count(tokenizer, present),
                "past_token_count": token_count(tokenizer, past),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    noun_lengths = parse_allowed_lengths(args.noun_lengths)
    verb_present_lengths = parse_allowed_lengths(args.verb_present_lengths)
    verb_past_lengths = parse_allowed_lengths(args.verb_past_lengths)

    payload = json.loads(args.vocab_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    noun_frame = pd.DataFrame(noun_rows(tokenizer, payload["nouns"]))
    noun_frame["keep"] = noun_frame["token_count"].isin(noun_lengths)

    verb_frame = pd.DataFrame(verb_rows(tokenizer, payload["verb_stems"]))
    verb_frame["keep"] = (
        verb_frame["present_token_count"].isin(verb_present_lengths)
        & verb_frame["past_token_count"].isin(verb_past_lengths)
    )

    kept_nouns = noun_frame.loc[noun_frame["keep"], "word"].tolist()
    kept_stems = verb_frame.loc[verb_frame["keep"], "stem"].tolist()

    if len(kept_nouns) < args.min_nouns:
        raise ValueError(f"Filtering left too few nouns: {len(kept_nouns)} < {args.min_nouns}")
    if len(kept_stems) < args.min_verb_stems:
        raise ValueError(f"Filtering left too few verb stems: {len(kept_stems)} < {args.min_verb_stems}")

    filtered_payload = {
        "metadata": {
            "description": "Strict jabberwocky_transitive Jabberwocky vocabulary filtered by GPT-2 BPE token length.",
            "source_vocabulary": str(args.vocab_path.relative_to(REPO_ROOT)),
            "model_name": args.model_name,
            "noun_allowed_lengths": sorted(noun_lengths),
            "verb_present_allowed_lengths": sorted(verb_present_lengths),
            "verb_past_allowed_lengths": sorted(verb_past_lengths),
            "noun_count_before": int(len(noun_frame)),
            "noun_count_after": int(len(kept_nouns)),
            "verb_stem_count_before": int(len(verb_frame)),
            "verb_stem_count_after": int(len(kept_stems)),
        },
        "nouns": kept_nouns,
        "verb_stems": kept_stems,
    }

    summary = pd.DataFrame(
        [
            {
                "model_name": args.model_name,
                "noun_allowed_lengths": ",".join(str(length) for length in sorted(noun_lengths)),
                "verb_present_allowed_lengths": ",".join(
                    str(length) for length in sorted(verb_present_lengths)
                ),
                "verb_past_allowed_lengths": ",".join(
                    str(length) for length in sorted(verb_past_lengths)
                ),
                "noun_count_before": int(len(noun_frame)),
                "noun_count_after": int(len(kept_nouns)),
                "verb_stem_count_before": int(len(verb_frame)),
                "verb_stem_count_after": int(len(kept_stems)),
                "noun_token_count_distribution_before": summarize_counts(noun_frame["token_count"].tolist()),
                "noun_token_count_distribution_after": summarize_counts(
                    noun_frame.loc[noun_frame["keep"], "token_count"].tolist()
                ),
                "verb_present_distribution_before": summarize_counts(
                    verb_frame["present_token_count"].tolist()
                ),
                "verb_present_distribution_after": summarize_counts(
                    verb_frame.loc[verb_frame["keep"], "present_token_count"].tolist()
                ),
                "verb_past_distribution_before": summarize_counts(
                    verb_frame["past_token_count"].tolist()
                ),
                "verb_past_distribution_after": summarize_counts(
                    verb_frame.loc[verb_frame["keep"], "past_token_count"].tolist()
                ),
            }
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_vocab.parent.mkdir(parents=True, exist_ok=True)
    noun_frame.to_csv(args.output_dir / "noun_token_lengths.csv", index=False)
    verb_frame.to_csv(args.output_dir / "verb_token_lengths.csv", index=False)
    summary.to_csv(args.output_dir / "tokenizer_length_summary.csv", index=False)
    args.output_vocab.write_text(json.dumps(filtered_payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
