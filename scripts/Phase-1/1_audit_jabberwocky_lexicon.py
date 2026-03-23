import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "corpora" / "jabberwocky_transitive" / "jabberwocky_transitive.csv"
DEFAULT_DICT = Path("/usr/share/dict/words")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "behavioral_results" / "jabberwocky_lexicon_audit"
FUNCTION_WORDS = {"the", "a", "is", "was", "by", "."}
COMMON_SUFFIXES = ("ed", "es", "s", "ing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Jabberwocky content words for overlap with real English words."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--dict-path", type=Path, default=DEFAULT_DICT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-distance", type=int, default=2)
    return parser.parse_args()


def load_dictionary(path: Path) -> Set[str]:
    words: Set[str] = set()
    with path.open() as handle:
        for line in handle:
            word = line.strip().lower()
            if word.isalpha():
                words.add(word)
    return words


def load_primelm_vocabulary() -> Set[str]:
    vocab: Set[str] = set()
    vocab_files = [
        REPO_ROOT / "PrimeLM" / "vocabulary_lists" / "nounlist_usf_freq.csv",
        REPO_ROOT / "PrimeLM" / "vocabulary_lists" / "verblist_T_usf_freq.csv",
        REPO_ROOT / "PrimeLM" / "vocabulary_lists" / "verblist_DT_usf_freq.csv",
        REPO_ROOT / "PrimeLM" / "vocabulary_lists" / "adjective_voc_list_USF.csv",
    ]
    for path in vocab_files:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for value in row.values():
                    if not value:
                        continue
                    cell = value.strip().lower()
                    if cell.isalpha():
                        vocab.add(cell)
    return vocab


def extract_content_words(frame: pd.DataFrame) -> List[str]:
    words: Set[str] = set()
    for column in frame.columns:
        for sentence in frame[column].astype(str):
            for token in sentence.split():
                token = token.lower()
                if token not in FUNCTION_WORDS and token.isalpha():
                    words.add(token)
    return sorted(words)


def possible_stems(word: str) -> List[str]:
    stems = [word]
    for suffix in COMMON_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            base = word[: -len(suffix)]
            stems.append(base)
            if suffix == "ed" and base.endswith("i"):
                stems.append(base[:-1] + "y")
            if suffix == "s" and base.endswith("e"):
                stems.append(base[:-1])
    return list(dict.fromkeys(stems))


def levenshtein(a: str, b: str, max_distance: int) -> Optional[int]:
    if abs(len(a) - len(b)) > max_distance:
        return None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        row_min = curr[0]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            value = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
            curr.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return None
        prev = curr
    distance = prev[-1]
    return distance if distance <= max_distance else None


def nearest_english_words(word: str, english_words: Sequence[str], max_distance: int) -> List[Tuple[str, int]]:
    matches: List[Tuple[str, int]] = []
    for candidate in english_words:
        if candidate == word:
            continue
        distance = levenshtein(word, candidate, max_distance=max_distance)
        if distance is not None:
            matches.append((candidate, distance))
    matches.sort(key=lambda item: (item[1], abs(len(item[0]) - len(word)), item[0]))
    return matches[:5]


def candidate_pool(word: str, english_words: Set[str], primelm_vocab: Set[str]) -> List[str]:
    pool = set()
    target_lengths = {len(word) - 2, len(word) - 1, len(word), len(word) + 1, len(word) + 2}
    initial = word[:1]
    for source in (english_words, primelm_vocab):
        for candidate in source:
            if len(candidate) in target_lengths and candidate[:1] == initial:
                pool.add(candidate)
    if not pool:
        for source in (english_words, primelm_vocab):
            for candidate in source:
                if len(candidate) in target_lengths:
                    pool.add(candidate)
    return sorted(pool)


def audit_word(word: str, english_words: Set[str], primelm_vocab: Set[str], max_distance: int) -> Dict[str, object]:
    stems = possible_stems(word)
    exact_english = word in english_words
    exact_primelm = word in primelm_vocab
    english_stems = [stem for stem in stems if stem in english_words and stem != word]
    primelm_stems = [stem for stem in stems if stem in primelm_vocab and stem != word]
    neighbors = nearest_english_words(word, candidate_pool(word, english_words, primelm_vocab), max_distance=max_distance)

    return {
        "word": word,
        "exact_english_word": exact_english,
        "exact_primelm_vocab": exact_primelm,
        "english_stem_overlap": ";".join(english_stems),
        "primelm_stem_overlap": ";".join(primelm_stems),
        "nearest_neighbors": ";".join(f"{neighbor}:{distance}" for neighbor, distance in neighbors),
        "min_neighbor_distance": neighbors[0][1] if neighbors else None,
    }


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.corpus)
    frame.columns = [column.strip().lower() for column in frame.columns]
    english_words = load_dictionary(args.dict_path)
    primelm_vocab = load_primelm_vocabulary()
    content_words = extract_content_words(frame)

    rows = [
        audit_word(
            word=word,
            english_words=english_words,
            primelm_vocab=primelm_vocab,
            max_distance=args.max_distance,
        )
        for word in content_words
    ]
    results = pd.DataFrame(rows)

    summary = pd.DataFrame(
        [
            {
                "n_content_words": len(results),
                "n_exact_english_words": int(results["exact_english_word"].sum()),
                "n_exact_primelm_vocab": int(results["exact_primelm_vocab"].sum()),
                "n_with_english_stem_overlap": int((results["english_stem_overlap"] != "").sum()),
                "n_with_primelm_stem_overlap": int((results["primelm_stem_overlap"] != "").sum()),
                "n_with_neighbor_distance_le_1": int((results["min_neighbor_distance"] <= 1).fillna(False).sum()),
                "n_with_neighbor_distance_le_2": int((results["min_neighbor_distance"] <= 2).fillna(False).sum()),
            }
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "jabberwocky_lexicon_audit.csv", index=False)
    summary.to_csv(args.output_dir / "jabberwocky_lexicon_summary.csv", index=False)


if __name__ == "__main__":
    main()
