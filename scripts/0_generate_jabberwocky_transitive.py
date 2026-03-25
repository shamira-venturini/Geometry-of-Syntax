import csv
import json
import random
import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
VOCAB_PATH = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_strict_vocabulary.json"
OUTPUT_PATH = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive.csv"
NUM_SAMPLES = 15000
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate jabberwocky_transitive Jabberwocky corpora from a vetted vocabulary."
    )
    parser.add_argument("--vocab-path", type=Path, default=VOCAB_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def load_dictionary() -> Set[str]:
    words: Set[str] = set()
    with Path("/usr/share/dict/words").open() as handle:
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


def candidate_pool(word: str, lexicon: Set[str]) -> List[str]:
    pool = []
    for candidate in lexicon:
        if abs(len(candidate) - len(word)) > 2:
            continue
        if candidate[:1] == word[:1] or candidate[:2] == word[:2] or candidate[-2:] == word[-2:]:
            pool.append(candidate)
    return pool if pool else [candidate for candidate in lexicon if abs(len(candidate) - len(word)) <= 2]


def validate_surface_form(word: str, english_words: Set[str], primelm_vocab: Set[str], max_distance: int = 2) -> None:
    if word in english_words or word in primelm_vocab:
        raise ValueError(f"Vocabulary item overlaps exactly with existing lexicon: {word}")

    lexicon = english_words.union(primelm_vocab)
    for candidate in candidate_pool(word, lexicon):
        if levenshtein(word, candidate, max_distance=max_distance) is not None:
            raise ValueError(f"Vocabulary item too close to existing word: {word} ~ {candidate}")


def validate_vocabulary(nouns: Sequence[str], verb_stems: Sequence[str]) -> None:
    english_words = load_dictionary()
    primelm_vocab = load_primelm_vocabulary()

    all_items = list(nouns) + list(verb_stems)
    if len(all_items) != len(set(all_items)):
        raise ValueError("Vocabulary contains duplicates across nouns and verb stems.")

    for noun in nouns:
        validate_surface_form(noun, english_words, primelm_vocab)

    for stem in verb_stems:
        for form in (stem, stem + "s", stem + "ed"):
            validate_surface_form(form, english_words, primelm_vocab)


def load_vocabulary(vocab_path: Path) -> Tuple[List[str], List[str]]:
    payload = json.loads(vocab_path.read_text())
    nouns = payload["nouns"]
    verb_stems = payload["verb_stems"]
    validate_vocabulary(nouns, verb_stems)
    return nouns, verb_stems


def inflect_active(stem: str, tense: str) -> str:
    if tense == "present":
        return stem + "s"
    return stem + "ed"


def inflect_passive_participle(stem: str) -> str:
    return stem + "ed"


def generate_transitive_row(rng: random.Random, nouns: Sequence[str], verb_stems: Sequence[str]) -> List[str]:
    p_n1, p_n2, t_n1, t_n2 = rng.sample(list(nouns), 4)
    p_stem, t_stem = rng.sample(list(verb_stems), 2)

    if rng.random() > 0.5:
        p_tense, t_tense = "present", "past"
        p_aux, t_aux = "is", "was"
    else:
        p_tense, t_tense = "past", "present"
        p_aux, t_aux = "was", "is"

    if rng.random() > 0.5:
        p_det, t_det = "the", "a"
    else:
        p_det, t_det = "a", "the"

    p_v_act = inflect_active(p_stem, p_tense)
    t_v_act = inflect_active(t_stem, t_tense)
    p_v_pass = inflect_passive_participle(p_stem)
    t_v_pass = inflect_passive_participle(t_stem)

    pa = f"{p_det} {p_n1} {p_v_act} {p_det} {p_n2} ."
    pp = f"{p_det} {p_n2} {p_aux} {p_v_pass} by {p_det} {p_n1} ."
    ta = f"{t_det} {t_n1} {t_v_act} {t_det} {t_n2} ."
    tp = f"{t_det} {t_n2} {t_aux} {t_v_pass} by {t_det} {t_n1} ."
    return [pa, pp, ta, tp]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    nouns, verb_stems = load_vocabulary(args.vocab_path)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pa", "pp", "ta", "tp"])
        for _ in range(args.num_samples):
            writer.writerow(generate_transitive_row(rng, nouns, verb_stems))

    print(f"Saved {args.num_samples} strict Jabberwocky jabberwocky_transitive rows to {args.output_path}")


if __name__ == "__main__":
    main()
