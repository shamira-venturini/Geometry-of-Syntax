import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORE = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_strict_4cell_counterbalanced.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_fragment_verbs_gpt2large_strict_4cell.csv"
)
DEFAULT_LEXICON = (
    REPO_ROOT
    / "corpora"
    / "transitive"
    / "vocabulary_lists"
    / "jabberwocky_fragment_nouns_gpt2large_lexicon.json"
)
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "corpora"
    / "transitive"
    / "jabberwocky_transitive_fragment_verbs_gpt2large_strict_4cell_summary.json"
)

FUNCTION_WORDS = {"a", "an", "the", "is", "was", "by", ".", "s", "ed"}
VOWELS = set("aeiou")
ONSETS = [
    "",
    "b",
    "br",
    "bl",
    "ch",
    "cl",
    "cr",
    "d",
    "dr",
    "f",
    "fl",
    "fr",
    "g",
    "gl",
    "gr",
    "j",
    "k",
    "kl",
    "kr",
    "l",
    "m",
    "n",
    "p",
    "pl",
    "pr",
    "r",
    "s",
    "sk",
    "sl",
    "sm",
    "sn",
    "sp",
    "st",
    "t",
    "tr",
    "v",
    "w",
    "z",
]
NUCLEI = ["a", "e", "i", "o", "u", "ai", "ea", "ee", "oa", "oo"]
CODAS = ["", "b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "t", "v", "z", "ck", "ld", "lt", "mp", "nd", "nt", "rk", "st"]
NONFINAL_CODAS = ["", "", "", "l", "m", "n", "r"]
BAD_INTERNAL_BIGRAMS = {
    "bg",
    "bk",
    "bp",
    "db",
    "dk",
    "dt",
    "fb",
    "fd",
    "fg",
    "fk",
    "fp",
    "gb",
    "gd",
    "gf",
    "gk",
    "gp",
    "kg",
    "kp",
    "pb",
    "pd",
    "pg",
    "pk",
    "td",
    "tg",
    "tk",
}
SEMANTIC_STOPLIST = {
    "abdom",
    "accommod",
    "acqu",
    "acquies",
    "buquerque",
    "cannabin",
    "carcin",
    "cellul",
    "cellaneous",
    "chimpan",
    "circumst",
    "clamation",
    "commod",
    "commem",
    "compuls",
    "condol",
    "contempor",
    "corros",
    "culation",
    "dehuman",
    "dermat",
    "depl",
    "disadvant",
    "distribut",
    "ejac",
    "eluc",
    "euphem",
    "extravag",
    "fixme",
    "geop",
    "gmaxwell",
    "gregation",
    "halluc",
    "hypothal",
    "intrins",
    "neoc",
    "neurolog",
    "orchestr",
    "perties",
    "predis",
    "presupp",
    "prophe",
    "reciation",
    "recl",
    "rejo",
    "ruciating",
    "shorth",
    "simulac",
    "taxp",
    "theolog",
    "versible",
}


try:
    from wordfreq import zipf_frequency
except Exception:  # pragma: no cover
    zipf_frequency = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a tokenizer-aligned fragment-verb Jabberwocky corpus. "
            "Only nouns are nonwords; verb positions are standalone s/ed fragments."
        )
    )
    parser.add_argument("--core-csv", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lexicon-json", type=Path, default=DEFAULT_LEXICON)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--tokenizer-model", default="gpt2-large")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--use-auth-token",
        action="store_true",
        help="Use the Hugging Face token from notebook_login/huggingface-cli login for gated tokenizers.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-generated-candidates-per-noun", type=int, default=1000)
    parser.add_argument("--max-vocab-candidates-per-noun", type=int, default=2500)
    parser.add_argument(
        "--common-word-count",
        type=int,
        default=50000,
        help="Top English words used to reject semantically transparent fragments.",
    )
    parser.add_argument(
        "--allow-semantic-fragments",
        action="store_true",
        help="Disable rejection of tokenizer pieces that look like common English stems/fragments.",
    )
    return parser.parse_args()


def vowel_initial(word: str) -> bool:
    return word[:1].lower() in VOWELS


def cv_template(word: str) -> str:
    return "".join("V" if char in VOWELS else "C" for char in word.lower() if char.isalpha())


def syllable_count(word: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", word.lower())))


def levenshtein(a: str, b: str, max_distance: Optional[int] = None) -> int:
    if max_distance is not None and abs(len(a) - len(b)) > max_distance:
        return max_distance + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        row_min = current[0]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
            row_min = min(row_min, current[-1])
        if max_distance is not None and row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def load_tokenizer(model_name: str, auth_token: object, local_files_only: bool):
    from transformers import AutoTokenizer

    kwargs = {"local_files_only": local_files_only}
    if auth_token:
        kwargs["use_auth_token"] = auth_token
    return AutoTokenizer.from_pretrained(model_name, **kwargs)


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(" " + str(text), add_special_tokens=False))


def read_core(path: Path) -> pd.DataFrame:
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


def collect_core_nouns(frame: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
    nouns: Set[str] = set()
    surface_words: Set[str] = set()
    for row in frame.itertuples(index=False):
        for sentence in (row.pa, row.ta):
            det_a, agent, verb, det_p, patient = active_parts(str(sentence))
            nouns.update([agent.lower(), patient.lower()])
            surface_words.update([det_a.lower(), agent.lower(), verb.lower(), det_p.lower(), patient.lower()])
        for sentence in (row.pp, row.tp):
            det_p, patient, aux, participle, det_a, agent = passive_parts(str(sentence))
            nouns.update([agent.lower(), patient.lower()])
            surface_words.update([det_p.lower(), patient.lower(), aux.lower(), participle.lower(), det_a.lower(), agent.lower()])
    return nouns, surface_words


def load_english_words() -> Set[str]:
    words: Set[str] = set()
    path = Path("/usr/share/dict/words")
    if not path.exists():
        return words
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            token = line.strip().lower()
            if token.isascii() and token.isalpha():
                words.add(token)
    return words


def load_common_words(n: int) -> Set[str]:
    if zipf_frequency is None or n <= 0:
        return set()
    try:
        from wordfreq import top_n_list

        return {
            word.lower()
            for word in top_n_list("en", n, ascii_only=True)
            if word.isascii() and word.isalpha() and len(word) >= 4
        }
    except Exception:
        return set()


def build_common_word_index(common_words: Set[str]) -> Dict[str, Set[str]]:
    prefixes: Set[str] = set()
    contained_words: Set[str] = set()
    for word in common_words:
        if len(word) >= 7:
            for length in range(5, min(11, len(word) - 1) + 1):
                prefixes.add(word[:length])
        if 5 <= len(word) <= 11:
            contained_words.add(word)
    return {
        "words": set(common_words),
        "prefixes": prefixes,
        "contained_words": contained_words,
    }


KNOWN_WORD_CACHE: Dict[str, bool] = {}


def is_known_word(candidate: str, english_words: Set[str], project_words: Set[str]) -> bool:
    if candidate in KNOWN_WORD_CACHE:
        return KNOWN_WORD_CACHE[candidate]
    if candidate in english_words or candidate in project_words:
        KNOWN_WORD_CACHE[candidate] = True
        return True
    if zipf_frequency is None:
        KNOWN_WORD_CACHE[candidate] = False
        return False
    KNOWN_WORD_CACHE[candidate] = zipf_frequency(candidate, "en") >= 1.0
    return KNOWN_WORD_CACHE[candidate]


def too_close(candidate: str, source: str, project_words: Set[str]) -> bool:
    threshold = 1 if len(source) <= 5 else 2
    if levenshtein(candidate, source, threshold) <= threshold:
        return True
    for word in project_words:
        if word[:1] != candidate[:1]:
            continue
        if abs(len(candidate) - len(word)) <= threshold and levenshtein(candidate, word, threshold) <= threshold:
            return True
    return False


def normalize_vocab_piece(piece: str) -> Optional[str]:
    text = piece.strip().lower()
    for prefix in ("Ġ", "▁"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = text.strip().lower()
    if 3 <= len(text) <= 11 and text.isascii() and text.isalpha():
        return text
    return None


def tokenizer_vocab_candidates(tokenizer) -> List[str]:
    candidates: Set[str] = set()
    for token, index in tokenizer.get_vocab().items():
        for raw in (token, tokenizer.decode([index])):
            candidate = normalize_vocab_piece(raw)
            if candidate:
                candidates.add(candidate)
    return sorted(candidates)


def make_syllable(rng: random.Random, *, initial_vowel: Optional[bool], final_syllable: bool) -> str:
    if initial_vowel is True:
        onset = ""
    elif initial_vowel is False:
        onset = rng.choice([item for item in ONSETS if item])
    else:
        onset = rng.choice(ONSETS)
    nucleus = rng.choice(NUCLEI)
    coda = rng.choice(CODAS if final_syllable else NONFINAL_CODAS)
    return onset + nucleus + coda


def generated_candidate(rng: random.Random, source: str) -> str:
    pieces = []
    n_syllables = syllable_count(source)
    for index in range(n_syllables):
        pieces.append(
            make_syllable(
                rng,
                initial_vowel=vowel_initial(source) if index == 0 else None,
                final_syllable=index == n_syllables - 1,
            )
        )
    return "".join(pieces)


def semantic_fragment_reason(candidate: str, common_index: Mapping[str, Set[str]]) -> Optional[str]:
    if not common_index:
        return None
    if candidate in SEMANTIC_STOPLIST:
        return "semantic_stoplist"
    common_words = common_index.get("words", set())
    if candidate in common_words:
        return "common_word"
    if len(candidate) >= 5 and candidate in common_index.get("prefixes", set()):
        return "prefix_of_common_word"
    contained_words = common_index.get("contained_words", set())
    for start in range(len(candidate)):
        for end in range(start + 5, len(candidate) + 1):
            piece = candidate[start:end]
            if piece in contained_words:
                return f"contains_common_word:{piece}"
    return None


def valid_candidate(
    candidate: str,
    *,
    source: str,
    used: Set[str],
    english_words: Set[str],
    project_words: Set[str],
    common_index: Mapping[str, Set[str]],
    allow_semantic_fragments: bool,
) -> bool:
    if not (candidate.isascii() and candidate.isalpha()):
        return False
    if candidate in used:
        return False
    if candidate in FUNCTION_WORDS:
        return False
    if vowel_initial(candidate) != vowel_initial(source):
        return False
    if is_known_word(candidate, english_words, project_words):
        return False
    if not allow_semantic_fragments and semantic_fragment_reason(candidate, common_index):
        return False
    if too_close(candidate, source, project_words):
        return False
    return True


def candidate_score(tokenizer, candidate: str, source: str, origin_penalty: int) -> Tuple[int, int, int, int, int, int, int, str]:
    source_tokens = token_count(tokenizer, source)
    candidate_tokens = token_count(tokenizer, candidate)
    bad_bigram_count = sum(candidate.count(bigram) for bigram in BAD_INTERNAL_BIGRAMS)
    generated_like_bonus = 0 if any(vowel in candidate for vowel in "aeiou") else 1
    return (
        abs(candidate_tokens - source_tokens),
        max(0, candidate_tokens - source_tokens),
        origin_penalty,
        bad_bigram_count,
        generated_like_bonus,
        abs(len(candidate) - len(source)),
        levenshtein(cv_template(candidate), cv_template(source)),
        candidate,
    )


def choose_nonce_noun(
    source: str,
    *,
    tokenizer,
    vocab_candidates: Sequence[str],
    used: Set[str],
    english_words: Set[str],
    project_words: Set[str],
    common_index: Mapping[str, Set[str]],
    allow_semantic_fragments: bool,
    seed: int,
    max_vocab_candidates: int,
    max_generated_candidates: int,
) -> str:
    rng = random.Random(f"{seed}:noun:{source}")
    raw_vocab_candidates = [candidate for candidate in vocab_candidates if vowel_initial(candidate) == vowel_initial(source)]
    rng.shuffle(raw_vocab_candidates)
    raw_candidates: List[Tuple[str, int]] = [
        (candidate, 1) for candidate in raw_vocab_candidates[:max_vocab_candidates]
    ]
    for _ in range(max_generated_candidates):
        raw_candidates.append((generated_candidate(rng, source), 0))

    best: Optional[Tuple[Tuple[int, int, int, int, int, int, int, str], str]] = None
    for candidate, origin_penalty in raw_candidates:
        if not valid_candidate(
            candidate,
            source=source,
            used=used,
            english_words=english_words,
            project_words=project_words,
            common_index=common_index,
            allow_semantic_fragments=allow_semantic_fragments,
        ):
            continue
        score = candidate_score(tokenizer, candidate, source, origin_penalty)
        if best is None or score < best[0]:
            best = (score, candidate)

    if best is None:
        raise RuntimeError(f"Could not find a tokenizer-aligned nonce noun for {source!r}.")
    used.add(best[1])
    return best[1]


def active_fragment(active_sentence: str, passive_sentence: str, noun_map: Mapping[str, str]) -> str:
    det_agent, agent, _, det_patient, patient = active_parts(active_sentence)
    _, _, aux, _, _, _ = passive_parts(passive_sentence)
    fragment = {"is": "s", "was": "ed"}.get(aux)
    if fragment is None:
        raise ValueError(f"Unsupported passive auxiliary for tense inference: {aux}")
    return f"{det_agent} {noun_map[agent.lower()]} {fragment} {det_patient} {noun_map[patient.lower()]} ."


def passive_fragment(passive_sentence: str, noun_map: Mapping[str, str]) -> str:
    det_patient, patient, aux, _, det_agent, agent = passive_parts(passive_sentence)
    return f"{det_patient} {noun_map[patient.lower()]} {aux} ed by {det_agent} {noun_map[agent.lower()]} ."


def build_fragment_frame(core: pd.DataFrame, noun_map: Mapping[str, str]) -> pd.DataFrame:
    rows = []
    for row in core.itertuples(index=False):
        rows.append(
            {
                "pa": active_fragment(str(row.pa), str(row.pp), noun_map),
                "pp": passive_fragment(str(row.pp), noun_map),
                "ta": active_fragment(str(row.ta), str(row.tp), noun_map),
                "tp": passive_fragment(str(row.tp), noun_map),
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
    bad_article_rows = 0
    for row in frame.itertuples(index=False):
        pa_det_a, pa_agent, pa_fragment, pa_det_p, pa_patient = active_parts(str(row.pa))
        ta_det_a, ta_agent, ta_fragment, ta_det_p, ta_patient = active_parts(str(row.ta))
        _, _, pp_aux, _, _, _ = passive_parts(str(row.pp))
        _, _, tp_aux, _, _, _ = passive_parts(str(row.tp))
        same_aux_rows += int(pp_aux == tp_aux)
        shared_noun_rows += int(bool({pa_agent, pa_patient} & {ta_agent, ta_patient}))
        same_active_fragment_rows += int(pa_fragment == ta_fragment)
        same_det_family_rows += int(det_family(pa_det_a) == det_family(ta_det_a))
        for det, noun in ((pa_det_a, pa_agent), (pa_det_p, pa_patient), (ta_det_a, ta_agent), (ta_det_p, ta_patient)):
            bad_article_rows += int((det == "an" and not vowel_initial(noun)) or (det == "a" and vowel_initial(noun)))
    return {
        "bad_indefinite_article_uses": bad_article_rows,
        "same_active_fragment_rows": same_active_fragment_rows,
        "same_aux_rows": same_aux_rows,
        "same_det_family_rows": same_det_family_rows,
        "shared_noun_rows": shared_noun_rows,
    }


def safe_stats(values: Sequence[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def tokenizer_audit(core: pd.DataFrame, fragment: pd.DataFrame, tokenizer, model_name: str) -> Dict[str, object]:
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


def noun_tokenization_audit(noun_map: Mapping[str, str], tokenizer, model_name: str) -> Dict[str, object]:
    rows = []
    for source, nonce in sorted(noun_map.items()):
        source_tokens = token_count(tokenizer, source)
        nonce_tokens = token_count(tokenizer, nonce)
        rows.append(
            {
                "source": source,
                "nonce": nonce,
                "source_tokens": source_tokens,
                "nonce_tokens": nonce_tokens,
                "difference": nonce_tokens - source_tokens,
            }
        )
    diffs = [row["difference"] for row in rows]
    return {
        "status": "ok",
        "model": model_name,
        "differences": safe_stats(diffs),
        "exact_matches": int(sum(diff == 0 for diff in diffs)),
        "nonzero_preview": [row for row in rows if row["difference"] != 0][:25],
        "preview": rows[:20],
    }


def fail_if_bad(summary: Mapping[str, object]) -> None:
    audit = summary["row_constraint_audit"]
    for key in ["bad_indefinite_article_uses", "same_aux_rows", "shared_noun_rows", "same_det_family_rows"]:
        if int(audit[key]) != 0:
            raise RuntimeError(f"Tokenizer-aligned fragment corpus audit failed: {key}={audit[key]}")


def main() -> None:
    args = parse_args()
    auth_token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or (True if args.use_auth_token else None)
    )
    tokenizer = load_tokenizer(args.tokenizer_model, auth_token, args.local_files_only)
    core = read_core(args.core_csv)
    nouns, surface_words = collect_core_nouns(core)
    project_words = {word for word in surface_words if word.isascii() and word.isalpha()} | FUNCTION_WORDS
    english_words = load_english_words()
    common_words = load_common_words(args.common_word_count)
    common_index = build_common_word_index(common_words)
    vocab_candidates = tokenizer_vocab_candidates(tokenizer)
    used: Set[str] = set()

    noun_map = {
        noun: choose_nonce_noun(
            noun,
            tokenizer=tokenizer,
            vocab_candidates=vocab_candidates,
            used=used,
            english_words=english_words,
            project_words=project_words,
            common_index=common_index,
            allow_semantic_fragments=args.allow_semantic_fragments,
            seed=args.seed,
            max_vocab_candidates=args.max_vocab_candidates_per_noun,
            max_generated_candidates=args.max_generated_candidates_per_noun,
        )
        for noun in sorted(nouns)
    }

    fragment = build_fragment_frame(core, noun_map)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fragment.to_csv(args.output_csv, index=False)

    args.lexicon_json.parent.mkdir(parents=True, exist_ok=True)
    lexicon = {
        "metadata": {
            "description": "Tokenizer-aligned noun lexicon for fragment-verb Jabberwocky.",
            "source_core_csv": str(args.core_csv.resolve()),
            "tokenizer_model": args.tokenizer_model,
            "seed": args.seed,
            "notes": [
                "Only nouns are mapped to nonwords.",
                "Verb slots are standalone inflectional fragments: present active=s, past active=ed, passive participle=ed.",
            ],
        },
        "nouns": noun_map,
    }
    args.lexicon_json.write_text(json.dumps(lexicon, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    target_cells = Counter(cell_from_active_passive(row.ta, row.tp) for row in fragment.itertuples(index=False))
    prime_cells = Counter(cell_from_active_passive(row.pa, row.pp) for row in fragment.itertuples(index=False))
    summary = {
        "source_core_csv": str(args.core_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "lexicon_json": str(args.lexicon_json.resolve()),
        "summary_json": str(args.summary_json.resolve()),
        "tokenizer_model": args.tokenizer_model,
        "row_count": int(len(fragment)),
        "target_cell_counts": dict(sorted(target_cells.items())),
        "prime_cell_counts": dict(sorted(prime_cells.items())),
        "row_constraint_audit": row_constraint_audit(fragment),
        "noun_tokenization": noun_tokenization_audit(noun_map, tokenizer, args.tokenizer_model),
        "sentence_tokenization": tokenizer_audit(core, fragment, tokenizer, args.tokenizer_model),
        "vocab_candidate_count": len(vocab_candidates),
        "semantic_filter": {
            "allow_semantic_fragments": bool(args.allow_semantic_fragments),
            "common_word_count": len(common_words),
            "policy": "reject exact common words, prefixes of common words, and candidates containing common words",
        },
        "manipulation": "tokenizer-aligned nonce nouns plus standalone inflectional verb fragments s/ed",
    }
    fail_if_bad(summary)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Saved tokenizer-aligned fragment corpus to {args.output_csv}")
    print(f"Saved noun lexicon to {args.lexicon_json}")
    print(f"Summary: {args.summary_json}")


if __name__ == "__main__":
    main()
