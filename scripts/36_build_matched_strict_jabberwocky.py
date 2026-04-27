import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORE = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_strict_4cell_counterbalanced.csv"
DEFAULT_VERBS = REPO_ROOT / "PrimeLM" / "vocabulary_lists" / "verblist_T_usf_freq.csv"
DEFAULT_OUTPUT = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_matched_strict_4cell.csv"
DEFAULT_LEXICON = (
    REPO_ROOT / "corpora" / "transitive" / "vocabulary_lists" / "jabberwocky_matched_strict_4cell_lexicon.json"
)
DEFAULT_SUMMARY = (
    REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_matched_strict_4cell_summary.json"
)

FUNCTION_WORDS = {"a", "an", "the", "is", "was", "by", "."}
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
VERB_SAFE_CODAS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "t", "v", "ck", "ld", "lt", "mp", "nd", "nt", "rk"]
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


try:
    from wordfreq import zipf_frequency
except Exception:  # pragma: no cover - optional dependency in local environments.
    zipf_frequency = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Jabberwocky corpus by replacing each strict CORE content "
            "word with a stable, phonotactically plausible nonword while "
            "preserving the exact pa/pp/ta/tp templates."
        )
    )
    parser.add_argument("--core-csv", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--verb-list", type=Path, default=DEFAULT_VERBS)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lexicon-json", type=Path, default=DEFAULT_LEXICON)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--max-candidates-per-word", type=int, default=700)
    parser.add_argument("--tokenizer-model", default="gpt2-large")
    parser.add_argument(
        "--disable-tokenizer-control",
        action="store_true",
        help="Use only phonotactic candidate scoring; do not optimize nonwords for BPE token length.",
    )
    parser.add_argument(
        "--tokenizer-audit",
        action="store_true",
        help="Also compare token lengths for the selected tokenizer if a local tokenizer cache is available.",
    )
    return parser.parse_args()


def vowel_initial(word: str) -> bool:
    return word[:1].lower() in VOWELS


def cv_template(word: str) -> str:
    return "".join("V" if char in VOWELS else "C" for char in word.lower() if char.isalpha())


def syllable_count(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    return max(1, len(groups))


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


def load_english_words() -> Set[str]:
    words: Set[str] = set()
    for path in (Path("/usr/share/dict/words"),):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                token = line.strip().lower()
                if token.isalpha():
                    words.add(token)
    return words


KNOWN_WORD_CACHE: Dict[str, bool] = {}


def is_known_word(candidate: str, english_words: Set[str], project_words: Set[str]) -> bool:
    if candidate in KNOWN_WORD_CACHE:
        return KNOWN_WORD_CACHE[candidate]
    if candidate in english_words or candidate in project_words:
        KNOWN_WORD_CACHE[candidate] = True
        return KNOWN_WORD_CACHE[candidate]
    if zipf_frequency is None:
        KNOWN_WORD_CACHE[candidate] = False
        return KNOWN_WORD_CACHE[candidate]
    KNOWN_WORD_CACHE[candidate] = zipf_frequency(candidate, "en") >= 1.0
    return KNOWN_WORD_CACHE[candidate]


def candidate_is_too_close(candidate: str, source: str, project_words: Set[str]) -> bool:
    threshold = 1 if len(source) <= 5 else 2
    if levenshtein(candidate, source, max_distance=threshold) <= threshold:
        return True
    for word in project_words:
        if word[:1] != candidate[:1]:
            continue
        if abs(len(candidate) - len(word)) <= threshold and levenshtein(candidate, word, threshold) <= threshold:
            return True
    return False


def make_syllable(
    rng: random.Random,
    *,
    initial_vowel: Optional[bool],
    final_for_verb: bool,
    final_syllable: bool,
) -> str:
    if initial_vowel is True:
        onset = ""
    elif initial_vowel is False:
        onset = rng.choice([item for item in ONSETS if item])
    else:
        onset = rng.choice(ONSETS)

    nucleus = rng.choice(NUCLEI)
    if not final_syllable:
        coda_pool = NONFINAL_CODAS
    elif final_for_verb:
        coda_pool = VERB_SAFE_CODAS
    else:
        coda_pool = CODAS
    coda = rng.choice(coda_pool)
    return onset + nucleus + coda


def generate_raw_candidate(
    rng: random.Random,
    source: str,
    *,
    kind: str,
    initial_vowel_required: bool,
) -> str:
    n_syllables = syllable_count(source)
    pieces = []
    for index in range(n_syllables):
        pieces.append(
            make_syllable(
                rng,
                initial_vowel=initial_vowel_required if index == 0 else None,
                final_for_verb=kind == "verb" and index == n_syllables - 1,
                final_syllable=index == n_syllables - 1,
            )
        )
    return "".join(pieces)


def candidate_score(candidate: str, source: str) -> Tuple[int, int, int, int, str]:
    bad_bigram_count = sum(candidate.count(bigram) for bigram in BAD_INTERNAL_BIGRAMS)
    return (
        bad_bigram_count,
        abs(len(candidate) - len(source)),
        levenshtein(cv_template(candidate), cv_template(source)),
        abs(syllable_count(candidate) - syllable_count(source)),
        candidate,
    )


def load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name, local_files_only=True), None
    except Exception as exc:
        return None, str(exc)


def token_count(tokenizer, word_or_sentence: str) -> int:
    return len(tokenizer.encode(" " + word_or_sentence, add_special_tokens=False))


def tokenizer_vocab_candidates(tokenizer) -> List[str]:
    candidates: Set[str] = set()
    for token, index in tokenizer.get_vocab().items():
        text = tokenizer.decode([index])
        if not text.startswith(" "):
            continue
        candidate = text.strip().lower()
        if 3 <= len(candidate) <= 10 and candidate.isascii() and candidate.isalpha():
            candidates.add(candidate)
    return sorted(candidates)


def valid_nonce_candidate(
    candidate: str,
    *,
    source: str,
    kind: str,
    used: Set[str],
    english_words: Set[str],
    project_words: Set[str],
) -> bool:
    if not candidate.isalpha():
        return False
    if not candidate.isascii():
        return False
    if candidate in used:
        return False
    if vowel_initial(candidate) != vowel_initial(source):
        return False
    if kind == "verb" and candidate[-1] in "aeiousxyz":
        return False
    if kind == "verb" and (candidate + "s" in used or candidate + "ed" in used):
        return False
    if is_known_word(candidate, english_words, project_words):
        return False
    if kind == "verb" and (
        is_known_word(candidate + "s", english_words, project_words)
        or is_known_word(candidate + "ed", english_words, project_words)
    ):
        return False
    if candidate_is_too_close(candidate, source, project_words):
        return False
    return True


def tokenizer_score(
    candidate: str,
    *,
    source: str,
    kind: str,
    tokenizer,
    verb_source_forms: Optional[Tuple[str, str, str]],
) -> Tuple[int, int, int, int]:
    if tokenizer is None:
        return (0, 0, 0, 0)
    if kind == "noun":
        source_count = token_count(tokenizer, source)
        candidate_count = token_count(tokenizer, candidate)
        diff = abs(candidate_count - source_count)
        return (diff, max(0, candidate_count - source_count), candidate_count, 0)

    if verb_source_forms is None:
        raise ValueError("verb_source_forms is required for verb token scoring.")
    source_present, source_past_active, source_past_passive = verb_source_forms
    candidate_present = candidate + "s"
    candidate_past = candidate + "ed"
    diffs = [
        abs(token_count(tokenizer, candidate_present) - token_count(tokenizer, source_present)),
        abs(token_count(tokenizer, candidate_past) - token_count(tokenizer, source_past_active)),
        abs(token_count(tokenizer, candidate_past) - token_count(tokenizer, source_past_passive)),
    ]
    signed_overage = [
        token_count(tokenizer, candidate_present) - token_count(tokenizer, source_present),
        token_count(tokenizer, candidate_past) - token_count(tokenizer, source_past_active),
        token_count(tokenizer, candidate_past) - token_count(tokenizer, source_past_passive),
    ]
    return (sum(diffs), max([0] + signed_overage), max(diffs), token_count(tokenizer, candidate))


def make_nonce_word(
    source: str,
    *,
    kind: str,
    used: Set[str],
    english_words: Set[str],
    project_words: Set[str],
    seed: int,
    max_candidates: int,
    tokenizer=None,
    vocab_candidates: Optional[Sequence[str]] = None,
    verb_source_forms: Optional[Tuple[str, str, str]] = None,
) -> str:
    rng = random.Random(f"{seed}:{kind}:{source}")
    best: Optional[Tuple[Tuple[int, int, int, int, int, int, int, int, str], str]] = None
    raw_candidates: List[str] = []
    if vocab_candidates:
        matching_initial = [
            candidate
            for candidate in vocab_candidates
            if vowel_initial(candidate) == vowel_initial(source)
        ]
        rng.shuffle(matching_initial)
        raw_candidates.extend(matching_initial[: max_candidates * 2])

    for _ in range(max_candidates):
        raw_candidates.append(
            generate_raw_candidate(
                rng,
                source,
                kind=kind,
                initial_vowel_required=vowel_initial(source),
            )
        )

    for candidate in raw_candidates:
        if not valid_nonce_candidate(
            candidate,
            source=source,
            kind=kind,
            used=used,
            english_words=english_words,
            project_words=project_words,
        ):
            continue

        score = (
            tokenizer_score(
                candidate,
                source=source,
                kind=kind,
                tokenizer=tokenizer,
                verb_source_forms=verb_source_forms,
            )
            + candidate_score(candidate, source)
        )
        if best is None or score < best[0]:
            best = (score, candidate)

    if best is None:
        raise RuntimeError(f"Could not generate a vetted {kind} nonword for {source!r}.")

    nonce = best[1]
    used.add(nonce)
    if kind == "verb":
        used.add(nonce + "s")
        used.add(nonce + "ed")
    return nonce


def read_core(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip().str.lower()
    expected = ["pa", "pp", "ta", "tp"]
    if list(frame.columns) != expected:
        missing = set(expected).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
        frame = frame[expected]
    return frame


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


def load_verb_forms(
    path: Path,
) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, str], Set[str], Dict[str, Tuple[str, str, str]]]:
    verbs = pd.read_csv(path, sep=";")
    active_forms: Dict[str, Tuple[str, str]] = {}
    passive_forms: Dict[str, str] = {}
    source_forms: Dict[str, Tuple[str, str, str]] = {}
    project_verb_words: Set[str] = set()
    for row in verbs.itertuples(index=False):
        lemma = str(row.V).strip().lower()
        past_a = str(row.past_A).strip().lower()
        past_p = str(row.past_P).strip().lower()
        pres_3s = str(row.pres_3s).strip().lower()
        active_forms[pres_3s] = (lemma, "present")
        active_forms[past_a] = (lemma, "past")
        passive_forms[past_p] = lemma
        source_forms[lemma] = (pres_3s, past_a, past_p)
        project_verb_words.update([lemma, past_a, past_p, pres_3s])
    return active_forms, passive_forms, project_verb_words, source_forms


def collect_core_words(
    frame: pd.DataFrame,
    active_forms: Mapping[str, Tuple[str, str]],
    passive_forms: Mapping[str, str],
) -> Tuple[Set[str], Set[str], Set[str]]:
    nouns: Set[str] = set()
    verb_lemmas: Set[str] = set()
    surface_words: Set[str] = set()

    for row in frame.itertuples(index=False):
        for sentence in (row.pa, row.ta):
            det_a, agent, verb_form, det_p, patient = active_parts(str(sentence))
            nouns.update([agent.lower(), patient.lower()])
            surface_words.update([det_a.lower(), agent.lower(), verb_form.lower(), det_p.lower(), patient.lower()])
            if verb_form.lower() not in active_forms:
                raise ValueError(f"Unknown active verb form: {verb_form}")
            verb_lemmas.add(active_forms[verb_form.lower()][0])

        for sentence in (row.pp, row.tp):
            det_p, patient, aux, participle, det_a, agent = passive_parts(str(sentence))
            nouns.update([agent.lower(), patient.lower()])
            surface_words.update([det_p.lower(), patient.lower(), aux.lower(), participle.lower(), det_a.lower(), agent.lower()])
            if participle.lower() not in passive_forms:
                raise ValueError(f"Unknown passive verb form: {participle}")
            verb_lemmas.add(passive_forms[participle.lower()])

    return nouns, verb_lemmas, surface_words


def nonce_active_form(stem: str, tense: str) -> str:
    if tense == "present":
        return stem + "s"
    if tense == "past":
        return stem + "ed"
    raise ValueError(f"Unsupported tense: {tense}")


def transform_active(
    sentence: str,
    noun_map: Mapping[str, str],
    verb_map: Mapping[str, str],
    active_forms: Mapping[str, Tuple[str, str]],
) -> str:
    det_a, agent, verb_form, det_p, patient = active_parts(sentence)
    lemma, tense = active_forms[verb_form.lower()]
    return " ".join(
        [
            det_a,
            noun_map[agent.lower()],
            nonce_active_form(verb_map[lemma], tense),
            det_p,
            noun_map[patient.lower()],
            ".",
        ]
    )


def transform_passive(
    sentence: str,
    noun_map: Mapping[str, str],
    verb_map: Mapping[str, str],
    passive_forms: Mapping[str, str],
) -> str:
    det_p, patient, aux, participle, det_a, agent = passive_parts(sentence)
    lemma = passive_forms[participle.lower()]
    return " ".join(
        [
            det_p,
            noun_map[patient.lower()],
            aux,
            verb_map[lemma] + "ed",
            "by",
            det_a,
            noun_map[agent.lower()],
            ".",
        ]
    )


def transform_frame(
    frame: pd.DataFrame,
    noun_map: Mapping[str, str],
    verb_map: Mapping[str, str],
    active_forms: Mapping[str, Tuple[str, str]],
    passive_forms: Mapping[str, str],
) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "pa": transform_active(str(row.pa), noun_map, verb_map, active_forms),
                "pp": transform_passive(str(row.pp), noun_map, verb_map, passive_forms),
                "ta": transform_active(str(row.ta), noun_map, verb_map, active_forms),
                "tp": transform_passive(str(row.tp), noun_map, verb_map, passive_forms),
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
    same_verb_rows = 0
    same_det_family_rows = 0
    bad_article_rows = 0

    for row in frame.itertuples(index=False):
        pa_det_a, pa_agent, pa_verb, pa_det_p, pa_patient = active_parts(str(row.pa))
        ta_det_a, ta_agent, ta_verb, ta_det_p, ta_patient = active_parts(str(row.ta))
        _, _, pp_aux, _, _, _ = passive_parts(str(row.pp))
        _, _, tp_aux, _, _, _ = passive_parts(str(row.tp))

        same_aux_rows += int(pp_aux == tp_aux)
        shared_noun_rows += int(bool({pa_agent, pa_patient} & {ta_agent, ta_patient}))
        same_verb_rows += int(nonce_active_stem(pa_verb) == nonce_active_stem(ta_verb))
        same_det_family_rows += int(det_family(pa_det_a) == det_family(ta_det_a))
        for det, noun in ((pa_det_a, pa_agent), (pa_det_p, pa_patient), (ta_det_a, ta_agent), (ta_det_p, ta_patient)):
            bad_article_rows += int((det == "an" and not vowel_initial(noun)) or (det == "a" and vowel_initial(noun)))

    return {
        "same_aux_rows": same_aux_rows,
        "shared_noun_rows": shared_noun_rows,
        "same_active_verb_stem_rows": same_verb_rows,
        "same_det_family_rows": same_det_family_rows,
        "bad_indefinite_article_uses": bad_article_rows,
    }


def nonce_active_stem(active_verb: str) -> str:
    if active_verb.endswith("ed"):
        return active_verb[:-2]
    if active_verb.endswith("s"):
        return active_verb[:-1]
    return active_verb


def role_balance(frame: pd.DataFrame) -> Dict[str, object]:
    counts: Dict[str, Counter] = {
        "target_active_only": Counter(),
        "target_active_passive": Counter(),
        "full_pa_pp_ta_tp": Counter(),
    }
    for row in frame.itertuples(index=False):
        _, p_agent, _, _, p_patient = active_parts(str(row.pa))
        _, t_agent, _, _, t_patient = active_parts(str(row.ta))
        counts["target_active_only"][(t_agent, "agent")] += 1
        counts["target_active_only"][(t_patient, "patient")] += 1
        for scope in ("target_active_passive", "full_pa_pp_ta_tp"):
            counts[scope][(t_agent, "agent")] += 2
            counts[scope][(t_patient, "patient")] += 2
        counts["full_pa_pp_ta_tp"][(p_agent, "agent")] += 2
        counts["full_pa_pp_ta_tp"][(p_patient, "patient")] += 2

    summaries = {}
    for scope, counter in counts.items():
        words = sorted({word for word, _ in counter})
        imbalanced = []
        max_abs = 0
        for word in words:
            diff = counter[(word, "agent")] - counter[(word, "patient")]
            max_abs = max(max_abs, abs(diff))
            if diff:
                imbalanced.append({"word": word, "agent_minus_patient": int(diff)})
        summaries[scope] = {
            "word_count": len(words),
            "imbalanced_word_count": len(imbalanced),
            "max_abs_agent_minus_patient": int(max_abs),
            "imbalanced_preview": imbalanced[:25],
        }
    return summaries


def safe_stats(values: Sequence[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def lexical_summary(source_to_nonce: Mapping[str, str]) -> Dict[str, object]:
    length_diffs = [len(nonce) - len(source) for source, nonce in source_to_nonce.items()]
    cv_distances = [levenshtein(cv_template(source), cv_template(nonce)) for source, nonce in source_to_nonce.items()]
    return {
        "count": len(source_to_nonce),
        "length_difference": safe_stats(length_diffs),
        "cv_template_distance": safe_stats(cv_distances),
        "initial_vowel_class_mismatches": sum(
            int(vowel_initial(source) != vowel_initial(nonce)) for source, nonce in source_to_nonce.items()
        ),
        "preview": [
            {"source": source, "nonce": nonce}
            for source, nonce in sorted(source_to_nonce.items())[:20]
        ],
    }


def maybe_tokenizer_audit(core: pd.DataFrame, jabber: pd.DataFrame, tokenizer, model_name: str) -> Dict[str, object]:
    if tokenizer is None:
        return {"status": "skipped", "reason": f"Could not load local {model_name} tokenizer."}
    result = {}
    for column in ["pa", "pp", "ta", "tp"]:
        core_counts = [token_count(tokenizer, text) for text in core[column].astype(str)]
        jabber_counts = [token_count(tokenizer, text) for text in jabber[column].astype(str)]
        diffs = [jabber_value - core_value for core_value, jabber_value in zip(core_counts, jabber_counts)]
        result[column] = {
            "core": safe_stats(core_counts),
            "jabberwocky": safe_stats(jabber_counts),
            "jabber_minus_core": safe_stats(diffs),
            "exact_match_rows": int(sum(diff == 0 for diff in diffs)),
            "nonzero_diff_rows": int(sum(diff != 0 for diff in diffs)),
        }
    return {"status": "ok", "model": model_name, "columns": result}


def tokenization_lexicon_summary(
    *,
    noun_map: Mapping[str, str],
    verb_map: Mapping[str, str],
    verb_source_forms: Mapping[str, Tuple[str, str, str]],
    tokenizer,
    model_name: str,
) -> Dict[str, object]:
    if tokenizer is None:
        return {"status": "skipped", "reason": f"Could not load local {model_name} tokenizer."}

    noun_rows = []
    for source, nonce in sorted(noun_map.items()):
        noun_rows.append(
            {
                "source": source,
                "nonce": nonce,
                "source_tokens": token_count(tokenizer, source),
                "nonce_tokens": token_count(tokenizer, nonce),
                "difference": token_count(tokenizer, nonce) - token_count(tokenizer, source),
            }
        )

    verb_rows = []
    for lemma, nonce in sorted(verb_map.items()):
        source_present, source_past_active, source_past_passive = verb_source_forms[lemma]
        nonce_present = nonce + "s"
        nonce_past = nonce + "ed"
        verb_rows.append(
            {
                "source": lemma,
                "nonce": nonce,
                "source_present": source_present,
                "nonce_present": nonce_present,
                "source_past_active": source_past_active,
                "source_past_passive": source_past_passive,
                "nonce_past": nonce_past,
                "present_difference": token_count(tokenizer, nonce_present)
                - token_count(tokenizer, source_present),
                "past_active_difference": token_count(tokenizer, nonce_past)
                - token_count(tokenizer, source_past_active),
                "past_passive_difference": token_count(tokenizer, nonce_past)
                - token_count(tokenizer, source_past_passive),
            }
        )

    noun_diffs = [row["difference"] for row in noun_rows]
    verb_diffs = [
        value
        for row in verb_rows
        for value in (
            row["present_difference"],
            row["past_active_difference"],
            row["past_passive_difference"],
        )
    ]
    return {
        "status": "ok",
        "model": model_name,
        "noun_differences": safe_stats(noun_diffs),
        "verb_form_differences": safe_stats(verb_diffs),
        "noun_exact_matches": int(sum(diff == 0 for diff in noun_diffs)),
        "verb_form_exact_matches": int(sum(diff == 0 for diff in verb_diffs)),
        "noun_nonzero_preview": [row for row in noun_rows if row["difference"] != 0][:25],
        "verb_nonzero_preview": [
            row
            for row in verb_rows
            if row["present_difference"] != 0
            or row["past_active_difference"] != 0
            or row["past_passive_difference"] != 0
        ][:25],
    }


def build_project_words(surface_words: Iterable[str], verb_words: Iterable[str]) -> Set[str]:
    project_words = {word.lower() for word in surface_words if str(word).isalpha()}
    project_words.update(word.lower() for word in verb_words if str(word).isalpha())
    project_words.update(FUNCTION_WORDS)
    return project_words


def fail_if_audit_bad(summary: Mapping[str, object]) -> None:
    row_audit = summary["row_constraint_audit"]
    for key in [
        "same_aux_rows",
        "shared_noun_rows",
        "same_active_verb_stem_rows",
        "same_det_family_rows",
        "bad_indefinite_article_uses",
    ]:
        if int(row_audit[key]) != 0:
            raise RuntimeError(f"Matched Jabberwocky audit failed: {key}={row_audit[key]}")

    for kind in ["noun_lexicon", "verb_lexicon"]:
        if int(summary[kind]["initial_vowel_class_mismatches"]) != 0:
            raise RuntimeError(f"Matched Jabberwocky audit failed: {kind} has initial-vowel mismatches.")

    for scope, payload in summary["role_balance"].items():
        if int(payload["imbalanced_word_count"]) != 0:
            raise RuntimeError(f"Matched Jabberwocky audit failed: role imbalance in {scope}.")


def main() -> None:
    args = parse_args()
    core = read_core(args.core_csv)
    active_forms, passive_forms, project_verb_words, verb_source_forms = load_verb_forms(args.verb_list)
    nouns, verb_lemmas, surface_words = collect_core_words(core, active_forms, passive_forms)

    english_words = load_english_words()
    project_words = build_project_words(surface_words, project_verb_words)
    used: Set[str] = set()
    tokenizer = None
    tokenizer_load_error = None
    vocab_candidates: List[str] = []
    if not args.disable_tokenizer_control:
        tokenizer, tokenizer_load_error = load_tokenizer(args.tokenizer_model)
        if tokenizer is not None:
            vocab_candidates = tokenizer_vocab_candidates(tokenizer)

    noun_map = {
        noun: make_nonce_word(
            noun,
            kind="noun",
            used=used,
            english_words=english_words,
            project_words=project_words,
            seed=args.seed,
            max_candidates=args.max_candidates_per_word,
            tokenizer=tokenizer,
            vocab_candidates=vocab_candidates,
        )
        for noun in sorted(nouns)
    }
    verb_map = {
        lemma: make_nonce_word(
            lemma,
            kind="verb",
            used=used,
            english_words=english_words,
            project_words=project_words,
            seed=args.seed,
            max_candidates=args.max_candidates_per_word,
            tokenizer=tokenizer,
            vocab_candidates=vocab_candidates,
            verb_source_forms=verb_source_forms[lemma],
        )
        for lemma in sorted(verb_lemmas)
    }

    jabber = transform_frame(core, noun_map, verb_map, active_forms, passive_forms)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    jabber.to_csv(args.output_csv, index=False)

    lexicon = {
        "metadata": {
            "description": (
                "One-to-one strict CORE-to-Jabberwocky lexicon. Nonwords are generated "
                "from English-like syllable templates and preserve source initial vowel "
                "class so a/an remains grammatical."
            ),
            "source_core_csv": str(args.core_csv.resolve()),
            "seed": args.seed,
            "tokenizer_control": {
                "enabled": not args.disable_tokenizer_control,
                "model": args.tokenizer_model,
                "status": "ok" if tokenizer is not None else "skipped",
                "load_error": tokenizer_load_error,
            },
            "notes": [
                "Retires the older independently generated Jabberwocky materials for strict-design analyses.",
                "Nouns map one-to-one from CORE nouns; verbs map one-to-one from CORE verb lemmas.",
                "Nonce verbs use regular English-looking morphology: stem+s and stem+ed.",
                "When a local tokenizer is available, candidate selection minimizes BPE token-length mismatch.",
            ],
        },
        "nouns": noun_map,
        "verb_stems": verb_map,
    }
    args.lexicon_json.parent.mkdir(parents=True, exist_ok=True)
    args.lexicon_json.write_text(json.dumps(lexicon, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    target_cells = Counter(cell_from_active_passive(row.ta, row.tp) for row in jabber.itertuples(index=False))
    prime_cells = Counter(cell_from_active_passive(row.pa, row.pp) for row in jabber.itertuples(index=False))
    summary = {
        "source_core_csv": str(args.core_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "lexicon_json": str(args.lexicon_json.resolve()),
        "summary_json": str(args.summary_json.resolve()),
        "seed": args.seed,
        "row_count": int(len(jabber)),
        "column_parity_with_core": list(jabber.columns) == list(core.columns),
        "target_cell_counts": dict(sorted(target_cells.items())),
        "prime_cell_counts": dict(sorted(prime_cells.items())),
        "row_constraint_audit": row_constraint_audit(jabber),
        "role_balance": role_balance(jabber),
        "noun_lexicon": lexical_summary(noun_map),
        "verb_lexicon": lexical_summary(verb_map),
        "tokenizer_control": {
            "enabled": not args.disable_tokenizer_control,
            "model": args.tokenizer_model,
            "status": "ok" if tokenizer is not None else "skipped",
            "load_error": tokenizer_load_error,
            "vocab_candidate_count": len(vocab_candidates),
            "lexicon_tokenization": tokenization_lexicon_summary(
                noun_map=noun_map,
                verb_map=verb_map,
                verb_source_forms=verb_source_forms,
                tokenizer=tokenizer,
                model_name=args.tokenizer_model,
            ),
        },
        "tokenizer_audit": (
            maybe_tokenizer_audit(core, jabber, tokenizer, args.tokenizer_model)
            if args.tokenizer_audit or tokenizer is not None
            else {"status": "skipped", "reason": "Use --tokenizer-audit to run optional local tokenizer comparison."}
        ),
        "retirement_note": (
            "Older independently generated Jabberwocky files are retained for provenance only; "
            "strict analyses should use this matched-from-core corpus."
        ),
    }
    fail_if_audit_bad(summary)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Saved matched Jabberwocky corpus to {args.output_csv}")
    print(f"Saved matched lexicon to {args.lexicon_json}")
    print(f"Saved summary to {args.summary_json}")


if __name__ == "__main__":
    main()
