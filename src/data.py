from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_VOCAB_DIR = REPO_ROOT / "corpora" / "transitive" / "vocabulary_lists"
LEGACY_VOCAB_DIR = REPO_ROOT / "PrimeLM" / "vocabulary_lists"
DEFAULT_VERB_LIST_PATH = BUNDLED_VOCAB_DIR / "verblist_T_usf_freq.csv"
DEFAULT_ASSOCIATION_CSV_PATH = (
    REPO_ROOT / "corpora" / "transitive" / "usf_association_edges_core_vocab.csv"
)


REQUIRED_COLUMNS: Sequence[str] = (
    "item_id",
    "model_condition",
    "prime_condition",
    "prime_text",
    "target_event_prompt",
    "active_completion",
    "passive_completion",
    "target_sentence_active",
    "target_sentence_passive",
    "question_template",
    "correct_answer_for_active",
    "incorrect_answer_for_active",
    "correct_answer_for_passive",
    "incorrect_answer_for_passive",
    "lexicality_condition",
    "notes",
)

CORPUS_REQUIRED_COLUMNS: Sequence[str] = ("pa", "pp", "ta", "tp")

VALID_PRIME_CONDITIONS = {
    "active",
    "passive",
    "filler",
    "no_prime",
}

PRIME_CONDITION_ALIASES = {
    "active_prime": "active",
    "passive_prime": "passive",
    "filler_prime": "filler",
    "no_prime_eos": "no_prime",
    "no_prime_empty": "no_prime",
    "no_demo": "no_prime",
    "none": "no_prime",
}

VALID_LEXICALITY_CONDITIONS = {
    "real",
    "nonce",
}


CORE_FILLER_SENTENCES: Sequence[str] = (
    "The lantern glowed near sunset .",
    "The traveler rested beside the river .",
    "The notebook stayed on the shelf .",
    "The window rattled during the storm .",
    "The singer smiled after rehearsal .",
    "The painter waited near the doorway .",
    "The engine cooled before dawn .",
    "The market opened after sunrise .",
    "The blanket dried on the line .",
    "The planet shimmered above the valley .",
    "The baker laughed during breakfast .",
    "The signal flickered across the screen .",
    "The tourist wandered through the plaza .",
    "The kettle whistled in the kitchen .",
    "The garden brightened after rain .",
    "The jacket hung behind the chair .",
    "The radio crackled in the attic .",
    "The witness paused near the doorway .",
    "The package arrived before noon .",
    "The sculpture stood in the hallway .",
    "The captain waited beside the harbor .",
    "The teacher listened during assembly .",
    "The violin echoed in the theater .",
    "The pillow rested on the sofa .",
)

NONCE_FILLER_SENTENCES: Sequence[str] = (
    "The bri ed near dil .",
    "The dru ed beside the gri .",
    "The mur ed on the dir .",
    "The dur ed during the fab .",
    "The fav ed after fel .",
    "The fil ed near the fol .",
    "The ful ed before gir .",
    "The lav ed after lil .",
    "The mol ed on the mul .",
    "The nav ed above the nin .",
    "The nom ed during nud .",
    "The pav ed across the pel .",
    "The pil ed through the por .",
    "The rav ed in the rom .",
    "The sav ed after som .",
    "The sor ed behind the sul .",
    "The tel ed in the tir .",
    "The tul ed near the vel .",
    "The vil ed before vir .",
    "The vom ed in the vul .",
    "The bri ed beside the dru .",
    "The gri ed during mur .",
    "The dir ed in the dur .",
    "The fab ed on the fav .",
)


@dataclass(frozen=True)
class ExperimentItem:
    item_id: str
    model_condition: str
    prime_condition: str
    prime_text: str
    target_event_prompt: str
    active_completion: str
    passive_completion: str
    target_sentence_active: str
    target_sentence_passive: str
    question_template: str
    correct_answer_for_active: str
    incorrect_answer_for_active: str
    correct_answer_for_passive: str
    incorrect_answer_for_passive: str
    lexicality_condition: str
    notes: str


@dataclass(frozen=True)
class DatasetBundle:
    path: Path
    frame: pd.DataFrame
    items: List[ExperimentItem]


class DatasetValidationError(ValueError):
    """Raised when an input dataset does not satisfy expected constraints."""


def _read_jsonl(path: Path) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"Invalid JSON on line {line_number} in {path}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise DatasetValidationError(
                    f"Expected a JSON object on line {line_number} in {path}."
                )
            records.append(payload)
    return pd.DataFrame.from_records(records)


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return _read_jsonl(path)
    raise DatasetValidationError(
        f"Unsupported dataset extension for {path}. Use .csv or .jsonl"
    )


def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DatasetValidationError(f"Missing required columns: {missing}")


def _coerce_string_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = result[column].fillna("").astype(str)
    return result


def _validate_values(frame: pd.DataFrame) -> None:
    unknown_prime_conditions = sorted(
        set(frame["prime_condition"]).difference(VALID_PRIME_CONDITIONS)
    )
    if unknown_prime_conditions:
        raise DatasetValidationError(
            "Unknown prime_condition values: "
            f"{unknown_prime_conditions}. Expected one of {sorted(VALID_PRIME_CONDITIONS)}"
        )

    unknown_lexicality = sorted(
        set(frame["lexicality_condition"]).difference(VALID_LEXICALITY_CONDITIONS)
    )
    if unknown_lexicality:
        raise DatasetValidationError(
            "Unknown lexicality_condition values: "
            f"{unknown_lexicality}. Expected one of {sorted(VALID_LEXICALITY_CONDITIONS)}"
        )

    if frame["item_id"].str.strip().eq("").any():
        raise DatasetValidationError("item_id must be non-empty for all rows.")


def _canonical_prime_condition(value: object) -> str:
    normalized = str(value).strip().lower()
    return PRIME_CONDITION_ALIASES.get(normalized, normalized)


def _normalize_prime_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["prime_condition"] = result["prime_condition"].map(_canonical_prime_condition)
    return result


def _to_items(frame: pd.DataFrame) -> List[ExperimentItem]:
    items: List[ExperimentItem] = []
    for record in frame.to_dict(orient="records"):
        payload = {column: str(record.get(column, "")) for column in REQUIRED_COLUMNS}
        items.append(ExperimentItem(**payload))
    return items


def load_dataset(path: str | Path) -> DatasetBundle:
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise DatasetValidationError(f"Dataset file not found: {dataset_path}")

    frame = _load_frame(dataset_path)
    _require_columns(frame, REQUIRED_COLUMNS)
    frame = _coerce_string_columns(frame, REQUIRED_COLUMNS)
    frame = _normalize_prime_conditions(frame)
    _validate_values(frame)

    return DatasetBundle(
        path=dataset_path,
        frame=frame,
        items=_to_items(frame),
    )


def _pretty_sentence(text: str) -> str:
    compact = " ".join(str(text).strip().split())
    compact = compact.replace(" .", ".")
    compact = compact.replace(" ,", ",")
    if not compact:
        return compact
    return compact[0].upper() + compact[1:]


def _to_ing(verb: str) -> str:
    token = verb.lower().strip()
    if not token:
        return "doing"
    stripped_past_suffix = False
    # Convert simple finite forms to a rough lemma first.
    if token.endswith("ied") and len(token) > 4:
        token = token[:-3] + "y"
        stripped_past_suffix = True
    elif token.endswith("ed") and len(token) > 4:
        token = token[:-2]
        stripped_past_suffix = True
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith(("ches", "shes", "sses", "xes", "zes", "oes")) and len(token) > 4:
        token = token[:-2]
    elif token.endswith("es") and len(token) > 3:
        token = token[:-1]
    elif token.endswith("s") and len(token) > 3:
        token = token[:-1]

    if token.endswith("ie") and len(token) > 2:
        return token[:-2] + "ying"
    if token.endswith("e") and not token.endswith(("ee", "oe", "ye")):
        return token[:-1] + "ing"
    if token.endswith("y") and len(token) > 1:
        return token + "ing"
    if (
        not stripped_past_suffix
        and
        len(token) >= 3
        and token[-1] not in "aeiouy"
        and token[-2] in "aeiou"
        and token[-3] not in "aeiou"
    ):
        return token + token[-1] + "ing"
    return token + "ing"


def _strip_period_tokens(sentence: str) -> List[str]:
    tokens = [token for token in str(sentence).strip().split() if token != "."]
    if tokens and tokens[-1].endswith("."):
        tokens[-1] = tokens[-1][:-1]
    return [token for token in tokens if token]


def _active_components(sentence: str) -> Tuple[str, str, str, str, str]:
    tokens = _strip_period_tokens(sentence)
    if len(tokens) < 5:
        raise DatasetValidationError(
            f"Could not parse active sentence (expected at least 5 tokens): '{sentence}'"
        )

    # Current corpora format: DET AGENT VERB DET PATIENT
    agent_phrase = " ".join(tokens[0:2])
    verb = tokens[2]
    patient_phrase = " ".join(tokens[3:5])
    agent_head = tokens[1]
    patient_head = tokens[4]
    return agent_phrase, agent_head, verb, patient_phrase, patient_head


def _target_event_prompt(agent_phrase: str, patient_phrase: str, verb: str) -> str:
    if verb.lower().strip() in {"s", "ed"}:
        return (
            f"There was an event involving {agent_phrase} and {patient_phrase}.\n"
            f"The one who did it was {agent_phrase}.\n"
            f"The one it happened to was {patient_phrase}.\n\n"
            'Bridget asked, "What happened?"\n'
            'Mary answered, "'
        )
    event_name = _to_ing(verb)
    article = "an" if event_name[:1].lower() in "aeiou" else "a"
    return (
        f"There was {article} {event_name} event involving {agent_phrase} and {patient_phrase}.\n"
        f"The one who did it was {agent_phrase}.\n"
        f"The one it happened to was {patient_phrase}.\n\n"
        'Bridget asked, "What happened?"\n'
        'Mary answered, "'
    )


def _sentence_tokens_lower(text: str) -> List[str]:
    cleaned = str(text).replace(".", " ").replace(",", " ")
    return [token.strip().lower() for token in cleaned.split() if token.strip()]


def _choose_filler_from_pool(
    *,
    filler_sentences: Sequence[str],
    item_index: int,
    filler_seed: int,
    blocked_nouns: Sequence[str],
) -> str:
    if not filler_sentences:
        raise DatasetValidationError("Filler pool is empty while filler_prime is requested.")

    blocked = {noun.strip().lower() for noun in blocked_nouns if noun.strip()}
    candidates: List[str] = []
    for sentence in filler_sentences:
        sentence_tokens = set(_sentence_tokens_lower(sentence))
        if sentence_tokens.intersection(blocked):
            continue
        candidates.append(sentence)
    if not candidates:
        candidates = list(filler_sentences)

    rng = random.Random(int(filler_seed) + int(item_index) * 7919)
    index = rng.randrange(len(candidates))
    return _pretty_sentence(candidates[index])


def _load_corpus(path: Path, max_items: Optional[int]) -> pd.DataFrame:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise DatasetValidationError(f"Corpus not found: {resolved}")
    frame = pd.read_csv(resolved)
    _require_columns(frame, CORPUS_REQUIRED_COLUMNS)
    if max_items is not None:
        frame = frame.head(int(max_items)).copy()
    return frame.reset_index(drop=True)


def _passive_aux(sentence: str) -> str:
    tokens = str(sentence).strip().split()
    if len(tokens) != 8:
        raise DatasetValidationError(
            f"Could not parse passive sentence for auxiliary audit: '{sentence}'"
        )
    return tokens[2].lower()


def _parse_active_sentence(sentence: str) -> Tuple[str, str, str, str, str]:
    tokens = str(sentence).strip().split()
    if len(tokens) != 6:
        raise DatasetValidationError(
            f"Could not parse active sentence for strict validation: '{sentence}'"
        )
    return tokens[0].lower(), tokens[1].lower(), tokens[2].lower(), tokens[3].lower(), tokens[4].lower()


def _det_family(det: str) -> str:
    token = str(det).strip().lower()
    if token == "the":
        return "def"
    if token in {"a", "an"}:
        return "indef"
    return "unknown"


def _load_verb_maps(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    requested = path.expanduser().resolve()
    fallback_candidates = [
        requested,
        (BUNDLED_VOCAB_DIR / "verblist_T_usf_freq.csv").resolve(),
        (LEGACY_VOCAB_DIR / "verblist_T_usf_freq.csv").resolve(),
    ]
    seen: Set[Path] = set()
    candidate_paths: List[Path] = []
    for candidate in fallback_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        candidate_paths.append(candidate)

    resolved: Optional[Path] = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    if resolved is None:
        checked = ", ".join(str(candidate) for candidate in candidate_paths)
        raise DatasetValidationError(
            "Verb list for strict validation not found. "
            f"Checked: {checked}"
        )

    frame = pd.read_csv(resolved, sep=";")
    required = {"v", "pres_3s", "past_a", "past_p"}
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DatasetValidationError(
            f"Verb list is missing required columns for strict validation: {missing}"
        )

    form_to_lemma: Dict[str, str] = {}
    form_to_tense: Dict[str, str] = {}
    for row in frame.itertuples(index=False):
        lemma = str(row.v).strip().lower()
        pres_3s = str(row.pres_3s).strip().lower()
        past_a = str(row.past_a).strip().lower()
        past_p = str(row.past_p).strip().lower()
        form_to_lemma[pres_3s] = lemma
        form_to_lemma[past_a] = lemma
        form_to_lemma[past_p] = lemma
        form_to_tense[pres_3s] = "present"
        form_to_tense[past_a] = "past"
    return form_to_lemma, form_to_tense


def _load_semantic_edges(path: Path) -> Dict[str, Set[str]]:
    requested = path.expanduser().resolve()
    fallback_candidates = [
        requested,
        DEFAULT_ASSOCIATION_CSV_PATH.resolve(),
    ]
    seen: Set[Path] = set()
    candidate_paths: List[Path] = []
    for candidate in fallback_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        candidate_paths.append(candidate)

    resolved: Optional[Path] = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    if resolved is None:
        checked = ", ".join(str(candidate) for candidate in candidate_paths)
        raise DatasetValidationError(
            "USF association CSV for strict validation not found. "
            f"Checked: {checked}. "
            "Run scripts/33_build_usf_association_edges.py first."
        )

    frame = pd.read_csv(resolved)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = {"cue", "target"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DatasetValidationError(
            f"Association CSV is missing required columns: {missing}"
        )

    edges: Dict[str, Set[str]] = {}
    for row in frame.itertuples(index=False):
        cue = str(row.cue).strip().lower()
        target = str(row.target).strip().lower()
        if not cue or not target:
            continue
        edges.setdefault(cue, set()).add(target)
        edges.setdefault(target, set()).add(cue)
    return edges


def _active_tense(verb_form: str, form_to_tense: Mapping[str, str]) -> str:
    token = str(verb_form).strip().lower()
    mapped = form_to_tense.get(token)
    if mapped:
        return mapped
    if token.endswith("ed"):
        return "past"
    if token.endswith("s"):
        return "present"
    return "unknown"


def _active_lemma(verb_form: str, form_to_lemma: Mapping[str, str]) -> str:
    token = str(verb_form).strip().lower()
    return str(form_to_lemma.get(token, token))


def _validate_corpus_strict_controls(
    *,
    frame: pd.DataFrame,
    corpus_name: str,
    lexicality_condition: str,
    form_to_lemma: Mapping[str, str],
    form_to_tense: Mapping[str, str],
    semantic_edges: Mapping[str, Set[str]],
    enforce_semantic_nonassociation: bool,
) -> None:
    violations: Dict[str, int] = {
        "shared_aux_rows": 0,
        "shared_noun_rows": 0,
        "shared_verb_rows": 0,
        "same_det_family_rows": 0,
        "det_side_mismatch_rows": 0,
        "prime_tense_mismatch_rows": 0,
        "target_tense_mismatch_rows": 0,
        "semantic_association_rows": 0,
        "unknown_tokenization_rows": 0,
    }
    preview: List[Dict[str, object]] = []

    for item_index, row in frame.iterrows():
        try:
            pa_det_a, pa_agent, pa_verb, pa_det_p, pa_patient = _parse_active_sentence(str(row["pa"]))
            ta_det_a, ta_agent, ta_verb, ta_det_p, ta_patient = _parse_active_sentence(str(row["ta"]))
            pp_aux = _passive_aux(str(row["pp"]))
            tp_aux = _passive_aux(str(row["tp"]))
        except DatasetValidationError:
            violations["unknown_tokenization_rows"] += 1
            if len(preview) < 5:
                preview.append(
                    {
                        "item_index": int(item_index),
                        "error": "tokenization",
                        "pa": str(row["pa"]),
                        "pp": str(row["pp"]),
                        "ta": str(row["ta"]),
                        "tp": str(row["tp"]),
                    }
                )
            continue

        prime_det_family = _det_family(pa_det_a)
        target_det_family = _det_family(ta_det_a)
        if prime_det_family == "unknown" or target_det_family == "unknown":
            violations["det_side_mismatch_rows"] += 1
        if _det_family(pa_det_p) != prime_det_family or _det_family(ta_det_p) != target_det_family:
            violations["det_side_mismatch_rows"] += 1
        if prime_det_family == target_det_family:
            violations["same_det_family_rows"] += 1

        if pp_aux == tp_aux:
            violations["shared_aux_rows"] += 1

        prime_nouns = {pa_agent, pa_patient}
        target_nouns = {ta_agent, ta_patient}
        if prime_nouns.intersection(target_nouns):
            violations["shared_noun_rows"] += 1

        prime_lemma = _active_lemma(pa_verb, form_to_lemma)
        target_lemma = _active_lemma(ta_verb, form_to_lemma)
        if prime_lemma == target_lemma:
            violations["shared_verb_rows"] += 1

        prime_expected_tense = "present" if pp_aux == "is" else "past" if pp_aux == "was" else "unknown"
        target_expected_tense = "present" if tp_aux == "is" else "past" if tp_aux == "was" else "unknown"
        prime_actual_tense = _active_tense(pa_verb, form_to_tense)
        target_actual_tense = _active_tense(ta_verb, form_to_tense)
        if prime_expected_tense != prime_actual_tense:
            violations["prime_tense_mismatch_rows"] += 1
        if target_expected_tense != target_actual_tense:
            violations["target_tense_mismatch_rows"] += 1

        if enforce_semantic_nonassociation and lexicality_condition == "real":
            prime_content = [pa_agent, pa_patient, prime_lemma]
            target_content = [ta_agent, ta_patient, target_lemma]
            associated = False
            for target_word in target_content:
                for prime_word in prime_content:
                    if str(prime_word) in semantic_edges.get(str(target_word), set()):
                        associated = True
                        break
                if associated:
                    break
            if associated:
                violations["semantic_association_rows"] += 1

        if len(preview) < 5:
            # Keep preview only for rows with at least one strict violation.
            if any(
                (
                    pp_aux == tp_aux,
                    bool(prime_nouns.intersection(target_nouns)),
                    prime_lemma == target_lemma,
                    prime_det_family == target_det_family,
                    prime_expected_tense != prime_actual_tense,
                    target_expected_tense != target_actual_tense,
                )
            ):
                preview.append(
                    {
                        "item_index": int(item_index),
                        "prime_lemma": prime_lemma,
                        "target_lemma": target_lemma,
                        "prime_nouns": sorted(prime_nouns),
                        "target_nouns": sorted(target_nouns),
                        "prime_aux": pp_aux,
                        "target_aux": tp_aux,
                        "prime_det_family": prime_det_family,
                        "target_det_family": target_det_family,
                        "prime_expected_tense": prime_expected_tense,
                        "prime_actual_tense": prime_actual_tense,
                        "target_expected_tense": target_expected_tense,
                        "target_actual_tense": target_actual_tense,
                    }
                )

    violated = {name: count for name, count in violations.items() if count > 0}
    if violated:
        raise DatasetValidationError(
            f"Corpus '{corpus_name}' failed strict Sinclair-style validation: {violated}. "
            f"Preview: {preview}"
        )


def _build_rows_from_corpus(
    corpus_name: str,
    lexicality_condition: str,
    frame: pd.DataFrame,
    prime_conditions: Sequence[str],
    question_template: str,
    filler_mode: str,
    filler_offset: int,
    filler_seed: int,
    filler_sentences: Sequence[str],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    n_items = len(frame)
    if n_items == 0:
        return rows

    if lexicality_condition not in VALID_LEXICALITY_CONDITIONS:
        raise DatasetValidationError(
            f"Invalid lexicality_condition '{lexicality_condition}' for corpus '{corpus_name}'."
        )

    for item_index, row in frame.iterrows():
        base_item_id = f"{corpus_name}_{item_index:05d}"

        active_prime_sentence = _pretty_sentence(str(row["pa"]))
        passive_prime_sentence = _pretty_sentence(str(row["pp"]))
        target_active = _pretty_sentence(str(row["ta"]))
        target_passive = _pretty_sentence(str(row["tp"]))

        agent_phrase, agent_head, verb, patient_phrase, patient_head = _active_components(str(row["ta"]))
        event_prompt = _target_event_prompt(agent_phrase=agent_phrase, patient_phrase=patient_phrase, verb=verb)
        if filler_mode == "offset_target":
            filler_index = (item_index + max(1, int(filler_offset))) % n_items
            filler_prime_sentence = _pretty_sentence(str(frame.loc[filler_index, "ta"]))
            filler_source = f"offset_target_{filler_offset}"
        elif filler_mode == "pool":
            filler_prime_sentence = _choose_filler_from_pool(
                filler_sentences=filler_sentences,
                item_index=item_index,
                filler_seed=filler_seed,
                blocked_nouns=(agent_head, patient_head),
            )
            filler_source = "intransitive_pool"
        else:
            raise DatasetValidationError(
                f"Unknown filler_mode '{filler_mode}'. Use 'pool' or 'offset_target'."
            )

        for prime_condition in prime_conditions:
            if prime_condition == "active":
                prime_text = active_prime_sentence
            elif prime_condition == "passive":
                prime_text = passive_prime_sentence
            elif prime_condition == "filler":
                prime_text = filler_prime_sentence
            elif prime_condition == "no_prime":
                prime_text = ""
            else:
                raise DatasetValidationError(f"Unsupported prime condition: {prime_condition}")

            rows.append(
                {
                    "item_id": base_item_id,
                    "model_condition": corpus_name,
                    "prime_condition": prime_condition,
                    "prime_text": prime_text,
                    "target_event_prompt": event_prompt,
                    "active_completion": target_active,
                    "passive_completion": target_passive,
                    "target_sentence_active": target_active,
                    "target_sentence_passive": target_passive,
                    "question_template": question_template,
                    "correct_answer_for_active": f"The {agent_head}.",
                    "incorrect_answer_for_active": f"The {patient_head}.",
                    "correct_answer_for_passive": f"The {agent_head}.",
                    "incorrect_answer_for_passive": f"The {patient_head}.",
                    "lexicality_condition": lexicality_condition,
                    "notes": (
                        f"generated_from_corpus={corpus_name};verb={verb};"
                        f"filler_source={filler_source}"
                    ),
                }
            )

    return rows


def load_dataset_from_experiment_config(experiment_cfg: Mapping[str, object]) -> DatasetBundle:
    dataset_path = experiment_cfg.get("dataset_path")
    if isinstance(dataset_path, str) and dataset_path.strip():
        return load_dataset(dataset_path)

    dataset_mode = str(experiment_cfg.get("dataset_mode", "corpora")).strip().lower()
    if dataset_mode != "corpora":
        raise DatasetValidationError(
            "dataset_mode must be 'corpora' when dataset_path is not supplied."
        )

    corpora = experiment_cfg.get("corpora")
    if not isinstance(corpora, list) or not corpora:
        raise DatasetValidationError(
            "experiment.corpora must be a non-empty list when using dataset_mode=corpora."
        )

    prime_conditions = experiment_cfg.get("prime_conditions", ["active", "passive", "filler", "no_prime"])
    if not isinstance(prime_conditions, list) or not prime_conditions:
        raise DatasetValidationError("experiment.prime_conditions must be a non-empty list.")

    normalized_prime_conditions = [_canonical_prime_condition(condition) for condition in prime_conditions]
    unknown = sorted(set(normalized_prime_conditions).difference(VALID_PRIME_CONDITIONS))
    if unknown:
        raise DatasetValidationError(
            f"Unknown experiment.prime_conditions values: {unknown}."
        )

    max_items_default_raw = experiment_cfg.get("max_items_per_corpus")
    max_items_default: Optional[int] = int(max_items_default_raw) if max_items_default_raw is not None else None
    filler_mode = str(experiment_cfg.get("filler_mode", "pool")).strip().lower()
    filler_offset = int(experiment_cfg.get("filler_offset", 137))
    filler_seed = int(experiment_cfg.get("filler_seed", int(experiment_cfg.get("seed", 13))))
    enforce_aux_mismatch_default = bool(experiment_cfg.get("enforce_aux_mismatch", True))
    enforce_strict_controls_default = bool(experiment_cfg.get("enforce_strict_controls", True))
    enforce_semantic_default = bool(experiment_cfg.get("enforce_semantic_nonassociation", True))
    verb_list_path = Path(str(experiment_cfg.get("verb_list_path", DEFAULT_VERB_LIST_PATH)))
    association_csv_path = Path(
        str(experiment_cfg.get("association_csv_path", DEFAULT_ASSOCIATION_CSV_PATH))
    )
    form_to_lemma, form_to_tense = _load_verb_maps(verb_list_path)
    semantic_edges: Optional[Dict[str, Set[str]]] = None
    core_filler_sentences_cfg = experiment_cfg.get("core_filler_sentences")
    nonce_filler_sentences_cfg = experiment_cfg.get("nonce_filler_sentences")
    core_filler_sentences = (
        [str(value) for value in core_filler_sentences_cfg]
        if isinstance(core_filler_sentences_cfg, list) and core_filler_sentences_cfg
        else list(CORE_FILLER_SENTENCES)
    )
    nonce_filler_sentences = (
        [str(value) for value in nonce_filler_sentences_cfg]
        if isinstance(nonce_filler_sentences_cfg, list) and nonce_filler_sentences_cfg
        else list(NONCE_FILLER_SENTENCES)
    )
    question_template = str(experiment_cfg.get("question_template", "Who was the agent?"))

    all_rows: List[Dict[str, str]] = []
    for entry in corpora:
        if not isinstance(entry, Mapping):
            raise DatasetValidationError("Each entry in experiment.corpora must be a mapping.")

        name = str(entry.get("name", "")).strip()
        path_raw = str(entry.get("path", "")).strip()
        if not name:
            raise DatasetValidationError("Each experiment.corpora entry requires 'name'.")
        if not path_raw:
            raise DatasetValidationError(f"Corpus '{name}' is missing 'path'.")

        lexicality = str(entry.get("lexicality_condition", "real")).strip().lower()
        enforce_aux_mismatch = bool(entry.get("enforce_aux_mismatch", enforce_aux_mismatch_default))
        enforce_strict_controls = bool(entry.get("enforce_strict_controls", enforce_strict_controls_default))
        enforce_semantic_nonassociation = bool(
            entry.get("enforce_semantic_nonassociation", enforce_semantic_default)
        )
        max_items_raw = entry.get("max_items", max_items_default)
        max_items = int(max_items_raw) if max_items_raw is not None else None

        corpus_path = Path(path_raw)
        frame = _load_corpus(path=corpus_path, max_items=max_items)
        if enforce_strict_controls:
            if enforce_semantic_nonassociation:
                if semantic_edges is None:
                    semantic_edges = _load_semantic_edges(association_csv_path)
                semantic_edges_for_validation: Mapping[str, Set[str]] = semantic_edges
            else:
                semantic_edges_for_validation = {}
            _validate_corpus_strict_controls(
                frame=frame,
                corpus_name=name,
                lexicality_condition=lexicality,
                form_to_lemma=form_to_lemma,
                form_to_tense=form_to_tense,
                semantic_edges=semantic_edges_for_validation,
                enforce_semantic_nonassociation=enforce_semantic_nonassociation,
            )
        elif enforce_aux_mismatch:
            # Backward-compatible minimal check for legacy configurations.
            _validate_corpus_strict_controls(
                frame=frame,
                corpus_name=name,
                lexicality_condition=lexicality,
                form_to_lemma=form_to_lemma,
                form_to_tense=form_to_tense,
                semantic_edges={},
                enforce_semantic_nonassociation=False,
            )
        fillers_for_corpus = nonce_filler_sentences if lexicality == "nonce" else core_filler_sentences

        rows = _build_rows_from_corpus(
            corpus_name=name,
            lexicality_condition=lexicality,
            frame=frame,
            prime_conditions=normalized_prime_conditions,
            question_template=question_template,
            filler_mode=filler_mode,
            filler_offset=filler_offset,
            filler_seed=filler_seed,
            filler_sentences=fillers_for_corpus,
        )
        all_rows.extend(rows)

    if not all_rows:
        raise DatasetValidationError("No rows generated from corpora. Check corpus paths and max_items settings.")

    built_frame = pd.DataFrame(all_rows)
    _require_columns(built_frame, REQUIRED_COLUMNS)
    built_frame = _coerce_string_columns(built_frame, REQUIRED_COLUMNS)
    built_frame = _normalize_prime_conditions(built_frame)
    _validate_values(built_frame)

    pseudo_path = Path("generated_from_corpora")
    return DatasetBundle(
        path=pseudo_path,
        frame=built_frame,
        items=_to_items(built_frame),
    )
