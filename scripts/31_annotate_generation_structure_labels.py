import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd


DEFAULT_INPUTS = [
    Path(
        "behavioral_results/experiment-2/llama323binstruct/experiment-2_generation_audit_lexically_controlled/core/item_generations.csv"
    ),
    Path(
        "behavioral_results/experiment-2/llama323binstruct/experiment-2_generation_audit_lexically_controlled/jabberwocky/item_generations.csv"
    ),
]

WORD_RE = re.compile(r"[a-z']+")

DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "our", "their"}
SUBJECT_PRONOUNS = {"i", "you", "he", "she", "it", "we", "they"}
OBJECT_PRONOUNS = {"me", "you", "him", "her", "it", "us", "them"}
PRONOUNS = SUBJECT_PRONOUNS | OBJECT_PRONOUNS
BE_FORMS = {"am", "is", "are", "was", "were", "be", "been", "being"}
GET_FORMS = {"get", "gets", "got", "getting", "gotten"}
AUX_FORMS = BE_FORMS | GET_FORMS | {"do", "does", "did", "have", "has", "had", "can", "could", "will", "would"}
PREPOSITIONS = {
    "by",
    "to",
    "for",
    "with",
    "near",
    "behind",
    "before",
    "after",
    "during",
    "across",
    "through",
    "in",
    "on",
    "at",
    "from",
    "into",
    "onto",
    "over",
    "under",
    "around",
    "between",
    "without",
    "within",
    "of",
    "as",
}
COORD_CONJ = {"and", "but", "or"}
ADV_OR_NEG = {"not", "never", "just", "already", "still", "really", "very", "quite", "too", "also"}
COPULAR_PARTICIPIAL_ADJ = {"married", "involved", "related", "known", "located"}
IRREGULAR_PARTICIPLES = {
    "beaten",
    "bent",
    "bound",
    "brought",
    "built",
    "bought",
    "caught",
    "chosen",
    "done",
    "driven",
    "eaten",
    "fallen",
    "felt",
    "found",
    "forgiven",
    "forgotten",
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
IRREGULAR_FINITE_VERBS = {
    "beat",
    "beats",
    "beat",
    "did",
    "does",
    "do",
    "drank",
    "fought",
    "found",
    "gave",
    "got",
    "grew",
    "had",
    "hung",
    "hurt",
    "is",
    "knew",
    "left",
    "made",
    "met",
    "paid",
    "put",
    "ran",
    "read",
    "said",
    "saw",
    "shot",
    "slept",
    "spoke",
    "stole",
    "struck",
    "swam",
    "taught",
    "told",
    "understood",
    "was",
    "were",
    "won",
}


def normalize_generated_text(text: str) -> str:
    normalized = str(text).strip().lower().replace(".", " . ")
    return " ".join(normalized.split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate generation outputs with richer sentence-structure labels "
            "(active/passive subtypes, copular, relative, ditransitive, etc.)."
        )
    )
    parser.add_argument("--input-csv", action="append", type=Path, default=None)
    parser.add_argument("--sentence-column", default="greedy_answer_first_sentence_normalized")
    parser.add_argument("--target-active-column", default="target_active")
    parser.add_argument("--target-passive-column", default="target_passive")
    parser.add_argument("--output-column", default="generation_class_detailed")
    parser.add_argument("--voice-column", default="generation_voice_auto")
    parser.add_argument("--reason-column", default="generation_structure_reason")
    parser.add_argument("--arg-structure-column", default="argument_structure_inferred")
    parser.add_argument("--role-frame-column", default="role_frame_inferred")
    parser.add_argument("--role-note-column", default="argument_inference_note")
    parser.add_argument("--strict-column", default="generation_class_strict")
    parser.add_argument("--lax-column", default="generation_class_lax")
    parser.add_argument("--review-strict-column", default="review_remaining_strict")
    parser.add_argument("--review-lax-column", default="review_remaining_lax")
    parser.add_argument(
        "--overwrite-generation-class",
        action="store_true",
        help="If set, replace generation_class with the strict label.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute annotations and print summaries without writing files.",
    )
    return parser.parse_args()


def rough_verb_key(token: str) -> str:
    token = token.lower()
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ied") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        return stem
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        return stem
    if token.endswith("es") and len(token) > 4:
        if token.endswith(("ches", "shes", "sses", "xes", "zes", "oes", "ges")):
            return token[:-2]
        return token[:-1]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _drop_final_period(tokens: List[str]) -> List[str]:
    if tokens and tokens[-1] == ".":
        return tokens[:-1]
    return tokens


def _last_content_token(tokens: Sequence[str]) -> Optional[str]:
    for token in reversed(tokens):
        if token not in DETERMINERS:
            return token
    return tokens[-1] if tokens else None


def _extract_target_signature(target_active: str, target_passive: str) -> Optional[Dict[str, object]]:
    active_tokens = _drop_final_period(normalize_generated_text(target_active).split())
    passive_tokens = _drop_final_period(normalize_generated_text(target_passive).split())
    if "by" not in passive_tokens:
        return None

    by_index = passive_tokens.index("by")
    be_positions = [index for index, token in enumerate(passive_tokens[:by_index]) if token in BE_FORMS]
    if not be_positions:
        return None

    be_index = be_positions[-1]
    patient_phrase = passive_tokens[:be_index]
    agent_phrase = passive_tokens[by_index + 1 :]
    if not patient_phrase or not agent_phrase or be_index + 1 >= by_index:
        return None

    passive_verb = passive_tokens[be_index + 1]
    active_verb = None
    if (
        len(active_tokens) > len(agent_phrase) + len(patient_phrase)
        and active_tokens[: len(agent_phrase)] == agent_phrase
        and active_tokens[-len(patient_phrase) :] == patient_phrase
    ):
        active_verb = active_tokens[len(agent_phrase)]

    expected_verb_keys = {rough_verb_key(passive_verb)}
    if active_verb is not None:
        expected_verb_keys.add(rough_verb_key(active_verb))

    agent_head = _last_content_token(agent_phrase)
    patient_head = _last_content_token(patient_phrase)
    if agent_head is None or patient_head is None:
        return None

    return {
        "agent_head": agent_head,
        "patient_head": patient_head,
        "expected_verb_keys": expected_verb_keys,
    }


def _first_index(tokens: Sequence[str], target: str) -> Optional[int]:
    for index, token in enumerate(tokens):
        if token == target:
            return index
    return None


def _contains_expected_verb(tokens: Sequence[str], expected_verb_keys: Sequence[str]) -> bool:
    return any(rough_verb_key(token) in expected_verb_keys for token in tokens)


def classify_generated_structure(text: str, target_active: str, target_passive: str) -> str:
    normalized = normalize_generated_text(text)
    active_norm = normalize_generated_text(target_active)
    passive_norm = normalize_generated_text(target_passive)
    if normalized == active_norm:
        return "active_exact"
    if normalized == passive_norm:
        return "passive_exact"
    if normalized.startswith(active_norm):
        return "active_prefix"
    if normalized.startswith(passive_norm):
        return "passive_prefix"

    signature = _extract_target_signature(target_active=target_active, target_passive=target_passive)
    if signature is None:
        return "other"

    tokens = _drop_final_period(normalized.split())
    agent_index = _first_index(tokens, str(signature["agent_head"]))
    patient_index = _first_index(tokens, str(signature["patient_head"]))
    expected_verb_keys = signature["expected_verb_keys"]
    if agent_index is None or patient_index is None:
        return "other"

    if agent_index < patient_index:
        span = tokens[agent_index + 1 : patient_index]
        if "by" not in span and _contains_expected_verb(span, expected_verb_keys):
            return "active_structural"

    if patient_index < agent_index:
        middle = tokens[patient_index + 1 : agent_index]
        if "by" in middle:
            by_index = patient_index + 1 + middle.index("by")
            pre_by = tokens[patient_index + 1 : by_index]
            if any(token in BE_FORMS for token in pre_by) and _contains_expected_verb(pre_by, expected_verb_keys):
                return "passive_structural"

    return "other"


def is_participle(token: str) -> bool:
    return token in IRREGULAR_PARTICIPLES or (len(token) > 3 and token.endswith(("ed", "en")))


def is_verbish(token: str) -> bool:
    if token in AUX_FORMS or token in PREPOSITIONS or token in DETERMINERS or token in PRONOUNS:
        return False
    if token in IRREGULAR_FINITE_VERBS:
        return True
    if token.endswith(("ed", "ing", "ies", "ied")) and len(token) > 3:
        return True
    if token.endswith(("es", "s")) and len(token) > 3:
        return True
    return False


def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(str(text).lower())


def has_coordination(words: Sequence[str], raw_text: str) -> bool:
    if any(conj in words for conj in COORD_CONJ) and ("," in raw_text or ";" in raw_text):
        return True
    return False


def is_relative_identity_clause(words: Sequence[str]) -> bool:
    # canonical pattern found often in this dataset:
    # "the one who did it was a/an/the ..."
    if len(words) < 7:
        return False
    if words[:5] == ["the", "one", "who", "did", "it"] and words[5] in BE_FORMS:
        return True
    if len(words) >= 8 and words[0] in DETERMINERS and words[1] in {"man", "woman", "person"}:
        if "who" in words and any(be in words for be in BE_FORMS):
            return True
    return False


def detect_passive(words: Sequence[str]) -> Tuple[str, str]:
    for idx, token in enumerate(words[:-1]):
        if token not in (BE_FORMS | GET_FORMS):
            continue
        cursor = idx + 1
        while cursor < len(words) and words[cursor] in ADV_OR_NEG:
            cursor += 1
        if cursor >= len(words):
            continue

        # Progressive passive: be + being + participle (+ by ...)
        if words[cursor] == "being":
            cursor += 1
            while cursor < len(words) and words[cursor] in ADV_OR_NEG:
                cursor += 1
            if cursor < len(words) and is_participle(words[cursor]):
                if "by" in words[cursor + 1 :]:
                    return "passive_progressive_by", "be_being_participle_by"
                return "passive_progressive_short", "be_being_participle_no_by"
            continue

        if is_participle(words[cursor]):
            participle = words[cursor]
            if participle in COPULAR_PARTICIPIAL_ADJ and (cursor + 1) < len(words):
                next_token = words[cursor + 1]
                if next_token in PREPOSITIONS and next_token != "by":
                    continue
            if "by" in words[cursor + 1 :]:
                return "passive_by", "be_participle_by"
            return "passive_short", "be_participle_no_by"

    return "", ""


def detect_copular(words: Sequence[str]) -> Tuple[str, str]:
    for idx, token in enumerate(words[:-1]):
        if token not in BE_FORMS:
            continue
        cursor = idx + 1
        while cursor < len(words) and words[cursor] in ADV_OR_NEG:
            cursor += 1
        if cursor >= len(words):
            return "copular_incomplete", "be_without_predicate"
        pred = words[cursor]
        if pred in DETERMINERS:
            return "copular_nominal", "be_det_nominal_predicate"
        if pred in PREPOSITIONS:
            return "copular_prepositional", "be_prepositional_predicate"
        if pred.endswith(("ous", "ful", "ive", "able", "ible", "less", "al", "ic", "y")):
            return "copular_adjectival", "be_adjectival_predicate"
        if is_participle(pred) and "by" not in words[cursor + 1 :]:
            return "copular_participial", "be_participial_no_by"
    return "", ""


def detect_active(words: Sequence[str]) -> Tuple[str, str]:
    if len(words) < 3:
        return "", ""

    if words[0] not in DETERMINERS and words[0] not in SUBJECT_PRONOUNS:
        return "", ""

    if words[0] in DETERMINERS and len(words) >= 3 and words[1] == "one" and words[2] == "who":
        return "", ""

    # Typical canonical slot for this dataset.
    verb_idx = 2 if words[0] in DETERMINERS else 1
    if verb_idx >= len(words):
        return "", ""

    verb = words[verb_idx]
    tail = list(words[verb_idx + 1 :])

    # Active progressive: subject + be + V-ing ...
    if verb in BE_FORMS and tail:
        if tail[0].endswith("ing"):
            remainder = tail[1:]
            if remainder and (remainder[0] in DETERMINERS or remainder[0] in PRONOUNS):
                if "to" in remainder or "for" in remainder:
                    return "active_progressive_ditransitive", "be_ing_object_to_for"
                return "active_progressive_transitive", "be_ing_object"
            if remainder and remainder[0] in PREPOSITIONS:
                return "active_progressive_intransitive_pp", "be_ing_prepositional"
            return "active_progressive_intransitive", "be_ing_no_object"
        return "", ""

    if not is_verbish(verb):
        return "", ""

    if not tail:
        return "active_intransitive", "finite_verb_no_complement"

    if tail[0] in PREPOSITIONS:
        return "active_intransitive_prepositional", "finite_verb_prepositional"

    if tail[0] in DETERMINERS or tail[0] in PRONOUNS:
        if "to" in tail[1:] or "for" in tail[1:]:
            return "active_ditransitive_prepositional", "object_plus_to_for_phrase"
        if any(token in PREPOSITIONS for token in tail[1:]):
            return "active_transitive_with_pp", "object_plus_prepositional_modifier"
        return "active_transitive", "finite_verb_direct_object"

    if tail[0] == "to":
        return "active_infinitival_complement", "finite_verb_to_infinitive"

    return "active_other", "finite_verb_noncanonical_complement"


def classify_detailed(sentence: str, target_active: str, target_passive: str) -> Tuple[str, str, str]:
    raw_text = str(sentence or "").strip()
    if not raw_text:
        return "other_empty", "other", "empty_sentence"

    normalized = normalize_generated_text(raw_text)
    words = tokenize_words(normalized)
    if not words:
        return "other_empty", "other", "no_word_tokens"

    target_class = classify_generated_structure(
        text=raw_text,
        target_active=str(target_active),
        target_passive=str(target_passive),
    )

    if target_class.startswith("active"):
        if "to" in words or "for" in words:
            return "active_target_ditransitive_like", "active", f"target_{target_class}_to_for"
        if "being" in words:
            return "active_target_progressive_like", "active", f"target_{target_class}_being"
        return "active_target_transitive_like", "active", f"target_{target_class}"

    if target_class.startswith("passive"):
        if "being" in words and "by" in words:
            return "passive_target_progressive_by", "passive", f"target_{target_class}_being_by"
        if "by" in words:
            return "passive_target_by", "passive", f"target_{target_class}_by"
        return "passive_target_short", "passive", f"target_{target_class}_short"

    if has_coordination(words, raw_text):
        return "other_coordinate_multiclause", "other", "coordination_with_punctuation"

    if is_relative_identity_clause(words):
        return "other_relative_identity_clause", "other", "one_who_did_it_pattern"

    passive_label, passive_reason = detect_passive(words)
    if passive_label:
        return passive_label, "passive", passive_reason

    copular_label, copular_reason = detect_copular(words)
    if copular_label:
        return copular_label, "other", copular_reason

    active_label, active_reason = detect_active(words)
    if active_label:
        return active_label, "active", active_reason

    if "who" in words:
        return "other_relative_clause_misc", "other", "relative_marker_without_confident_parse"
    if any(token in PREPOSITIONS for token in words):
        return "other_prepositional_misc", "other", "prepositional_without_confident_parse"
    return "other_unclassified", "other", "fallback_unclassified"


STRICT_PASSIVE_LABELS = {
    "passive_target_by",
    "passive_by",
    "passive_target_progressive_by",
    "passive_progressive_by",
}
STRICT_ACTIVE_LABELS = {
    "active_target_transitive_like",
    "active_transitive",
}

LAX_PASSIVE_LABELS = STRICT_PASSIVE_LABELS | {
    "passive_target_short",
    "passive_short",
    "passive_progressive_short",
}
LAX_ACTIVE_LABELS = STRICT_ACTIVE_LABELS | {
    "active_target_ditransitive_like",
    "active_target_progressive_like",
    "active_ditransitive_prepositional",
    "active_transitive_with_pp",
    "active_progressive_transitive",
    "active_progressive_ditransitive",
}


def map_binary_class(detailed_label: str, active_labels: Set[str], passive_labels: Set[str]) -> str:
    if detailed_label in active_labels:
        return "active"
    if detailed_label in passive_labels:
        return "passive"
    return "other"


THEME_PRONOUN_HEADS = {"something", "anything", "nothing", "everything", "what"}
THEME_NOUN_HEADS = {"money", "event", "book", "story", "message", "birth", "news", "fact", "idea"}
THEME_BIAS_VERBS = {
    "discover",
    "describe",
    "read",
    "write",
    "say",
    "tell",
    "show",
    "explain",
    "raise",
    "know",
    "learn",
    "find",
}
PARTICLE_TOKENS = {"up", "out", "off", "away", "down", "over", "back"}


def _human_heads_from_targets(frame: pd.DataFrame, target_active_column: str) -> Set[str]:
    heads: Set[str] = set()
    for text in frame[target_active_column].fillna(""):
        tokens = tokenize_words(text)
        if len(tokens) >= 5 and tokens[0] in DETERMINERS and tokens[3] in DETERMINERS:
            heads.add(tokens[1])
            heads.add(tokens[4])
    return heads


def _parse_object_np(tokens: Sequence[str], start: int) -> Tuple[Optional[str], int]:
    if start >= len(tokens):
        return None, start

    tok = tokens[start]
    if tok in THEME_PRONOUN_HEADS:
        return tok, start + 1
    if tok in PRONOUNS:
        return tok, start + 1
    if tok in DETERMINERS:
        if start + 1 < len(tokens):
            return tokens[start + 1], start + 2
        return tok, start + 1
    return None, start


def infer_argument_structure(
    sentence: str,
    human_heads: Set[str],
) -> Tuple[str, str, str, bool, bool]:
    raw_text = str(sentence or "").strip()
    words = tokenize_words(raw_text)
    if not words:
        return "unknown", "unknown", "empty_sentence", False, False

    if detect_passive(words)[0]:
        return "passive_like", "patient>(agent)", "passive_pattern_detected", False, False
    if detect_copular(words)[0]:
        return "copular_like", "theme>predicate", "copular_pattern_detected", False, False
    if has_coordination(words, raw_text):
        return "multiclause", "mixed", "coordination_detected", False, False

    if words[0] not in DETERMINERS and words[0] not in SUBJECT_PRONOUNS:
        return "unknown", "unknown", "noncanonical_subject_start", False, False
    if words[0] in DETERMINERS and len(words) >= 3 and words[1] == "one" and words[2] == "who":
        return "relative_identity", "referent>predicate", "one_who_pattern", False, False

    verb_idx = 2 if words[0] in DETERMINERS else 1
    if verb_idx >= len(words):
        return "unknown", "unknown", "no_verb_slot", False, False

    verb = words[verb_idx]
    tail = list(words[verb_idx + 1 :])

    if verb in BE_FORMS and tail and tail[0].endswith("ing"):
        verb = tail[0]
        tail = tail[1:]

    if not is_verbish(verb):
        return "unknown", "unknown", "verb_not_confident", False, False

    cursor = 0
    while cursor < len(tail) and tail[cursor] in ADV_OR_NEG:
        cursor += 1

    obj_head, next_idx = _parse_object_np(tail, cursor)

    # Handle phrasal-verb particles: e.g., "picked up the book".
    particle_used = False
    if obj_head is None and cursor < len(tail) and tail[cursor] in PARTICLE_TOKENS:
        particle_used = True
        obj_head, next_idx = _parse_object_np(tail, cursor + 1)

    if obj_head is None:
        return "active_nonobject", "single_core_argument(+oblique)", "no_direct_object_detected", False, False

    remainder = tail[next_idx:]
    has_to_for_recipient = False
    for idx, tok in enumerate(remainder[:-1]):
        if tok in {"to", "for"} and (remainder[idx + 1] in DETERMINERS or remainder[idx + 1] in PRONOUNS):
            has_to_for_recipient = True
            break

    verb_key = rough_verb_key(verb)
    obj_key = rough_verb_key(obj_head)
    if has_to_for_recipient:
        return (
            "active_ditransitive_prepositional",
            "agent>theme>recipient",
            "direct_object_plus_to_for_recipient",
            False,
            True,
        )

    if obj_key in THEME_PRONOUN_HEADS or obj_key in THEME_NOUN_HEADS or verb_key in THEME_BIAS_VERBS:
        note = "object_inferred_as_theme"
        if particle_used:
            note = "phrasal_verb_with_theme_object"
        return "active_monotransitive_theme", "agent>theme", note, False, True

    if obj_key in human_heads or obj_key in PRONOUNS:
        note = "object_inferred_as_patient_human"
    else:
        note = "object_inferred_as_patient_default"
    if particle_used:
        note = "phrasal_verb_with_patient_object"
    return "active_monotransitive_patient", "agent>patient", note, True, True


def summarize(frame: pd.DataFrame, path: Path, output_column: str, voice_column: str) -> None:
    print(f"\n=== {path} ({len(frame)} rows)")
    print("\nDetailed label counts:")
    print(frame[output_column].value_counts(dropna=False).to_string())
    print("\nVoice counts:")
    print(frame[voice_column].value_counts(dropna=False).to_string())


def annotate_file(
    path: Path,
    sentence_column: str,
    target_active_column: str,
    target_passive_column: str,
    output_column: str,
    voice_column: str,
    reason_column: str,
    arg_structure_column: str,
    role_frame_column: str,
    role_note_column: str,
    strict_column: str,
    lax_column: str,
    review_strict_column: str,
    review_lax_column: str,
    overwrite_generation_class: bool,
    dry_run: bool,
) -> None:
    frame = pd.read_csv(path)
    for legacy_col in ("argument_structure_auto", "role_frame_auto", "role_frame_note"):
        if legacy_col in frame.columns:
            frame = frame.drop(columns=[legacy_col])

    required = {sentence_column, target_active_column, target_passive_column}
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    triples = [
        classify_detailed(sentence, target_active, target_passive)
        for sentence, target_active, target_passive in zip(
            frame[sentence_column],
            frame[target_active_column],
            frame[target_passive_column],
        )
    ]
    frame[output_column] = [triple[0] for triple in triples]
    frame[voice_column] = [triple[1] for triple in triples]
    frame[reason_column] = [triple[2] for triple in triples]
    human_heads = _human_heads_from_targets(frame=frame, target_active_column=target_active_column)
    arg_inference = frame[sentence_column].map(lambda sent: infer_argument_structure(sent, human_heads))
    frame[arg_structure_column] = arg_inference.map(lambda x: x[0])
    frame[role_frame_column] = arg_inference.map(lambda x: x[1])
    frame[role_note_column] = arg_inference.map(lambda x: x[2])
    inferred_strict_active = arg_inference.map(lambda x: x[3])
    inferred_lax_active = arg_inference.map(lambda x: x[4])

    frame[strict_column] = frame[output_column].map(
        lambda label: map_binary_class(
            detailed_label=str(label),
            active_labels=STRICT_ACTIVE_LABELS,
            passive_labels=STRICT_PASSIVE_LABELS,
        )
    )
    frame[lax_column] = frame[output_column].map(
        lambda label: map_binary_class(
            detailed_label=str(label),
            active_labels=LAX_ACTIVE_LABELS,
            passive_labels=LAX_PASSIVE_LABELS,
        )
    )

    nonpassive_mask = ~frame[output_column].astype(str).str.startswith("passive")
    frame.loc[(frame[strict_column] == "other") & nonpassive_mask & inferred_strict_active, strict_column] = "active"
    frame.loc[(frame[lax_column] == "other") & nonpassive_mask & inferred_lax_active, lax_column] = "active"

    frame[review_strict_column] = frame[strict_column].eq("other")
    frame[review_lax_column] = frame[lax_column].eq("other")

    if overwrite_generation_class:
        frame["generation_class"] = frame[strict_column]

    summarize(frame, path=path, output_column=output_column, voice_column=voice_column)
    print("\nArgument-structure counts:")
    print(frame[arg_structure_column].value_counts(dropna=False).to_string())
    print("\nStrict binary counts:")
    print(frame[strict_column].value_counts(dropna=False).to_string())
    print("\nLax binary counts:")
    print(frame[lax_column].value_counts(dropna=False).to_string())
    print(f"\nStrict review-needed rows: {int(frame[review_strict_column].sum())}")
    print(f"Lax review-needed rows: {int(frame[review_lax_column].sum())}")
    if not dry_run:
        frame.to_csv(path, index=False)
        print(f"\nWrote: {path}")


def main() -> None:
    args = parse_args()
    inputs = args.input_csv if args.input_csv else DEFAULT_INPUTS
    for raw_path in inputs:
        path = raw_path.resolve()
        annotate_file(
            path=path,
            sentence_column=args.sentence_column,
            target_active_column=args.target_active_column,
            target_passive_column=args.target_passive_column,
            output_column=args.output_column,
            voice_column=args.voice_column,
            reason_column=args.reason_column,
            arg_structure_column=args.arg_structure_column,
            role_frame_column=args.role_frame_column,
            role_note_column=args.role_note_column,
            strict_column=args.strict_column,
            lax_column=args.lax_column,
            review_strict_column=args.review_strict_column,
            review_lax_column=args.review_lax_column,
            overwrite_generation_class=args.overwrite_generation_class,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
