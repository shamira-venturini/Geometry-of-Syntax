from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Mapping, Tuple

from .data import ExperimentItem
from .prompts import render_prime_block


DEFAULT_NOMINALIZATION_OVERRIDES = {
    "be": "being",
    "do": "doing",
    "go": "going",
    "have": "having",
    "die": "dying",
    "lie": "lying",
    "tie": "tying",
    "see": "seeing",
}


@dataclass(frozen=True)
class Exp4Prompt:
    prompt_text: str
    target_sentence: str
    question_text: str
    target_voice: str
    verb_lemma: str
    verb_nominalized: str
    nominalization_fallback_used: bool


def _sentence_tokens(text: str) -> list[str]:
    cleaned = re.sub(r"[^A-Za-z'\- ]+", " ", str(text))
    return [token.strip().lower() for token in cleaned.split() if token.strip()]


def _extract_active_verb_form(sentence: str) -> str:
    tokens = _sentence_tokens(sentence)
    if len(tokens) < 3:
        raise ValueError(f"Could not extract active verb from sentence: {sentence!r}")
    return tokens[2]


def _lemmatize_third_person(verb_form: str) -> str:
    token = str(verb_form).strip().lower()
    if not token:
        return "do"

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "sses", "xes", "zes", "oes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 3:
        return token[:-1]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def _is_cvc(token: str) -> bool:
    if len(token) < 3:
        return False
    vowels = set("aeiou")
    return (
        token[-1] not in vowels
        and token[-2] in vowels
        and token[-3] not in vowels
        and token[-1] not in {"w", "x", "y"}
    )


def _default_nominalization(lemma: str) -> str:
    token = str(lemma).strip().lower()
    if not token:
        return "doing"
    if token.endswith("ie") and len(token) > 2:
        return token[:-2] + "ying"
    if token.endswith("e") and not token.endswith(("ee", "oe", "ye")):
        return token[:-1] + "ing"
    if _is_cvc(token):
        return token + token[-1] + "ing"
    return token + "ing"


def build_nominalized_question(
    *,
    item: ExperimentItem,
    overrides: Mapping[str, str] | None,
    question_template: str,
    logger: logging.Logger,
) -> Tuple[str, str, str, bool]:
    fallback_used = False

    try:
        verb_form = _extract_active_verb_form(item.target_sentence_active)
        verb_lemma = _lemmatize_third_person(verb_form)
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning(
            "Exp4 nominalization fallback for item_id=%s (reason=%s): using default lemma 'do'.",
            item.item_id,
            exc,
        )
        verb_lemma = "do"
        fallback_used = True

    normalized_overrides = {
        str(key).strip().lower(): str(value).strip().lower()
        for key, value in (overrides or {}).items()
        if str(key).strip() and str(value).strip()
    }

    if verb_lemma in normalized_overrides:
        nominalized = normalized_overrides[verb_lemma]
    elif verb_lemma in DEFAULT_NOMINALIZATION_OVERRIDES:
        nominalized = DEFAULT_NOMINALIZATION_OVERRIDES[verb_lemma]
    else:
        nominalized = _default_nominalization(verb_lemma)

    if not nominalized:
        logger.warning(
            "Exp4 nominalization empty after processing for item_id=%s verb_lemma=%s; using 'doing'.",
            item.item_id,
            verb_lemma,
        )
        nominalized = "doing"
        fallback_used = True

    try:
        question_text = question_template.format(nominalized_verb=nominalized)
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning(
            "Exp4 question template fallback for item_id=%s (template=%r, reason=%s).",
            item.item_id,
            question_template,
            exc,
        )
        question_text = f"Who did the {nominalized}?"
        fallback_used = True

    return question_text, verb_lemma, nominalized, fallback_used


def build_exp4_prompt(
    *,
    item: ExperimentItem,
    target_voice: str,
    question_template: str,
    nominalization_overrides: Mapping[str, str] | None,
    answer_prefix: str,
    logger: logging.Logger,
) -> Exp4Prompt:
    if target_voice not in {"active", "passive"}:
        raise ValueError(f"Unsupported target_voice={target_voice!r}; expected 'active' or 'passive'.")

    target_sentence = (
        item.target_sentence_active
        if target_voice == "active"
        else item.target_sentence_passive
    )

    question_text, verb_lemma, nominalized, fallback_used = build_nominalized_question(
        item=item,
        overrides=nominalization_overrides,
        question_template=question_template,
        logger=logger,
    )

    prime_block = render_prime_block(item=item, eos_token_text="")
    normalized_answer_prefix = str(answer_prefix).strip() or "The"

    sections: list[str] = []
    if prime_block:
        sections.append(prime_block)
    sections.append(target_sentence.strip())
    sections.append(f"Question: {question_text}")
    sections.append(f"Answer: {normalized_answer_prefix}")

    prompt_text = "\n\n".join(section for section in sections if section).rstrip()

    return Exp4Prompt(
        prompt_text=prompt_text,
        target_sentence=target_sentence,
        question_text=question_text,
        target_voice=target_voice,
        verb_lemma=verb_lemma,
        verb_nominalized=nominalized,
        nominalization_fallback_used=fallback_used,
    )
