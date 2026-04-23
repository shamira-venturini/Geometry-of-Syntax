from __future__ import annotations

import re
from dataclasses import dataclass


_DETERMINERS = {"the", "a", "an"}


@dataclass(frozen=True)
class AnswerEvaluation:
    generated_answer_normalized: str
    matched_label: str
    is_correct: bool


def _word_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", str(text).lower()) if token]


def _head_noun(answer_text: str) -> str:
    tokens = [token for token in _word_tokens(answer_text) if token not in _DETERMINERS]
    if not tokens:
        return ""
    return tokens[-1]


def _first_clause(text: str) -> str:
    compact = re.sub(r"\s+", " ", str(text).strip())
    if not compact:
        return ""
    compact = compact.replace("\u2019", "'")
    compact = compact.split("\n", 1)[0].strip()
    for separator in [".", "?", "!", ";"]:
        if separator in compact:
            compact = compact.split(separator, 1)[0].strip()
            break
    return compact


def normalize_generated_answer(
    *,
    generated_answer_raw: str,
    correct_answer: str,
    foil_answer: str,
) -> str:
    text = _first_clause(generated_answer_raw)
    text = re.sub(r"^answer\s*:\s*", "", text, flags=re.IGNORECASE).strip()

    correct_head = _head_noun(correct_answer)
    foil_head = _head_noun(foil_answer)
    tokens = _word_tokens(text)

    if correct_head and correct_head in tokens:
        return f"the {correct_head}"
    if foil_head and foil_head in tokens:
        return f"the {foil_head}"

    was_match = re.search(r"\bwas\s+(?:the|a|an)?\s*([A-Za-z]+)\b", text, flags=re.IGNORECASE)
    if was_match:
        return f"the {was_match.group(1).lower()}"

    stripped = [token for token in tokens if token not in _DETERMINERS]
    if not stripped:
        return ""
    return f"the {stripped[0]}"


def evaluate_generated_answer(
    *,
    generated_answer_raw: str,
    correct_answer: str,
    foil_answer: str,
) -> AnswerEvaluation:
    normalized_answer = normalize_generated_answer(
        generated_answer_raw=generated_answer_raw,
        correct_answer=correct_answer,
        foil_answer=foil_answer,
    )

    correct_head = _head_noun(correct_answer)
    foil_head = _head_noun(foil_answer)

    token_universe = set(_word_tokens(generated_answer_raw)) | set(_word_tokens(normalized_answer))
    has_correct = bool(correct_head) and correct_head in token_universe
    has_foil = bool(foil_head) and foil_head in token_universe

    if has_correct and not has_foil:
        matched_label = "correct"
    elif has_foil and not has_correct:
        matched_label = "foil"
    elif has_correct and has_foil:
        matched_label = "ambiguous"
    else:
        matched_label = "other"

    return AnswerEvaluation(
        generated_answer_normalized=normalized_answer,
        matched_label=matched_label,
        is_correct=(matched_label == "correct"),
    )
