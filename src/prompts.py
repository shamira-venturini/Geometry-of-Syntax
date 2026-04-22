from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List

from .data import ExperimentItem


def _normalize_spacing(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def render_prime_block(
    item: ExperimentItem,
    eos_token_text: str = "",
) -> str:
    """Render the optional prime block for an item.

    no_prime: omitted
    legacy no-prime aliases are also omitted
    otherwise: uses the provided prime sentence.
    """
    condition = item.prime_condition
    if condition in {"no_prime", "no_prime_eos", "no_prime_empty", "no_demo"}:
        return ""
    return _normalize_spacing(item.prime_text)


def render_experiment_1_prompt(item: ExperimentItem, eos_token_text: str = "") -> str:
    prime_block = render_prime_block(item=item, eos_token_text=eos_token_text)
    target_block = _normalize_spacing(item.target_event_prompt)

    sections: List[str] = []
    if prime_block:
        sections.append(prime_block)
    sections.append(target_block)
    return "\n\n".join(section for section in sections if section).rstrip()


def _format_question_template(item: ExperimentItem, template: str, target_sentence: str) -> str:
    payload: Dict[str, str] = asdict(item)
    payload["target_sentence"] = target_sentence
    try:
        return template.format(**payload)
    except Exception:
        # Fallback to raw template if no valid placeholders are provided.
        return template


def render_experiment_2_prompt(
    item: ExperimentItem,
    target_sentence: str,
    question_template: str,
    eos_token_text: str = "",
) -> str:
    prime_block = render_prime_block(item=item, eos_token_text=eos_token_text)
    question = _format_question_template(item=item, template=question_template, target_sentence=target_sentence)

    sections: List[str] = []
    if prime_block:
        sections.append(prime_block)
    sections.append(f"Target sentence:\n{_normalize_spacing(target_sentence)}")
    sections.append(f"Question:\n{_normalize_spacing(question)}")
    sections.append("Answer:")
    return "\n\n".join(sections).rstrip()


def question_templates_for_item(
    item: ExperimentItem,
    additional_templates: Iterable[str],
) -> List[str]:
    templates = [item.question_template]
    for template in additional_templates:
        if template and template not in templates:
            templates.append(template)
    return [template for template in templates if template]
