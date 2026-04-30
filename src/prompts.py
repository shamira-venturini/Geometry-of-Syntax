from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List

from .data import ExperimentItem

IRREGULAR_PARTICIPLE_TO_ING: Dict[str, str] = {
    "beaten": "beating",
    "caught": "catching",
    "forgotten": "forgetting",
    "forgiven": "forgiving",
    "hurt": "hurting",
    "struck": "striking",
    "taught": "teaching",
    "understood": "understanding",
}


def _normalize_spacing(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def _normalize_sentence(text: str) -> str:
    compact = " ".join(str(text).strip().split())
    compact = compact.replace(" .", ".")
    compact = compact.replace(" ,", ",")
    if not compact:
        return compact
    return compact[0].upper() + compact[1:]


def _strip_period_tokens(sentence: str) -> List[str]:
    tokens = [token for token in str(sentence).strip().split() if token != "."]
    if tokens and tokens[-1].endswith("."):
        tokens[-1] = tokens[-1][:-1]
    return [token for token in tokens if token]


def _to_ing(verb: str) -> str:
    token = verb.lower().strip()
    if not token:
        return "doing"
    if token in IRREGULAR_PARTICIPLE_TO_ING:
        return IRREGULAR_PARTICIPLE_TO_ING[token]
    stripped_past_suffix = False
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
        and len(token) >= 3
        and token[-1] not in "aeiouy"
        and token[-2] in "aeiou"
        and token[-3] not in "aeiou"
    ):
        return token + token[-1] + "ing"
    return token + "ing"


def _uses_fragment_verb(verb: str) -> bool:
    return verb.lower().strip() in {"s", "ed"}


def _event_article(event_name: str) -> str:
    return "an" if event_name[:1].lower() in "aeiou" else "a"


def _parse_prime_event_fields(prime_sentence: str) -> tuple[str, str, str]:
    tokens = [token.lower() for token in _strip_period_tokens(prime_sentence)]
    if len(tokens) < 5:
        raise ValueError(f"Prime sentence is too short to parse: {prime_sentence!r}")

    if "by" in tokens and len(tokens) >= 7:
        by_index = tokens.index("by")
        if by_index >= 4 and by_index + 2 < len(tokens):
            patient_phrase = " ".join(tokens[0:2])
            verb = tokens[by_index - 1]
            agent_phrase = " ".join(tokens[by_index + 1 : by_index + 3])
            return agent_phrase, patient_phrase, verb

    agent_phrase = " ".join(tokens[0:2])
    verb = tokens[2]
    patient_phrase = " ".join(tokens[3:5])
    return agent_phrase, patient_phrase, verb


def _role_lines(agent_phrase: str, patient_phrase: str, role_order: str) -> List[str]:
    if role_order == "agent_first":
        return [
            f"The one who did it was {agent_phrase}.",
            f"The one it happened to was {patient_phrase}.",
        ]
    if role_order == "patient_first":
        return [
            f"The one it happened to was {patient_phrase}.",
            f"The one who did it was {agent_phrase}.",
        ]
    raise ValueError(f"Unsupported role_order: {role_order}")


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


def render_experiment_2_demo_prime_block(item: ExperimentItem) -> str:
    condition = item.prime_condition
    if condition in {"no_prime", "no_prime_eos", "no_prime_empty", "no_demo"}:
        return ""

    prime_sentence_raw = " ".join(str(item.prime_text).strip().split())
    prime_sentence = _normalize_sentence(prime_sentence_raw)
    if condition == "filler":
        return _normalize_spacing(
            "\n".join(
                [
                    "In another scene, something unrelated happened.",
                    'Bridget asked, "What happened?"',
                    f'Mary answered, "{prime_sentence}"',
                ]
            )
        )

    try:
        agent_phrase, patient_phrase, verb = _parse_prime_event_fields(prime_sentence_raw)
    except Exception:
        # Defensive fallback: preserve the prime as a solved example.
        return _normalize_spacing(
            "\n".join(
                [
                    'Bridget asked, "What happened?"',
                    f'Mary answered, "{prime_sentence}"',
                ]
            )
        )

    role_lines = _role_lines(
        agent_phrase=agent_phrase,
        patient_phrase=patient_phrase,
        role_order=item.role_order,
    )
    if _uses_fragment_verb(verb):
        return _normalize_spacing(
            "\n".join(
                [
                    f"There was an event involving {agent_phrase} and {patient_phrase}.",
                    role_lines[0],
                    role_lines[1],
                    "",
                    'Bridget asked, "What happened?"',
                    f'Mary answered, "{prime_sentence}"',
                ]
            )
        )

    event_name = _to_ing(verb)
    return _normalize_spacing(
        "\n".join(
            [
                f"There was {_event_article(event_name)} {event_name} event involving {agent_phrase} and {patient_phrase}.",
                role_lines[0],
                role_lines[1],
                "",
                'Bridget asked, "What happened?"',
                f'Mary answered, "{prime_sentence}"',
            ]
        )
    )


def render_experiment_1_prompt(item: ExperimentItem, eos_token_text: str = "") -> str:
    prime_block = render_prime_block(item=item, eos_token_text=eos_token_text)
    target_block = _normalize_spacing(item.target_event_prompt)

    sections: List[str] = []
    if prime_block:
        sections.append(prime_block)
    sections.append(target_block)
    return "\n\n".join(section for section in sections if section).rstrip()


def render_experiment_2_continuation_prompt(item: ExperimentItem) -> str:
    prime_block = render_experiment_2_demo_prime_block(item=item)
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
