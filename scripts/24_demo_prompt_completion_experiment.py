import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from production_priming_common import (
    CORE_FILLER_SENTENCES,
    JABBERWOCKY_FILLER_SENTENCES,
    LEXICALLY_CONTROLLED_CORE_CSV,
    REPO_ROOT,
    TargetBundle,
    batched_choice_log_probs,
    extract_bundle,
    get_device,
    lexical_overlap_audit,
    load_causal_lm_and_tokenizer,
    load_verb_lookup,
    normalize_transitive_frame,
    sample_condition_frames,
    write_common_outputs,
)


DEFAULT_JABBERWOCKY_PRIMES = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_gpt2_monosyllabic_strict_4cell.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "behavioral_results" / "experiment-2" / "demo_prompt_completion"
REAL_VERB_ING = {
    "ask": "asking",
    "beat": "beating",
    "bother": "bothering",
    "carry": "carrying",
    "catch": "catching",
    "chase": "chasing",
    "comfort": "comforting",
    "describe": "describing",
    "destroy": "destroying",
    "discover": "discovering",
    "drag": "dragging",
    "embrace": "embracing",
    "follow": "following",
    "forget": "forgetting",
    "forgive": "forgiving",
    "grab": "grabbing",
    "help": "helping",
    "hurry": "hurrying",
    "hurt": "hurting",
    "join": "joining",
    "judge": "judging",
    "kill": "killing",
    "kiss": "kissing",
    "protect": "protecting",
    "punch": "punching",
    "raise": "raising",
    "select": "selecting",
    "strike": "striking",
    "surprise": "surprising",
    "teach": "teaching",
    "understand": "understanding",
    "wash": "washing",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the demonstration-based completion-choice production experiment."
    )
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--input-csv", type=Path, default=LEXICALLY_CONTROLLED_CORE_CSV)
    parser.add_argument("--prime-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Torch dtype for model loading: auto, float32, float16, or bfloat16.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer/model from the local Hugging Face cache only.",
    )
    parser.add_argument(
        "--prime-conditions",
        nargs="+",
        default=["active", "passive", "no_prime", "filler"],
        help="Subset of active passive no_prime filler.",
    )
    parser.add_argument(
        "--filler-domain",
        choices=("auto", "core", "jabberwocky"),
        default="auto",
        help="Filler pool to use. auto infers from input/prime CSV path.",
    )
    parser.add_argument(
        "--event-style",
        choices=("there_was_event", "involving_event", "all"),
        default="involving_event",
        help="How the event itself is described in the scaffold.",
    )
    parser.add_argument(
        "--role-style",
        choices=("responsible_affected", "did_to", "all"),
        default="did_to",
        help="How participant roles are described in the scaffold.",
    )
    parser.add_argument(
        "--quote-style",
        choices=("mary_answered", "said_mary", "all"),
        default="mary_answered",
        help="How the demonstrated and target answers are framed.",
    )
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def pretty_sentence(text: str) -> str:
    tokens = text.strip().split()
    if not tokens:
        return ""
    body = " ".join(tokens)
    body = body.replace(" .", ".")
    return body[0].upper() + body[1:]


def to_event_label(lemma: str) -> str:
    lower = lemma.lower()
    if lower in {"s", "ed"}:
        return ""
    if lower in REAL_VERB_ING:
        return REAL_VERB_ING[lower]
    if lower.endswith("ie") and len(lower) > 2:
        return lower[:-2] + "ying"
    if lower.endswith("e") and not lower.endswith(("ee", "oe", "ye")):
        return lower[:-1] + "ing"
    if lower.endswith("y") and len(lower) > 1:
        return lower + "ing"
    return lower + "ing"


def event_article(event_label: str) -> str:
    return "an" if event_label[:1].lower() in "aeiou" else "a"


def noun_phrase(det: str, noun: str, capitalize: bool = False) -> str:
    phrase = f"{det} {noun}"
    if not capitalize:
        return phrase
    return phrase[0].upper() + phrase[1:]


def event_style_values(mode: str) -> List[str]:
    if mode == "all":
        return ["there_was_event", "involving_event"]
    return [mode]


def role_style_values(mode: str) -> List[str]:
    if mode == "all":
        return ["responsible_affected", "did_to"]
    return [mode]


def quote_style_values(mode: str) -> List[str]:
    if mode == "all":
        return ["mary_answered", "said_mary"]
    return [mode]


def event_lines(bundle: TargetBundle, event_style: str, role_style: str) -> List[str]:
    agent = noun_phrase(bundle.agent_det, bundle.agent_noun)
    patient = noun_phrase(bundle.patient_det, bundle.patient_noun)
    event_label = to_event_label(bundle.verb_lemma)

    if event_style == "there_was_event" and event_label:
        lines = [
            f"{noun_phrase(bundle.agent_det, bundle.agent_noun, capitalize=True)} and {patient} were involved in the same event.",
            f"There was {event_article(event_label)} {event_label} event.",
        ]
    elif event_style == "there_was_event":
        lines = [
            f"{noun_phrase(bundle.agent_det, bundle.agent_noun, capitalize=True)} and {patient} were involved in the same event.",
            "There was an event.",
        ]
    elif event_style == "involving_event" and event_label:
        lines = [
            f"There was {event_article(event_label)} {event_label} event involving {agent} and {patient}.",
        ]
    elif event_style == "involving_event":
        lines = [
            f"There was an event involving {agent} and {patient}.",
        ]
    else:
        raise ValueError(f"Unsupported event style: {event_style}")

    if role_style == "responsible_affected":
        lines.extend(
            [
                f"The responsible person was {agent}.",
                f"The affected person was {patient}.",
            ]
        )
    elif role_style == "did_to":
        lines.extend(
            [
                f"The one who did it was {agent}.",
                f"The one it happened to was {patient}.",
            ]
        )
    else:
        raise ValueError(f"Unsupported role style: {role_style}")

    return lines


def quote_prefix(style: str) -> str:
    if style == "mary_answered":
        return 'Mary answered, "'
    if style == "said_mary":
        return '"'
    raise ValueError(f"Unsupported quote style: {style}")


def quote_suffix(style: str) -> str:
    if style == "mary_answered":
        return '"'
    if style == "said_mary":
        return '" said Mary.'
    raise ValueError(f"Unsupported quote style: {style}")


def render_answer_line(answer_text: str, style: str) -> str:
    return f"{quote_prefix(style)}{answer_text}{quote_suffix(style)}"


def filler_demo_lines(filler_sentence: str, quote_style: str) -> List[str]:
    return [
        "In another scene, something unrelated happened.",
        'Bridget asked, "What happened?"',
        render_answer_line(pretty_sentence(filler_sentence), quote_style),
    ]


def demo_block(bundle: TargetBundle, answer_sentence: str, event_style: str, role_style: str, quote_style: str) -> List[str]:
    return event_lines(bundle, event_style=event_style, role_style=role_style) + [
        "",
        'Bridget asked, "What happened?"',
        render_answer_line(pretty_sentence(answer_sentence), quote_style),
    ]


def target_block(bundle: TargetBundle, event_style: str, role_style: str, quote_style: str) -> List[str]:
    return event_lines(bundle, event_style=event_style, role_style=role_style) + [
        "",
        'Bridget asked, "What happened?"',
        f'{quote_prefix(quote_style)}The',
    ]


def demo_prime_condition_order(raw_conditions: List[str]) -> List[str]:
    alias_map = {
        "no_prime": "no_prime",
        "no_prime_empty": "no_prime",
        "no_demo": "no_prime",
        "none": "no_prime",
    }
    conditions = [alias_map.get(condition.strip(), condition.strip()) for condition in raw_conditions if condition.strip()]
    allowed = {"active", "passive", "no_prime", "filler"}
    invalid = sorted(set(conditions).difference(allowed))
    if invalid:
        raise ValueError(f"Unsupported prime conditions: {invalid}")
    if not conditions:
        raise ValueError("At least one prime condition is required.")
    ordered: List[str] = []
    seen = set()
    for condition in conditions:
        if condition not in seen:
            ordered.append(condition)
            seen.add(condition)
    return ordered


def infer_filler_domain(input_csv: Path, prime_csv: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    probe = f"{input_csv.name} {prime_csv.name}".lower()
    return "jabberwocky" if "jabberwocky" in probe else "core"


def _passive_aux(sentence: str) -> str:
    tokens = str(sentence).strip().split()
    if len(tokens) != 8:
        raise ValueError(f"Unexpected passive sentence format: {sentence}")
    return tokens[2].lower()


def _det_family_from_active(sentence: str) -> str:
    tokens = str(sentence).strip().split()
    if len(tokens) != 6:
        raise ValueError(f"Unexpected active sentence format: {sentence}")
    det = tokens[0].lower()
    if det == "the":
        return "def"
    if det in {"a", "an"}:
        return "indef"
    raise ValueError(f"Unexpected determiner in active sentence: {sentence}")


def assert_aux_det_mismatch(
    *,
    target_frame: pd.DataFrame,
    prime_frame: pd.DataFrame,
    label: str,
) -> None:
    same_aux_rows = 0
    same_det_family_rows = 0
    preview: List[Dict[str, object]] = []
    for item_index, target_row in target_frame.iterrows():
        prime_row = prime_frame.loc[item_index]
        prime_aux = _passive_aux(str(prime_row["pp"]))
        target_aux = _passive_aux(str(target_row["tp"]))
        prime_det_family = _det_family_from_active(str(prime_row["pa"]))
        target_det_family = _det_family_from_active(str(target_row["ta"]))
        if prime_aux == target_aux:
            same_aux_rows += 1
            if len(preview) < 5:
                preview.append(
                    {
                        "item_index": int(item_index),
                        "prime_aux": prime_aux,
                        "target_aux": target_aux,
                        "prime_pp": str(prime_row["pp"]),
                        "target_tp": str(target_row["tp"]),
                    }
                )
        if prime_det_family == target_det_family:
            same_det_family_rows += 1
            if len(preview) < 5:
                preview.append(
                    {
                        "item_index": int(item_index),
                        "prime_det_family": prime_det_family,
                        "target_det_family": target_det_family,
                        "prime_pa": str(prime_row["pa"]),
                        "target_ta": str(target_row["ta"]),
                    }
                )
    if same_aux_rows:
        raise ValueError(
            f"{label}: found {same_aux_rows}/{len(target_frame)} rows where prime/target passive auxiliaries overlap. "
            f"Preview: {preview}"
        )
    if same_det_family_rows:
        raise ValueError(
            f"{label}: found {same_det_family_rows}/{len(target_frame)} rows where prime/target determiner families overlap. "
            f"Preview: {preview}"
        )


def sample_prime_frame_with_aux_det_mismatch(
    *,
    target_sample: pd.DataFrame,
    prime_frame: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    if len(prime_frame) < len(target_sample):
        raise ValueError(
            f"Prime corpus has too few rows to align with aux+determiner mismatch: "
            f"{len(prime_frame)} < {len(target_sample)}"
        )

    prime_aux = prime_frame["pp"].map(_passive_aux)
    prime_det = prime_frame["pa"].map(_det_family_from_active)
    unexpected_aux = sorted(set(prime_aux).difference({"is", "was"}))
    if unexpected_aux:
        raise ValueError(f"Unexpected passive auxiliaries in prime frame: {unexpected_aux}")
    unexpected_det = sorted(set(prime_det).difference({"def", "indef"}))
    if unexpected_det:
        raise ValueError(f"Unexpected determiner families in prime frame: {unexpected_det}")

    buckets: Dict[Tuple[str, str], List[int]] = {}
    for aux in ("is", "was"):
        for det in ("def", "indef"):
            mask = prime_aux.eq(aux) & prime_det.eq(det)
            buckets[(aux, det)] = prime_aux[mask].index.tolist()

    rng = np.random.default_rng(seed + 104729)
    for key in buckets:
        rng.shuffle(buckets[key])

    selected_indices: List[int] = []
    for _, target_row in target_sample.iterrows():
        target_aux = _passive_aux(str(target_row["tp"]))
        target_det = _det_family_from_active(str(target_row["ta"]))
        required_prime_aux = "was" if target_aux == "is" else "is"
        required_prime_det = "def" if target_det == "indef" else "indef"
        pool = buckets[(required_prime_aux, required_prime_det)]
        if not pool:
            raise ValueError(
                "Cannot satisfy strict mismatch: exhausted prime rows with "
                f"aux='{required_prime_aux}' and det='{required_prime_det}'."
            )
        selected_indices.append(pool.pop())

    return prime_frame.loc[selected_indices].reset_index(drop=True)


def build_prompt(
    target_bundle: TargetBundle,
    prime_condition: str,
    prime_bundle: Optional[TargetBundle],
    prime_sentence: Optional[str],
    filler_sentence: Optional[str],
    event_style: str,
    role_style: str,
    quote_style: str,
) -> str:
    lines: List[str] = []
    if prime_condition == "active":
        if prime_bundle is None or prime_sentence is None:
            raise ValueError("Active demo requested without prime bundle/sentence.")
        lines.extend(
            demo_block(
                prime_bundle,
                answer_sentence=prime_sentence,
                event_style=event_style,
                role_style=role_style,
                quote_style=quote_style,
            )
        )
        lines.append("")
    elif prime_condition == "passive":
        if prime_bundle is None or prime_sentence is None:
            raise ValueError("Passive demo requested without prime bundle/sentence.")
        lines.extend(
            demo_block(
                prime_bundle,
                answer_sentence=prime_sentence,
                event_style=event_style,
                role_style=role_style,
                quote_style=quote_style,
            )
        )
        lines.append("")
    elif prime_condition == "filler":
        if filler_sentence is None:
            raise ValueError("Filler demo requested without filler sentence.")
        lines.extend(filler_demo_lines(filler_sentence=filler_sentence, quote_style=quote_style))
        lines.append("")
    elif prime_condition == "no_prime":
        pass
    else:
        raise ValueError(f"Unsupported prime condition: {prime_condition}")

    lines.extend(
        target_block(
            bundle=target_bundle,
            event_style=event_style,
            role_style=role_style,
            quote_style=quote_style,
        )
    )
    return "\n".join(lines)


def build_prompt_groups(
    target_frame: pd.DataFrame,
    prime_frame: pd.DataFrame,
    tokenizer,
    prime_conditions: List[str],
    filler_sentences: List[str],
    seed: int,
    event_style_mode: str,
    role_style_mode: str,
    quote_style: str,
) -> Tuple[List[Tuple[str, int, List[str], List[int]]], List[Dict[str, object]]]:
    verb_lookup = load_verb_lookup()
    prompt_groups: List[Tuple[str, int, List[str], List[int]]] = []
    row_metadata: List[Dict[str, object]] = []

    for item_index, target_row in target_frame.iterrows():
        prime_row = prime_frame.loc[item_index]
        target_bundle = extract_bundle(str(target_row["ta"]), str(target_row["tp"]), verb_lookup=verb_lookup)
        prime_bundle = extract_bundle(str(prime_row["pa"]), str(prime_row["pp"]), verb_lookup=verb_lookup)
        filler_sentence = filler_sentences[item_index % len(filler_sentences)]

        candidates = [f" {target_bundle.agent_noun}", f" {target_bundle.patient_noun}"]
        candidate_lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in candidates]

        for event_style in event_style_values(event_style_mode):
            for role_style in role_style_values(role_style_mode):
                for quote_style_value in quote_style_values(quote_style):
                    prompt_template_name = f"demo__{event_style}__{role_style}__{quote_style_value}"
                    for prime_condition in prime_conditions:
                        prime_sentence: Optional[str]
                        if prime_condition == "active":
                            prime_sentence = str(prime_row["pa"])
                        elif prime_condition == "passive":
                            prime_sentence = str(prime_row["pp"])
                        else:
                            prime_sentence = None

                        prompt = build_prompt(
                            target_bundle=target_bundle,
                            prime_condition=prime_condition,
                            prime_bundle=prime_bundle,
                            prime_sentence=prime_sentence,
                            filler_sentence=filler_sentence if prime_condition == "filler" else None,
                            event_style=event_style,
                            role_style=role_style,
                            quote_style=quote_style_value,
                        )
                        prompt_groups.append(
                            (
                                prompt,
                                len(tokenizer(prompt, add_special_tokens=False)["input_ids"]),
                                candidates,
                                candidate_lengths,
                            )
                        )
                        row_metadata.append(
                            {
                                "item_index": item_index,
                                "prompt_template": prompt_template_name,
                                "prime_condition": prime_condition,
                                "prime_structure": prime_condition,
                                "prime_sentence": prime_sentence or "",
                                "target_active": str(target_row["ta"]),
                                "target_passive": str(target_row["tp"]),
                                "agent_det": target_bundle.agent_det,
                                "agent_noun": target_bundle.agent_noun,
                                "patient_det": target_bundle.patient_det,
                                "patient_noun": target_bundle.patient_noun,
                                "verb_lemma": target_bundle.verb_lemma,
                                "active_verb_form": target_bundle.active_verb_form,
                                "passive_verb_form": target_bundle.passive_verb_form,
                                "message_role_order_json": json.dumps(
                                    [
                                        f"event_style={event_style}",
                                        f"role_style={role_style}",
                                        f"quote_style={quote_style_value}",
                                        f"event={to_event_label(target_bundle.verb_lemma)}",
                                    ]
                                ),
                                "prompt": prompt,
                                "sentence_stub": f'{quote_prefix(quote_style_value)}The',
                                "choice_target": "first_noun",
                                "quote_style": quote_style_value,
                                "event_style": event_style,
                                "role_style": role_style,
                            }
                        )

    return prompt_groups, row_metadata


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    input_csv = args.input_csv.resolve()
    prime_csv = args.prime_csv.resolve() if args.prime_csv else input_csv
    target_source = normalize_transitive_frame(pd.read_csv(input_csv))
    prime_source = normalize_transitive_frame(pd.read_csv(prime_csv))

    if input_csv == prime_csv:
        target_frame, prime_frame, prime_alignment_mode = sample_condition_frames(
            target_frame=target_source,
            prime_frame=prime_source,
            max_items=args.max_items,
            seed=args.seed,
        )
    else:
        target_frame, _, _ = sample_condition_frames(
            target_frame=target_source,
            prime_frame=target_source,
            max_items=args.max_items,
            seed=args.seed,
        )
        prime_frame = sample_prime_frame_with_aux_det_mismatch(
            target_sample=target_frame,
            prime_frame=prime_source,
            seed=args.seed,
        )
        prime_alignment_mode = "aux_det_mismatch_matched"

    assert_aux_det_mismatch(
        target_frame=target_frame,
        prime_frame=prime_frame,
        label=f"{input_csv.name}<-{prime_csv.name}",
    )
    overlap_audit = lexical_overlap_audit(target_frame=target_frame, prime_frame=prime_frame)

    device = get_device(args.device)
    tokenizer, model, resolved_dtype = load_causal_lm_and_tokenizer(
        model_name=args.model_name,
        device=device,
        local_files_only=args.local_files_only,
        torch_dtype_name=args.torch_dtype,
    )

    prime_conditions = demo_prime_condition_order(list(args.prime_conditions))
    filler_domain = infer_filler_domain(input_csv=input_csv, prime_csv=prime_csv, requested=args.filler_domain)
    filler_sentences = JABBERWOCKY_FILLER_SENTENCES if filler_domain == "jabberwocky" else CORE_FILLER_SENTENCES
    prompt_groups, row_metadata = build_prompt_groups(
        target_frame=target_frame,
        prime_frame=prime_frame,
        tokenizer=tokenizer,
        prime_conditions=prime_conditions,
        filler_sentences=filler_sentences,
        seed=args.seed,
        event_style_mode=args.event_style,
        role_style_mode=args.role_style,
        quote_style=args.quote_style,
    )
    batched_scores = batched_choice_log_probs(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt_groups=prompt_groups,
        batch_size=args.batch_size,
    )

    rows: List[Dict[str, object]] = []
    for metadata, candidate_log_probs in zip(row_metadata, batched_scores):
        active_lp, passive_lp = candidate_log_probs
        chosen_structure = "passive" if passive_lp > active_lp else "active"
        rows.append(
            {
                **metadata,
                "active_choice_logprob": active_lp,
                "passive_choice_logprob": passive_lp,
                "passive_minus_active_logprob": passive_lp - active_lp,
                "chosen_structure": chosen_structure,
                "passive_choice_indicator": 1.0 if chosen_structure == "passive" else 0.0,
            }
        )

    results = pd.DataFrame(rows)
    metadata = {
        "model_name": args.model_name,
        "input_csv": str(input_csv),
        "prime_csv": str(prime_csv),
        "max_items": None if args.max_items is None else int(args.max_items),
        "batch_size": int(args.batch_size),
        "prime_conditions": prime_conditions,
        "seed": int(args.seed),
        "filler_domain": filler_domain,
        "device": device,
        "torch_dtype": str(resolved_dtype) if resolved_dtype is not None else "default",
        "prime_alignment_mode": prime_alignment_mode,
        "lexical_overlap_audit": overlap_audit,
        "quote_style": args.quote_style,
        "event_style": args.event_style,
        "role_style": args.role_style,
        "paradigm": "demonstration_prompt_completion",
    }
    write_common_outputs(
        frame=results,
        output_dir=output_dir,
        title="Demonstration-based completion-choice production experiment",
        prime_condition_ordering=prime_conditions,
        extra_metadata=metadata,
    )
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
