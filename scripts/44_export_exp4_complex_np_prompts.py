#!/usr/bin/env python3
"""Export bare concatenated Experiment 4 complex-NP prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = REPO_ROOT / "behavioral_results/generated_materials/experiment-4/complex_np"
DEFAULT_CORE = GENERATED_DIR / "experiment_4_complex_np_core_role_recovery.csv"
DEFAULT_JABBER = GENERATED_DIR / "experiment_4_complex_np_jabberwocky_role_recovery.csv"
DEFAULT_CORE_OUTPUT = GENERATED_DIR / "experiment_4_complex_np_core_role_recovery_prompts.csv"
DEFAULT_JABBER_OUTPUT = GENERATED_DIR / "experiment_4_complex_np_jabberwocky_role_recovery_prompts.csv"
DEFAULT_SUMMARY = REPO_ROOT / "corpora/transitive/experiment_4_complex_np_role_recovery_prompts_summary.json"
VERB_LIST_PATH = REPO_ROOT / "corpora/transitive/vocabulary_lists/verblist_T_usf_freq.csv"


def portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)

PRIME_CONDITIONS = ("active", "passive", "filler", "no_prime")
TARGET_VOICES = ("active", "passive")
QUESTION_TYPES = ("doer", "acted_on")

CORE_FILLERS_PRESENT = [
    "Clouds drift very quietly today .",
    "Rain stops very suddenly today .",
    "Music fades very softly today .",
    "Birds vanish very quickly today .",
    "Clouds slowly drift very quietly today .",
    "Rain suddenly stops very softly today .",
    "Music softly fades very gently today .",
    "Birds quickly vanish very silently today .",
    "Clouds calmly drift very slowly today .",
    "Rain briefly stops very suddenly today .",
    "Music gradually fades very softly today .",
    "Birds quietly vanish very quickly today .",
]
CORE_FILLERS_PAST = [
    "Clouds drifted very quietly yesterday .",
    "Rain stopped very suddenly yesterday .",
    "Music faded very softly yesterday .",
    "Birds vanished very quickly yesterday .",
    "Clouds slowly drifted very quietly yesterday .",
    "Rain suddenly stopped very softly yesterday .",
    "Music softly faded very gently yesterday .",
    "Birds quickly vanished very silently yesterday .",
    "Clouds calmly drifted very slowly yesterday .",
    "Rain briefly stopped very suddenly yesterday .",
    "Music gradually faded very softly yesterday .",
    "Birds quietly vanished very quickly yesterday .",
]
JABBER_FILLERS_PRESENT = [
    "{noun} s very quietly today .",
    "{noun} s very softly today .",
    "{noun} s very suddenly today .",
    "{noun} s very quickly today .",
    "{noun} s slowly very quietly today .",
    "{noun} s suddenly very softly today .",
    "{noun} s softly very gently today .",
    "{noun} s quickly very silently today .",
    "{noun} s calmly very slowly today .",
    "{noun} s briefly very suddenly today .",
    "{noun} s gradually very softly today .",
    "{noun} s quietly very quickly today .",
]
JABBER_FILLERS_PAST = [
    "{noun} ed very quietly yesterday .",
    "{noun} ed very softly yesterday .",
    "{noun} ed very suddenly yesterday .",
    "{noun} ed very quickly yesterday .",
    "{noun} ed slowly very quietly yesterday .",
    "{noun} ed suddenly very softly yesterday .",
    "{noun} ed softly very gently yesterday .",
    "{noun} ed quickly very silently yesterday .",
    "{noun} ed calmly very slowly yesterday .",
    "{noun} ed briefly very suddenly yesterday .",
    "{noun} ed gradually very softly yesterday .",
    "{noun} ed quietly very quickly yesterday .",
]
_FORM_TO_LEMMA: dict[str, str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export bare concatenated Exp4 prompts: optional prime, target sentence, "
            "role-recovery question, answer prefix."
        )
    )
    parser.add_argument("--core-csv", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--jabber-csv", type=Path, default=DEFAULT_JABBER)
    parser.add_argument("--core-output", type=Path, default=DEFAULT_CORE_OUTPUT)
    parser.add_argument("--jabber-output", type=Path, default=DEFAULT_JABBER_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def form_to_lemma() -> dict[str, str]:
    global _FORM_TO_LEMMA
    if _FORM_TO_LEMMA is not None:
        return _FORM_TO_LEMMA
    mapping: dict[str, str] = {}
    if VERB_LIST_PATH.exists():
        frame = pd.read_csv(VERB_LIST_PATH).fillna("")
        for _, row in frame.iterrows():
            lemma = str(row.get("V", "")).strip().lower()
            if not lemma:
                continue
            for column in ("V", "past_A", "past_P", "pres_3s"):
                form = str(row.get(column, "")).strip().lower()
                if form:
                    mapping[form] = lemma
    _FORM_TO_LEMMA = mapping
    return mapping


def fallback_lemma(verb_form: str) -> str:
    token = str(verb_form).strip().lower()
    if not token:
        return "event"
    if token.endswith("ied") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "sses", "xes", "zes", "oes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 3:
        return token[:-1]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def is_cvc(token: str) -> bool:
    vowels = set("aeiou")
    return (
        len(token) >= 3
        and token[-1] not in vowels
        and token[-2] in vowels
        and token[-3] not in vowels
        and token[-1] not in {"w", "x", "y"}
    )


def nominalize_lemma(lemma: str) -> str:
    token = str(lemma).strip().lower()
    if not token:
        return "event"
    overrides = {
        "be": "being",
        "do": "doing",
        "die": "dying",
        "lie": "lying",
        "tie": "tying",
        "see": "seeing",
    }
    if token in overrides:
        return overrides[token]
    if token.endswith("ie") and len(token) > 2:
        return token[:-2] + "ying"
    if token.endswith("e") and not token.endswith(("ee", "oe", "ye")):
        return token[:-1] + "ing"
    if is_cvc(token):
        return token + token[-1] + "ing"
    return token + "ing"


def event_reference(row: pd.Series, *, source_label: str) -> str:
    if source_label != "core":
        return "event just described"
    verb_form = str(row.get("target_active_verb_or_fragment", "")).strip().lower()
    lemma = form_to_lemma().get(verb_form, fallback_lemma(verb_form))
    nominalized = nominalize_lemma(lemma)
    return f"{nominalized} event"


def target_tense(row: pd.Series) -> str:
    aux = str(row["target_passive_aux"]).strip().lower()
    if aux == "is":
        return "present"
    if aux == "was":
        return "past"
    return "unknown"


def opposite_tense(row: pd.Series) -> str:
    tense = target_tense(row)
    if tense == "present":
        return "past"
    if tense == "past":
        return "present"
    return "present"


def choose_jabber_filler_noun(row: pd.Series, item_index: int) -> str:
    blocked = {
        str(row.get("target_agent", "")).strip(),
        str(row.get("target_patient", "")).strip(),
        str(row.get("agent_modifier_noun", "")).strip(),
        str(row.get("patient_modifier_noun", "")).strip(),
    }
    # The complex corpus already stores all nonce nouns needed for this row. Use
    # a small local pool from the active lexicon, skipping row nouns.
    pool = [
        "bri", "dil", "dru", "gri", "mur", "dir", "dur", "fab", "fav", "fel",
        "fil", "fol", "ful", "gir", "lav", "lil", "mol", "mul", "nav", "nin",
        "nom", "nud", "pav", "pel", "pil", "por", "rav", "rom", "sav", "som",
        "sor", "sul", "tel", "tir", "tul", "vel", "vil", "vir", "vom", "vul",
    ]
    candidates = [noun for noun in pool if noun not in blocked]
    if not candidates:
        raise ValueError("No available Jabberwocky filler noun after blocking row nouns.")
    return candidates[item_index % len(candidates)]


def filler_prime(row: pd.Series, *, source_label: str, row_number: int) -> str:
    tense = opposite_tense(row)
    if source_label == "core":
        pool = CORE_FILLERS_PRESENT if tense == "present" else CORE_FILLERS_PAST
        return pool[row_number % len(pool)]

    pool = JABBER_FILLERS_PRESENT if tense == "present" else JABBER_FILLERS_PAST
    noun = choose_jabber_filler_noun(row, row_number)
    return pool[row_number % len(pool)].format(noun=noun)


def question_text(row: pd.Series, question_type: str, *, source_label: str) -> str:
    reference = event_reference(row, source_label=source_label)
    if question_type == "doer":
        return f"In the {reference}, who was the doer?"
    if question_type == "acted_on":
        return f"In the {reference}, who was the one acted on?"
    raise ValueError(f"Unsupported question_type={question_type!r}")


def answer_prefix(question_type: str) -> str:
    if question_type == "doer":
        return "Answer: The doer was the"
    if question_type == "acted_on":
        return "Answer: The one acted on was the"
    raise ValueError(f"Unsupported question_type={question_type!r}")


def correct_answer(row: pd.Series, question_type: str) -> str:
    if question_type == "doer":
        return f"The doer was the {row['target_agent']}."
    if question_type == "acted_on":
        return f"The one acted on was the {row['target_patient']}."
    raise ValueError(f"Unsupported question_type={question_type!r}")


def foil_answer(row: pd.Series, question_type: str) -> str:
    if question_type == "doer":
        return f"The doer was the {row['target_patient']}."
    if question_type == "acted_on":
        return f"The one acted on was the {row['target_agent']}."
    raise ValueError(f"Unsupported question_type={question_type!r}")


def modifier_distractors(row: pd.Series) -> str:
    values = [
        str(row.get("agent_modifier_noun", "")).strip(),
        str(row.get("patient_modifier_noun", "")).strip(),
    ]
    return ";".join(value for value in values if value and value.lower() != "nan")


def prompt_text(prime_text: str, target_sentence: str, question: str, answer_prefix: str = "The") -> str:
    sections = []
    if prime_text.strip():
        sections.append(prime_text.strip())
    sections.append(target_sentence.strip())
    sections.append(f"{question.strip()}\n{answer_prefix.strip()}")
    return "\n\n".join(sections).rstrip()


def export_prompts(input_csv: Path, output_csv: Path, *, source_label: str) -> pd.DataFrame:
    frame = pd.read_csv(input_csv).fillna("")
    rows: list[dict[str, object]] = []
    for row_number, (_, row) in enumerate(frame.iterrows()):
        for prime_condition in PRIME_CONDITIONS:
            if prime_condition == "active":
                prime = str(row["prime_active"])
            elif prime_condition == "passive":
                prime = str(row["prime_passive"])
            elif prime_condition == "filler":
                prime = filler_prime(row, source_label=source_label, row_number=row_number)
            elif prime_condition == "no_prime":
                prime = ""
            else:
                raise ValueError(f"Unsupported prime condition: {prime_condition}")

            for target_voice in TARGET_VOICES:
                target = (
                    str(row["target_active_complex"])
                    if target_voice == "active"
                    else str(row["target_passive_complex"])
                )
                for question_type in QUESTION_TYPES:
                    question = question_text(row, question_type, source_label=source_label)
                    prefix = answer_prefix(question_type)
                    rows.append(
                        {
                            "prompt_id": f"{row['item_id']}_{prime_condition}_{target_voice}_{question_type}",
                            "item_id": row["item_id"],
                            "item_index": row["item_index"],
                            "complexity_condition": row["complexity_condition"],
                            "source_label": row["source_label"],
                            "lexicality_condition": "real" if source_label == "core" else "nonce",
                            "target_cell": row["target_cell"],
                            "prime_cell": row["prime_cell"],
                            "prime_condition": prime_condition,
                            "target_voice": target_voice,
                            "question_type": question_type,
                            "prime_text": prime,
                            "target_sentence": target,
                            "question_text": question,
                            "answer_prefix": prefix,
                            "prompt": prompt_text(prime, target, question, answer_prefix=prefix),
                            "correct_answer": correct_answer(row, question_type),
                            "foil_answer": foil_answer(row, question_type),
                            "modifier_distractors": modifier_distractors(row),
                            "target_agent": row["target_agent"],
                            "target_patient": row["target_patient"],
                            "agent_modifier_noun": row["agent_modifier_noun"],
                            "patient_modifier_noun": row["patient_modifier_noun"],
                            "agent_modifier_prep": row["agent_modifier_prep"],
                            "patient_modifier_prep": row["patient_modifier_prep"],
                            "target_active_complex": row["target_active_complex"],
                            "target_passive_complex": row["target_passive_complex"],
                        }
                    )
    output = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    return output


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    filler_lengths = [
        len(str(value).split())
        for value in frame.loc[frame["prime_condition"] == "filler", "prime_text"]
    ]
    return {
        "row_count": int(len(frame)),
        "counts_by_prime_condition": {
            str(k): int(v) for k, v in frame["prime_condition"].value_counts().sort_index().items()
        },
        "counts_by_target_voice": {
            str(k): int(v) for k, v in frame["target_voice"].value_counts().sort_index().items()
        },
        "counts_by_question_type": {
            str(k): int(v) for k, v in frame["question_type"].value_counts().sort_index().items()
        },
        "counts_by_complexity": {
            str(k): int(v) for k, v in frame["complexity_condition"].value_counts().sort_index().items()
        },
        "filler_prime_lengths": sorted(set(filler_lengths)),
        "filler_prime_length_counts": {
            str(k): int(v)
            for k, v in pd.Series(filler_lengths).value_counts().sort_index().items()
        },
    }


def main() -> None:
    args = parse_args()
    core = export_prompts(args.core_csv, args.core_output, source_label="core")
    jabber = export_prompts(args.jabber_csv, args.jabber_output, source_label="jabberwocky")
    summary = {
        "description": (
            "Bare concatenated Exp4 prompts. Format: optional prime sentence, blank line, "
            "target sentence, blank line, role question, newline, answer prefix."
        ),
        "core_output": portable_path(args.core_output),
        "jabberwocky_output": portable_path(args.jabber_output),
        "questions": {
            "core_doer": "In the [target nominalized-verb] event, who was the doer?",
            "core_acted_on": "In the [target nominalized-verb] event, who was the one acted on?",
            "jabberwocky_doer": "In the event just described, who was the doer?",
            "jabberwocky_acted_on": "In the event just described, who was the one acted on?",
        },
        "answer_prefixes": {
            "doer": "Answer: The doer was the",
            "acted_on": "Answer: The one acted on was the",
        },
        "filler_note": (
            "Fillers use no determiners and no PP prepositions. They are assigned the opposite "
            "tense from the target where possible, match the earlier 6/7-token filler length "
            "distribution, and serve as unrelated sentence baselines without introducing target "
            "preposition overlap."
        ),
        "core": summarize(core),
        "jabberwocky": summarize(jabber),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote CORE prompts: {args.core_output}")
    print(f"Wrote Jabberwocky prompts: {args.jabber_output}")
    print(f"Wrote summary: {args.summary_json}")
    print(json.dumps({"core": summary["core"], "jabberwocky": summary["jabberwocky"]}, indent=2))


if __name__ == "__main__":
    main()
