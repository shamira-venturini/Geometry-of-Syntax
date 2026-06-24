#!/usr/bin/env python3
"""Build strict Experiment 4 complex-NP role-recovery corpora."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORE = REPO_ROOT / "corpora/transitive/CORE_transitive_strict_4cell_counterbalanced.csv"
DEFAULT_JABBER = REPO_ROOT / "corpora/transitive/jabberwocky_transitive_gpt2_monosyllabic_strict_4cell.csv"
DEFAULT_NOUNS = REPO_ROOT / "corpora/transitive/vocabulary_lists/nounlist_usf_freq.csv"
DEFAULT_JABBER_LEXICON = (
    REPO_ROOT / "corpora/transitive/vocabulary_lists/jabberwocky_gpt2_monosyllabic_strict_4cell_lexicon.json"
)
GENERATED_DIR = REPO_ROOT / "behavioral_results/generated_materials/experiment-4/complex_np"
DEFAULT_CORE_OUTPUT = GENERATED_DIR / "experiment_4_complex_np_core_role_recovery.csv"
DEFAULT_JABBER_OUTPUT = GENERATED_DIR / "experiment_4_complex_np_jabberwocky_role_recovery.csv"
DEFAULT_SUMMARY = REPO_ROOT / "corpora/transitive/experiment_4_complex_np_role_recovery_summary.json"

TEXT_COLUMNS = ("pa", "pp", "ta", "tp")
COMPLEXITY_CONDITIONS = ("agent_complex", "patient_complex", "both_complex")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")


def portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)

# Keep preposition meanings far apart inside both-complex targets. Prime sentences
# are simple active/passive 1b sentences, so they contain no PP preposition except
# the unavoidable passive by-marker in passive primes.
PREPOSITION_GROUPS = {
    "proximity": ["near"],
    "vertical": ["above", "below"],
    "support": ["on"],
    "containment": ["in"],
    "source": ["from"],
    "instrument": ["with"],
}
PREPOSITION_SEQUENCE = [
    ("near", "vertical"),
    ("above", "proximity"),
    ("with", "source"),
    ("from", "instrument"),
    ("in", "support"),
    ("on", "containment"),
    ("below", "proximity"),
]
SEMANTIC_PREP_GROUP = {
    prep: group for group, preps in PREPOSITION_GROUPS.items() for prep in preps
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Experiment 4 sentence-to-event role-recovery materials. "
            "Primes reuse the strict 1b backbone; targets add controlled PP-complex NPs."
        )
    )
    parser.add_argument("--core-csv", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--jabber-csv", type=Path, default=DEFAULT_JABBER)
    parser.add_argument("--noun-list", type=Path, default=DEFAULT_NOUNS)
    parser.add_argument("--jabber-lexicon", type=Path, default=DEFAULT_JABBER_LEXICON)
    parser.add_argument("--core-output", type=Path, default=DEFAULT_CORE_OUTPUT)
    parser.add_argument("--jabber-output", type=Path, default=DEFAULT_JABBER_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def read_backbone(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip().str.lower()
    missing = set(TEXT_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return frame[list(TEXT_COLUMNS)].copy()


def parse_active(sentence: str) -> tuple[str, str, str, str, str]:
    tokens = str(sentence).strip().split()
    if len(tokens) != 6 or tokens[-1] != ".":
        raise ValueError(f"Unexpected active sentence format: {sentence}")
    return tokens[0].lower(), tokens[1].lower(), tokens[2].lower(), tokens[3].lower(), tokens[4].lower()


def parse_passive(sentence: str) -> tuple[str, str, str, str, str, str]:
    tokens = str(sentence).strip().split()
    if len(tokens) != 8 or tokens[4].lower() != "by" or tokens[-1] != ".":
        raise ValueError(f"Unexpected passive sentence format: {sentence}")
    return (
        tokens[0].lower(),
        tokens[1].lower(),
        tokens[2].lower(),
        tokens[3].lower(),
        tokens[5].lower(),
        tokens[6].lower(),
    )


def det_family(det: str) -> str:
    if det == "the":
        return "def"
    if det in {"a", "an"}:
        return "indef"
    return "unknown"


def indefinite_for(noun: str, known_indefinites: dict[str, str] | None = None) -> str:
    if known_indefinites and noun in known_indefinites:
        det = known_indefinites[noun]
        if det in {"a", "an"}:
            return det
    return "an" if noun[:1].lower() not in CONSONANTS else "a"


def det_for_family(noun: str, family: str, known_indefinites: dict[str, str] | None = None) -> str:
    if family == "def":
        return "the"
    if family == "indef":
        return indefinite_for(noun, known_indefinites)
    raise ValueError(f"Unsupported determiner family: {family}")


def load_human_nouns(path: Path) -> tuple[list[str], dict[str, str]]:
    frame = pd.read_csv(path, sep=";")
    frame.columns = frame.columns.str.strip()
    # Use person nouns, not broader cat_main=human, because group nouns such as
    # "club" are less natural answers to who-questions and would weaken the
    # modifier-distractor control.
    human = (
        frame.loc[frame["category"].astype(str).str.lower().eq("person")]
        .sort_values("F_rank")
        .reset_index(drop=True)
    )
    nouns = human["nouns"].astype(str).str.lower().tolist()
    indefinites = {
        str(row.nouns).lower(): str(row.det_a).lower()
        for row in frame.itertuples(index=False)
    }
    return nouns, indefinites


def load_jabber_nouns(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    nouns = [str(noun).strip().lower() for noun in data["nouns"]]
    if len(nouns) < 8:
        raise ValueError(f"Need at least 8 Jabberwocky nouns, found {len(nouns)}")
    return nouns


def choose_distinct(pool: Sequence[str], used: Iterable[str], offset: int, count: int) -> list[str]:
    used_set = set(used)
    chosen: list[str] = []
    n = len(pool)
    cursor = offset % n
    attempts = 0
    while len(chosen) < count and attempts < n * 3:
        candidate = pool[cursor % n]
        if candidate not in used_set and candidate not in chosen:
            chosen.append(candidate)
        cursor += 1
        attempts += 1
    if len(chosen) < count:
        raise ValueError(f"Could not choose {count} distinct modifiers from pool of {len(pool)}")
    return chosen


def prepositions_for(index: int, complexity_condition: str) -> tuple[str | None, str | None]:
    first_prep, second_group = PREPOSITION_SEQUENCE[index % len(PREPOSITION_SEQUENCE)]
    second_candidates = PREPOSITION_GROUPS[second_group]
    second_prep = second_candidates[(index // len(PREPOSITION_SEQUENCE)) % len(second_candidates)]
    if SEMANTIC_PREP_GROUP[first_prep] == SEMANTIC_PREP_GROUP[second_prep]:
        raise ValueError(f"Preposition semantic-group collision: {first_prep}, {second_prep}")
    if complexity_condition == "agent_complex":
        return first_prep, None
    if complexity_condition == "patient_complex":
        return None, first_prep
    if complexity_condition == "both_complex":
        return first_prep, second_prep
    raise ValueError(f"Unsupported complexity condition: {complexity_condition}")


def maybe_complex_np(
    det: str,
    head: str,
    prep: str | None,
    modifier_det: str | None,
    modifier_noun: str | None,
) -> str:
    if prep is None:
        return f"{det} {head}"
    if modifier_det is None or modifier_noun is None:
        raise ValueError("Complex NP requested without modifier determiner/noun")
    return f"{det} {head} {prep} {modifier_det} {modifier_noun}"


def complex_targets(
    active: str,
    passive: str,
    *,
    complexity_condition: str,
    agent_modifier: str | None,
    patient_modifier: str | None,
    agent_prep: str | None,
    patient_prep: str | None,
    known_indefinites: dict[str, str] | None = None,
) -> tuple[str, str]:
    det_agent, agent, verb, det_patient, patient = parse_active(active)
    p_det_patient, p_patient, aux, participle, p_det_agent, p_agent = parse_passive(passive)
    if (agent, patient) != (p_agent, p_patient):
        raise ValueError(f"Active/passive role mismatch: {active} / {passive}")

    target_family = det_family(det_agent)
    if target_family != det_family(det_patient):
        raise ValueError(f"Unexpected mixed target determiner families: {active}")

    agent_modifier_det = (
        det_for_family(agent_modifier, target_family, known_indefinites)
        if agent_modifier
        else None
    )
    patient_modifier_det = (
        det_for_family(patient_modifier, target_family, known_indefinites)
        if patient_modifier
        else None
    )

    active_agent_np = maybe_complex_np(det_agent, agent, agent_prep, agent_modifier_det, agent_modifier)
    active_patient_np = maybe_complex_np(det_patient, patient, patient_prep, patient_modifier_det, patient_modifier)
    passive_patient_np = maybe_complex_np(p_det_patient, patient, patient_prep, patient_modifier_det, patient_modifier)
    passive_agent_np = maybe_complex_np(p_det_agent, agent, agent_prep, agent_modifier_det, agent_modifier)

    return (
        f"{active_agent_np} {verb} {active_patient_np} .",
        f"{passive_patient_np} {aux} {participle} by {passive_agent_np} .",
    )


def cell_from_target(active: str, passive: str) -> str:
    det_agent, _, _, _, _ = parse_active(active)
    _, _, aux, _, _, _ = parse_passive(passive)
    tense = {"is": "present", "was": "past"}.get(aux, "unknown")
    return f"{det_family(det_agent)}_{tense}"


def build_corpus(
    backbone: pd.DataFrame,
    *,
    modifier_pool: Sequence[str],
    known_indefinites: dict[str, str] | None,
    source_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item_index, base in backbone.iterrows():
        pa = str(base["pa"])
        pp = str(base["pp"])
        ta = str(base["ta"])
        tp = str(base["tp"])
        pa_det_a, pa_agent, pa_verb, pa_det_p, pa_patient = parse_active(pa)
        ta_det_a, ta_agent, ta_verb, ta_det_p, ta_patient = parse_active(ta)
        _, _, pp_aux, _, _, _ = parse_passive(pp)
        _, _, tp_aux, _, _, _ = parse_passive(tp)

        base_used = {pa_agent, pa_patient, ta_agent, ta_patient}
        for complexity_index, complexity_condition in enumerate(COMPLEXITY_CONDITIONS):
            needed = 2 if complexity_condition == "both_complex" else 1
            modifiers = choose_distinct(
                modifier_pool,
                used=base_used,
                offset=item_index * 7 + complexity_index * 13,
                count=needed,
            )
            if complexity_condition == "agent_complex":
                agent_modifier, patient_modifier = modifiers[0], None
            elif complexity_condition == "patient_complex":
                agent_modifier, patient_modifier = None, modifiers[0]
            else:
                agent_modifier, patient_modifier = modifiers[0], modifiers[1]

            agent_prep, patient_prep = prepositions_for(
                item_index + complexity_index,
                complexity_condition,
            )
            target_active_complex, target_passive_complex = complex_targets(
                ta,
                tp,
                complexity_condition=complexity_condition,
                agent_modifier=agent_modifier,
                patient_modifier=patient_modifier,
                agent_prep=agent_prep,
                patient_prep=patient_prep,
                known_indefinites=known_indefinites,
            )
            rows.append(
                {
                    "item_id": f"exp4_{source_label}_{item_index:04d}_{complexity_condition}",
                    "item_index": int(item_index),
                    "complexity_condition": complexity_condition,
                    "source_label": source_label,
                    "target_cell": cell_from_target(ta, tp),
                    "prime_cell": cell_from_target(pa, pp),
                    "prime_active": pa,
                    "prime_passive": pp,
                    "target_active_simple": ta,
                    "target_passive_simple": tp,
                    "target_active_complex": target_active_complex,
                    "target_passive_complex": target_passive_complex,
                    "agent_modifier_noun": agent_modifier or "",
                    "patient_modifier_noun": patient_modifier or "",
                    "agent_modifier_prep": agent_prep or "",
                    "patient_modifier_prep": patient_prep or "",
                    "prime_active_det_family": det_family(pa_det_a),
                    "target_active_det_family": det_family(ta_det_a),
                    "prime_passive_aux": pp_aux,
                    "target_passive_aux": tp_aux,
                    "prime_active_verb_or_fragment": pa_verb,
                    "target_active_verb_or_fragment": ta_verb,
                    "target_agent": ta_agent,
                    "target_patient": ta_patient,
                }
            )
    return pd.DataFrame(rows)


def words(sentence: str) -> list[str]:
    return [part.strip(".,;:!?").lower() for part in str(sentence).split()]


def functional_words(sentence: str) -> set[str]:
    function_inventory = {
        "a",
        "an",
        "the",
        "is",
        "was",
        "by",
        "near",
        "above",
        "below",
        "on",
        "in",
        "from",
        "with",
    }
    return {word for word in words(sentence) if word in function_inventory}


def content_nouns_for_row(row: pd.Series) -> set[str]:
    fields = [
        "target_agent",
        "target_patient",
        "agent_modifier_noun",
        "patient_modifier_noun",
    ]
    return {str(row[field]) for field in fields if str(row[field]).strip()}


def audit(frame: pd.DataFrame) -> dict[str, object]:
    violations = Counter()
    by_overlap_rows = 0
    avoidable_function_overlap_rows = 0
    semantic_prep_overlap_rows = 0
    for row in frame.itertuples(index=False):
        prime_active_words = set(words(row.prime_active))
        prime_passive_words = set(words(row.prime_passive))
        target_active_words = set(words(row.target_active_complex))
        target_passive_words = set(words(row.target_passive_complex))

        target_modifiers = {
            value
            for value in (row.agent_modifier_noun, row.patient_modifier_noun)
            if str(value).strip()
        }
        prime_nouns = set(parse_active(row.prime_active)[1::3])
        target_base_nouns = {row.target_agent, row.target_patient}
        if prime_nouns & (target_base_nouns | target_modifiers):
            violations["prime_target_content_noun_overlap_rows"] += 1
        if row.agent_modifier_noun and row.agent_modifier_noun in target_base_nouns:
            violations["agent_modifier_reuses_target_base_noun_rows"] += 1
        if row.patient_modifier_noun and row.patient_modifier_noun in target_base_nouns:
            violations["patient_modifier_reuses_target_base_noun_rows"] += 1
        if row.agent_modifier_noun and row.agent_modifier_noun == row.patient_modifier_noun:
            violations["same_agent_patient_modifier_noun_rows"] += 1
        if row.prime_passive_aux == row.target_passive_aux:
            violations["same_passive_aux_rows"] += 1
        if row.prime_active_det_family == row.target_active_det_family:
            violations["same_det_family_rows"] += 1
        if row.prime_active_verb_or_fragment == row.target_active_verb_or_fragment:
            violations["same_active_verb_or_fragment_rows"] += 1

        prime_active_function = functional_words(row.prime_active)
        prime_passive_function = functional_words(row.prime_passive)
        target_active_function = functional_words(row.target_active_complex)
        target_passive_function = functional_words(row.target_passive_complex)

        if prime_active_function & target_active_function:
            avoidable_function_overlap_rows += 1
        passive_overlap = prime_passive_function & target_passive_function
        if passive_overlap == {"by"}:
            by_overlap_rows += 1
        elif passive_overlap:
            avoidable_function_overlap_rows += 1

        preps = [row.agent_modifier_prep, row.patient_modifier_prep]
        preps = [prep for prep in preps if str(prep).strip()]
        if len(preps) != len(set(preps)):
            violations["same_target_modifier_prep_rows"] += 1
        if len(preps) == 2 and SEMANTIC_PREP_GROUP[preps[0]] == SEMANTIC_PREP_GROUP[preps[1]]:
            semantic_prep_overlap_rows += 1

        if row.agent_modifier_prep == "by" or row.patient_modifier_prep == "by":
            violations["target_pp_uses_passive_by_rows"] += 1

    counts_by_complexity = frame["complexity_condition"].value_counts().sort_index().to_dict()
    counts_by_target_cell = frame.groupby(["complexity_condition", "target_cell"]).size().to_dict()
    return {
        "row_count": int(len(frame)),
        "counts_by_complexity": {str(k): int(v) for k, v in counts_by_complexity.items()},
        "counts_by_target_cell": {f"{k[0]}::{k[1]}": int(v) for k, v in counts_by_target_cell.items()},
        "violations": {str(k): int(v) for k, v in sorted(violations.items())},
        "unavoidable_passive_by_overlap_rows": int(by_overlap_rows),
        "avoidable_function_overlap_rows": int(avoidable_function_overlap_rows),
        "semantic_prep_overlap_rows": int(semantic_prep_overlap_rows),
    }


def write_outputs(core: pd.DataFrame, jabber: pd.DataFrame, summary: dict[str, object], args: argparse.Namespace) -> None:
    args.core_output.parent.mkdir(parents=True, exist_ok=True)
    args.jabber_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    core.to_csv(args.core_output, index=False)
    jabber.to_csv(args.jabber_output, index=False)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    core_backbone = read_backbone(args.core_csv)
    jabber_backbone = read_backbone(args.jabber_csv)
    if len(core_backbone) != len(jabber_backbone):
        raise ValueError("CORE and Jabberwocky backbones must have the same row count")

    human_nouns, indefinites = load_human_nouns(args.noun_list)
    jabber_nouns = load_jabber_nouns(args.jabber_lexicon)

    core = build_corpus(
        core_backbone,
        modifier_pool=human_nouns,
        known_indefinites=indefinites,
        source_label="core",
    )
    jabber = build_corpus(
        jabber_backbone,
        modifier_pool=jabber_nouns,
        known_indefinites=None,
        source_label="jabberwocky",
    )

    summary = {
        "description": (
            "Experiment 4 complex-NP role-recovery draft. Primes reuse the strict 1b "
            "simple-sentence backbone; targets add PP-complex NPs under agent_complex, "
            "patient_complex, and both_complex conditions. CORE is built first and "
            "Jabberwocky mirrors the same row/complexity/preposition plan."
        ),
        "core_source": portable_path(args.core_csv),
        "jabberwocky_source": portable_path(args.jabber_csv),
        "constraints": [
            "Prime sentences are the existing strict 1b simple pa/pp sentences.",
            "Targets are complex variants of the existing strict ta/tp target sentences.",
            "Complexity is counterbalanced across agent_complex, patient_complex, and both_complex.",
            "CORE PP modifier nouns are person nouns from the tracked USF noun vocabulary.",
            "Prime and target keep opposite determiner family and opposite passive auxiliary/tense.",
            "Prime and target keep no content noun or active verb/fragment overlap.",
            "Target PP prepositions never use the passive by-marker.",
            "Both-complex targets use two different PP prepositions from different semantic groups.",
            "Passive prime and passive target unavoidably share the English by-marker; this is audited separately.",
        ],
        "outputs": {
            "core": portable_path(args.core_output),
            "jabberwocky": portable_path(args.jabber_output),
        },
        "core_audit": audit(core),
        "jabberwocky_audit": audit(jabber),
    }
    write_outputs(core, jabber, summary, args)
    print(f"Wrote CORE: {args.core_output}")
    print(f"Wrote Jabberwocky: {args.jabber_output}")
    print(f"Wrote summary: {args.summary_json}")
    print(json.dumps({"core_audit": summary["core_audit"], "jabberwocky_audit": summary["jabberwocky_audit"]}, indent=2))


if __name__ == "__main__":
    main()
