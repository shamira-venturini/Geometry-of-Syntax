import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOUNS = REPO_ROOT / "PrimeLM" / "vocabulary_lists" / "nounlist_usf_freq.csv"
DEFAULT_VERBS = REPO_ROOT / "PrimeLM" / "vocabulary_lists" / "verblist_T_usf_freq.csv"
DEFAULT_ASSOCIATIONS = REPO_ROOT / "corpora" / "transitive" / "usf_association_edges_core_vocab.csv"
DEFAULT_OUTPUT = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_strict_4cell_counterbalanced.csv"
DEFAULT_SUMMARY = (
    REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_strict_4cell_counterbalanced_summary.json"
)

TARGET_CELLS: Sequence[Tuple[str, str]] = (
    ("def", "past"),
    ("def", "present"),
    ("indef", "past"),
    ("indef", "present"),
)


@dataclass(frozen=True)
class EventRow:
    active: str
    passive: str
    verb_lemma: str
    agent: str
    patient: str
    passive_aux: str
    det_family: str
    tense: str

    @property
    def nouns(self) -> Set[str]:
        return {self.agent, self.patient}

    @property
    def content_words(self) -> Tuple[str, str, str]:
        return self.agent, self.patient, self.verb_lemma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a stricter CORE transitive corpus with reversible verbs, exact "
            "target-side determiner-by-tense balance, lemma-level role balance, "
            "and one-to-one strict prime assignment."
        )
    )
    parser.add_argument("--noun-list", type=Path, default=DEFAULT_NOUNS)
    parser.add_argument("--verb-list", type=Path, default=DEFAULT_VERBS)
    parser.add_argument("--association-csv", type=Path, default=DEFAULT_ASSOCIATIONS)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-rank", type=int, default=5000)
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return [
            {(key or "").strip().lstrip("\ufeff"): (value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def rank_value(row: Mapping[str, str], key: str = "F_rank") -> int:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return 10**9
    try:
        return int(float(raw))
    except ValueError:
        return 10**9


def filter_rows_by_rank(
    rows: Sequence[Dict[str, str]],
    max_rank: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    kept: List[Dict[str, str]] = []
    dropped: List[Dict[str, str]] = []
    for row in rows:
        if rank_value(row) <= max_rank:
            kept.append(dict(row))
        else:
            dropped.append(dict(row))
    return kept, dropped


def build_category_maps(
    noun_rows: Sequence[Mapping[str, str]],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, str]]:
    by_category: Dict[str, Set[str]] = defaultdict(set)
    by_main: Dict[str, Set[str]] = defaultdict(set)
    determiners: Dict[str, str] = {}

    for row in noun_rows:
        noun = str(row["nouns"]).strip().lower()
        by_category[str(row["category"]).strip().lower()].add(noun)
        by_main[str(row["cat_main"]).strip().lower()].add(noun)
        determiners[noun] = str(row["det_a"]).strip().lower()

    return by_category, by_main, determiners


def allowed_nouns(label: str, by_category: Mapping[str, Set[str]], by_main: Mapping[str, Set[str]]) -> Set[str]:
    key = label.strip().lower()
    if key == "person":
        return set(by_category["person"])
    if key == "group":
        return set(by_category["group"])
    if key == "object":
        return set(by_category["object"])
    if key == "drinkable":
        return set(by_category["drinkable"])
    if key == "edible":
        return set(by_category["edible"])
    if key == "human":
        return set(by_main["human"])
    if key == "oed":
        return set(by_main["object"])
    if key == "po":
        return set(by_category["person"]) | set(by_category["object"])
    if key == "poed":
        return (
            set(by_category["person"])
            | set(by_category["object"])
            | set(by_category["edible"])
            | set(by_category["drinkable"])
        )
    raise ValueError(f"Unsupported role label: {label}")


def indefinite_or_definite(det_indef: str, mode: str) -> str:
    if mode == "def":
        return "the"
    if mode == "indef":
        return det_indef
    raise ValueError(f"Unknown determiner mode: {mode}")


def active_form(verb: Mapping[str, str], tense: str) -> str:
    if tense == "present":
        return str(verb["pres_3s"]).strip().lower()
    if tense == "past":
        return str(verb["past_A"]).strip().lower()
    raise ValueError(f"Unknown tense: {tense}")


def passive_aux(tense: str) -> str:
    if tense == "present":
        return "is"
    if tense == "past":
        return "was"
    raise ValueError(f"Unknown tense: {tense}")


def sentence_active(det_agent: str, agent: str, verb_form: str, det_patient: str, patient: str) -> str:
    return f"{det_agent} {agent} {verb_form} {det_patient} {patient} ."


def sentence_passive(
    det_agent: str,
    agent: str,
    passive_verb: str,
    det_patient: str,
    patient: str,
    auxiliary: str,
) -> str:
    return f"{det_patient} {patient} {auxiliary} {passive_verb} by {det_agent} {agent} ."


def build_event(
    *,
    verb: Mapping[str, str],
    agent: str,
    patient: str,
    det_family: str,
    tense: str,
    determiners: Mapping[str, str],
) -> EventRow:
    det_agent = indefinite_or_definite(determiners[agent], det_family)
    det_patient = indefinite_or_definite(determiners[patient], det_family)
    verb_lemma = str(verb["V"]).strip().lower()
    active = sentence_active(
        det_agent=det_agent,
        agent=agent,
        verb_form=active_form(verb, tense),
        det_patient=det_patient,
        patient=patient,
    )
    passive = sentence_passive(
        det_agent=det_agent,
        agent=agent,
        passive_verb=str(verb["past_P"]).strip().lower(),
        det_patient=det_patient,
        patient=patient,
        auxiliary=passive_aux(tense),
    )
    return EventRow(
        active=active,
        passive=passive,
        verb_lemma=verb_lemma,
        agent=agent,
        patient=patient,
        passive_aux=passive_aux(tense),
        det_family=det_family,
        tense=tense,
    )


def build_balanced_event_pool(
    verbs: Sequence[Mapping[str, str]],
    by_category: Mapping[str, Set[str]],
    by_main: Mapping[str, Set[str]],
    determiners: Mapping[str, str],
) -> Tuple[List[EventRow], List[Dict[str, object]], List[Dict[str, object]]]:
    events: List[EventRow] = []
    included: List[Dict[str, object]] = []
    excluded: List[Dict[str, object]] = []

    for verb in verbs:
        n1_pool = allowed_nouns(str(verb["N1"]), by_category, by_main)
        n2_pool = allowed_nouns(str(verb["N2"]), by_category, by_main)
        reversible = sorted(n1_pool & n2_pool)
        verb_lemma = str(verb["V"]).strip().lower()

        if len(reversible) < len(TARGET_CELLS):
            excluded.append(
                {
                    "verb": verb_lemma,
                    "reason": "Too few reversible nouns to cover all target determiner-by-tense cells.",
                    "N1": str(verb["N1"]),
                    "N2": str(verb["N2"]),
                    "eligible_reversible_nouns": len(reversible),
                }
            )
            continue
        if len(reversible) % len(TARGET_CELLS) != 0:
            excluded.append(
                {
                    "verb": verb_lemma,
                    "reason": "Reversible noun count is not divisible by the four target cells.",
                    "N1": str(verb["N1"]),
                    "N2": str(verb["N2"]),
                    "eligible_reversible_nouns": len(reversible),
                }
            )
            continue

        start_index = len(events)
        for index, agent in enumerate(reversible):
            patient = reversible[(index + 1) % len(reversible)]
            det_family, tense = TARGET_CELLS[index % len(TARGET_CELLS)]
            events.append(
                build_event(
                    verb=verb,
                    agent=agent,
                    patient=patient,
                    det_family=det_family,
                    tense=tense,
                    determiners=determiners,
                )
            )

        included.append(
            {
                "verb": verb_lemma,
                "N1": str(verb["N1"]),
                "N2": str(verb["N2"]),
                "eligible_reversible_nouns": len(reversible),
                "rows_added": len(events) - start_index,
                "rows_per_target_cell": int(len(reversible) / len(TARGET_CELLS)),
            }
        )

    return events, included, excluded


def load_semantic_edges(path: Path) -> Dict[str, Set[str]]:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = {"cue", "target"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Association CSV is missing required columns: {missing}")

    edges: Dict[str, Set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        cue = str(row.cue).strip().lower()
        target = str(row.target).strip().lower()
        if not cue or not target:
            continue
        edges[cue].add(target)
        edges[target].add(cue)
    return dict(edges)


def is_semantically_associated(target: EventRow, prime: EventRow, semantic_edges: Mapping[str, Set[str]]) -> bool:
    for target_word in target.content_words:
        neighbors = semantic_edges.get(str(target_word), set())
        for prime_word in prime.content_words:
            if str(prime_word) in neighbors:
                return True
    return False


def build_prime_assignment(
    events: Sequence[EventRow],
    semantic_edges: Mapping[str, Set[str]],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prime_order = rng.permutation(len(events))

    row_indices: List[int] = []
    col_indices: List[int] = []
    for target_index, target in enumerate(events):
        for permuted_col, prime_index in enumerate(prime_order):
            prime = events[int(prime_index)]
            if target.verb_lemma == prime.verb_lemma:
                continue
            if target.nouns & prime.nouns:
                continue
            if target.passive_aux == prime.passive_aux:
                continue
            if target.det_family == prime.det_family:
                continue
            if is_semantically_associated(target, prime, semantic_edges):
                continue
            row_indices.append(target_index)
            col_indices.append(permuted_col)

    graph = csr_matrix(
        (np.ones(len(row_indices), dtype=np.int8), (row_indices, col_indices)),
        shape=(len(events), len(events)),
    )
    matched_columns = maximum_bipartite_matching(graph, perm_type="column")
    if int(np.sum(matched_columns != -1)) != len(events):
        raise RuntimeError(
            "Failed to find a full strict prime assignment. "
            "Try changing --seed or loosening semantic association constraints."
        )
    return prime_order[matched_columns]


def cell_key(event: EventRow) -> str:
    return f"{event.det_family}_{event.tense}"


def role_balance(
    rows: Sequence[Tuple[EventRow, EventRow]],
    scope: str,
) -> Dict[str, object]:
    counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

    for prime, target in rows:
        selected: List[EventRow]
        if scope == "target_active_only":
            selected = [target]
        elif scope == "target_active_passive":
            selected = [target, target]
        elif scope == "full_pa_pp_ta_tp":
            selected = [prime, prime, target, target]
        else:
            raise ValueError(f"Unknown role-balance scope: {scope}")

        for event in selected:
            counts[(event.verb_lemma, event.agent)]["agent"] += 1
            counts[(event.verb_lemma, event.patient)]["patient"] += 1

    imbalanced = []
    for (verb, noun), role_counts in sorted(counts.items()):
        agent_count = int(role_counts["agent"])
        patient_count = int(role_counts["patient"])
        if agent_count != patient_count:
            imbalanced.append(
                {
                    "verb": verb,
                    "noun": noun,
                    "agent_count": agent_count,
                    "patient_count": patient_count,
                    "difference": agent_count - patient_count,
                }
            )

    return {
        "tracked_verb_noun_pairs": int(len(counts)),
        "imbalanced_pairs": imbalanced,
    }


def audit_rows(
    rows: Sequence[Tuple[EventRow, EventRow]],
    semantic_edges: Mapping[str, Set[str]],
) -> Dict[str, object]:
    same_verb_rows = 0
    shared_noun_rows = 0
    shared_aux_rows = 0
    same_det_family_rows = 0
    semantic_association_rows = 0

    target_cell_counts: Counter = Counter()
    prime_cell_counts: Counter = Counter()
    target_verb_counts: Counter = Counter()
    prime_verb_counts: Counter = Counter()

    for prime, target in rows:
        same_verb_rows += int(prime.verb_lemma == target.verb_lemma)
        shared_noun_rows += int(bool(prime.nouns & target.nouns))
        shared_aux_rows += int(prime.passive_aux == target.passive_aux)
        same_det_family_rows += int(prime.det_family == target.det_family)
        semantic_association_rows += int(is_semantically_associated(target, prime, semantic_edges))

        target_cell_counts[cell_key(target)] += 1
        prime_cell_counts[cell_key(prime)] += 1
        target_verb_counts[target.verb_lemma] += 1
        prime_verb_counts[prime.verb_lemma] += 1

    return {
        "row_count": int(len(rows)),
        "target_cell_counts": dict(sorted(target_cell_counts.items())),
        "prime_cell_counts": dict(sorted(prime_cell_counts.items())),
        "target_verb_count_range": [
            int(min(target_verb_counts.values())) if target_verb_counts else 0,
            int(max(target_verb_counts.values())) if target_verb_counts else 0,
        ],
        "prime_verb_count_range": [
            int(min(prime_verb_counts.values())) if prime_verb_counts else 0,
            int(max(prime_verb_counts.values())) if prime_verb_counts else 0,
        ],
        "same_verb_rows": int(same_verb_rows),
        "shared_noun_rows": int(shared_noun_rows),
        "shared_aux_rows": int(shared_aux_rows),
        "same_det_family_rows": int(same_det_family_rows),
        "semantic_association_rows": int(semantic_association_rows),
        "role_balance": {
            "target_active_only": role_balance(rows, "target_active_only"),
            "target_active_passive": role_balance(rows, "target_active_passive"),
            "full_pa_pp_ta_tp": role_balance(rows, "full_pa_pp_ta_tp"),
        },
    }


def assert_clean_audit(audit: Mapping[str, object]) -> None:
    for key in [
        "same_verb_rows",
        "shared_noun_rows",
        "shared_aux_rows",
        "same_det_family_rows",
        "semantic_association_rows",
    ]:
        if int(audit[key]) != 0:
            raise RuntimeError(f"Strict audit failed: {key}={audit[key]}")

    for scope, payload in dict(audit["role_balance"]).items():
        if payload["imbalanced_pairs"]:
            raise RuntimeError(f"Role-balance audit failed for {scope}.")

    cell_counts = dict(audit["target_cell_counts"])
    if len(set(cell_counts.values())) != 1:
        raise RuntimeError(f"Target cells are not exactly balanced: {cell_counts}")


def main() -> None:
    args = parse_args()

    noun_rows = load_rows(args.noun_list.resolve())
    verb_rows = load_rows(args.verb_list.resolve())
    filtered_nouns, dropped_nouns = filter_rows_by_rank(noun_rows, max_rank=args.max_rank)
    filtered_verbs, dropped_verbs = filter_rows_by_rank(verb_rows, max_rank=args.max_rank)

    by_category, by_main, determiners = build_category_maps(filtered_nouns)
    events, included, excluded = build_balanced_event_pool(
        verbs=filtered_verbs,
        by_category=by_category,
        by_main=by_main,
        determiners=determiners,
    )
    semantic_edges = load_semantic_edges(args.association_csv.resolve())
    assignment = build_prime_assignment(events=events, semantic_edges=semantic_edges, seed=args.seed)

    paired_rows = [(events[int(prime_index)], target) for target, prime_index in zip(events, assignment)]
    audit = audit_rows(rows=paired_rows, semantic_edges=semantic_edges)
    assert_clean_audit(audit)

    output_frame = pd.DataFrame(
        [
            {
                "pa": prime.active,
                "pp": prime.passive,
                "ta": target.active,
                "tp": target.passive,
            }
            for prime, target in paired_rows
        ],
        columns=["pa", "pp", "ta", "tp"],
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(args.output_csv, index=False)

    summary = {
        "noun_source": str(args.noun_list.resolve()),
        "verb_source": str(args.verb_list.resolve()),
        "association_csv": str(args.association_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "summary_json": str(args.summary_json.resolve()),
        "seed": int(args.seed),
        "max_rank": int(args.max_rank),
        "noun_rows_kept": int(len(filtered_nouns)),
        "noun_rows_dropped": int(len(dropped_nouns)),
        "verb_rows_kept": int(len(filtered_verbs)),
        "verb_rows_dropped": int(len(dropped_verbs)),
        "included_verbs": included,
        "excluded_verbs": excluded,
        "target_cells": [f"{det}_{tense}" for det, tense in TARGET_CELLS],
        "constraints": [
            "Only verbs whose N1 and N2 role constraints share a reversible noun pool are included.",
            "Each included verb contributes a target block with exact def/indef by past/present balance.",
            "For each included verb, every target noun appears once as agent and once as patient.",
            "Prime rows are assigned one-to-one from the same event pool.",
            "Prime and target share no verb lemma and no nouns.",
            "Prime and target passive auxiliaries are opposite.",
            "Prime and target determiner families are opposite.",
            "Prime and target content words are not connected in USF free association norms.",
        ],
        "audit": audit,
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved stricter CORE corpus to {args.output_csv}")
    print(f"Summary written to {args.summary_json}")


if __name__ == "__main__":
    main()
