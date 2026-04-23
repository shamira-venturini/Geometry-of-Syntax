import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_constrained_counterbalanced.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_constrained_counterbalanced_lexically_controlled.csv"
)
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "corpora"
    / "transitive"
    / "CORE_transitive_constrained_counterbalanced_lexically_controlled_summary.json"
)
DEFAULT_VERB_LIST = REPO_ROOT / "PrimeLM" / "vocabulary_lists" / "verblist_T_usf_freq.csv"
DEFAULT_ASSOCIATION_CSV = REPO_ROOT / "corpora" / "transitive" / "usf_association_edges_core_vocab.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a lexically controlled variant of the constrained counterbalanced CORE corpus "
            "where each target row is paired with a prime row that satisfies strict Sinclair-style controls: "
            "no shared nouns/verbs, opposite determiner family, opposite passive auxiliary, "
            "and no USF semantic association between prime and target content words."
        )
    )
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--verb-list", type=Path, default=DEFAULT_VERB_LIST)
    parser.add_argument("--association-csv", type=Path, default=DEFAULT_ASSOCIATION_CSV)
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Seed used to permute candidate prime rows before bipartite matching.",
    )
    return parser.parse_args()


def parse_active(sentence: str) -> Tuple[str, str, str, str, str]:
    tokens = sentence.strip().split()
    if len(tokens) != 6:
        raise ValueError(f"Unexpected active sentence format: {sentence}")
    return tokens[0].lower(), tokens[1].lower(), tokens[2].lower(), tokens[3].lower(), tokens[4].lower()


def parse_passive_aux(sentence: str) -> str:
    tokens = sentence.strip().split()
    if len(tokens) != 8:
        raise ValueError(f"Unexpected passive sentence format: {sentence}")
    return tokens[2].lower()


def det_family(det: str) -> str:
    token = det.lower().strip()
    if token == "the":
        return "def"
    if token in {"a", "an"}:
        return "indef"
    return "unknown"


def load_verb_maps(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        form_to_lemma: Dict[str, str] = {}
        form_to_tense: Dict[str, str] = {}
        for row in reader:
            clean = {(k or "").strip().lstrip("\ufeff"): (v or "").strip() for k, v in row.items()}
            lemma = clean["V"].lower()
            pres_3s = clean["pres_3s"].lower()
            past_a = clean["past_A"].lower()
            form_to_lemma[pres_3s] = lemma
            form_to_lemma[past_a] = lemma
            form_to_lemma[clean["past_P"].lower()] = lemma
            form_to_tense[pres_3s] = "present"
            form_to_tense[past_a] = "past"
        return form_to_lemma, form_to_tense


def load_semantic_edges(path: Path) -> Dict[str, Set[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"USF association CSV not found: {path}. "
            "Run scripts/33_build_usf_association_edges.py to build it."
        )

    frame = pd.read_csv(path)
    required = {"cue", "target"}
    missing = sorted(required.difference({str(c).strip().lower() for c in frame.columns}))
    if missing:
        raise ValueError(f"Association CSV is missing required columns: {missing}")

    # Normalize columns defensively.
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    edges: Dict[str, Set[str]] = {}
    for row in frame.itertuples(index=False):
        cue = str(getattr(row, "cue")).strip().lower()
        target = str(getattr(row, "target")).strip().lower()
        if not cue or not target:
            continue
        edges.setdefault(cue, set()).add(target)
        edges.setdefault(target, set()).add(cue)
    return edges


def lexical_signature(
    active_sentence: str,
    passive_sentence: str,
    form_to_lemma: Dict[str, str],
) -> Dict[str, object]:
    det_1, noun_1, verb_form, det_2, noun_2 = parse_active(active_sentence)
    lemma = form_to_lemma.get(verb_form, verb_form)
    signature = {
        "verb_lemma": lemma,
        "nouns": frozenset({noun_1, noun_2}),
        "passive_aux": parse_passive_aux(passive_sentence),
        "det_family": det_family(det_1),
        "content_words": (noun_1, noun_2, lemma),
    }
    return signature


def build_signatures(
    frame: pd.DataFrame,
    form_to_lemma: Dict[str, str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    target_signatures = [
        lexical_signature(str(row.ta), str(row.tp), form_to_lemma=form_to_lemma)
        for row in frame.itertuples(index=False)
    ]
    prime_signatures = [
        lexical_signature(str(row.pa), str(row.pp), form_to_lemma=form_to_lemma)
        for row in frame.itertuples(index=False)
    ]
    return target_signatures, prime_signatures


def is_semantically_associated(
    target_sig: Dict[str, object],
    prime_sig: Dict[str, object],
    semantic_edges: Dict[str, Set[str]],
) -> bool:
    target_words = list(target_sig["content_words"])
    prime_words = list(prime_sig["content_words"])
    for target_word in target_words:
        neighbors = semantic_edges.get(str(target_word), set())
        for prime_word in prime_words:
            if str(prime_word) in neighbors:
                return True
    return False


def build_assignment(
    target_signatures: Sequence[Dict[str, object]],
    prime_signatures: Sequence[Dict[str, object]],
    semantic_edges: Dict[str, Set[str]],
    seed: int,
) -> np.ndarray:
    n_rows = len(target_signatures)
    rng = np.random.default_rng(seed)
    prime_order = rng.permutation(n_rows)

    row_indices: List[int] = []
    col_indices: List[int] = []
    for target_index, target_sig in enumerate(target_signatures):
        target_verb = str(target_sig["verb_lemma"])
        target_nouns = set(target_sig["nouns"])
        for permuted_col, prime_index in enumerate(prime_order):
            prime_sig = prime_signatures[int(prime_index)]

            if target_verb == str(prime_sig["verb_lemma"]):
                continue
            if target_nouns & set(prime_sig["nouns"]):
                continue
            if str(target_sig["passive_aux"]) == str(prime_sig["passive_aux"]):
                continue
            if str(target_sig["det_family"]) == str(prime_sig["det_family"]):
                continue
            if is_semantically_associated(target_sig=target_sig, prime_sig=prime_sig, semantic_edges=semantic_edges):
                continue

            row_indices.append(target_index)
            col_indices.append(permuted_col)

    graph = csr_matrix(
        (np.ones(len(row_indices), dtype=np.int8), (row_indices, col_indices)),
        shape=(n_rows, n_rows),
    )
    matched_columns = maximum_bipartite_matching(graph, perm_type="column")
    if int(np.sum(matched_columns != -1)) != n_rows:
        raise RuntimeError(
            "Failed to find a full strict-control prime assignment. "
            "Try changing --seed or inspect semantic-edge strictness."
        )
    return prime_order[matched_columns]


def build_output_frame(source_frame: pd.DataFrame, assignment: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for target_index, prime_index in enumerate(assignment):
        target_row = source_frame.iloc[int(target_index)]
        prime_row = source_frame.iloc[int(prime_index)]
        rows.append(
            {
                "pa": str(prime_row["pa"]),
                "pp": str(prime_row["pp"]),
                "ta": str(target_row["ta"]),
                "tp": str(target_row["tp"]),
            }
        )
    return pd.DataFrame(rows, columns=["pa", "pp", "ta", "tp"])


def audit_assignment(
    source_frame: pd.DataFrame,
    assignment: np.ndarray,
    target_signatures: Sequence[Dict[str, object]],
    prime_signatures: Sequence[Dict[str, object]],
    semantic_edges: Dict[str, Set[str]],
    form_to_tense: Dict[str, str],
) -> Dict[str, object]:
    same_verb_rows = 0
    shared_noun_rows = 0
    same_aux_rows = 0
    same_det_family_rows = 0
    semantic_association_rows = 0
    prime_tense_mismatch_rows = 0
    target_tense_mismatch_rows = 0
    unknown_tense_rows = 0
    prime_source_rows: List[int] = []

    for target_index, prime_index in enumerate(assignment):
        prime_source_rows.append(int(prime_index))
        target_sig = target_signatures[target_index]
        prime_sig = prime_signatures[int(prime_index)]

        same_verb_rows += int(str(target_sig["verb_lemma"]) == str(prime_sig["verb_lemma"]))
        shared_noun_rows += int(bool(set(target_sig["nouns"]) & set(prime_sig["nouns"])))
        same_aux_rows += int(str(target_sig["passive_aux"]) == str(prime_sig["passive_aux"]))
        same_det_family_rows += int(str(target_sig["det_family"]) == str(prime_sig["det_family"]))
        semantic_association_rows += int(
            is_semantically_associated(target_sig=target_sig, prime_sig=prime_sig, semantic_edges=semantic_edges)
        )

        row = source_frame.iloc[int(target_index)]
        prime_row = source_frame.iloc[int(prime_index)]

        # Prime side tense audit.
        prime_active_tokens = str(prime_row["pa"]).strip().split()
        prime_passive_tokens = str(prime_row["pp"]).strip().split()
        prime_active_form = prime_active_tokens[2].lower()
        prime_aux = prime_passive_tokens[2].lower()
        prime_expected_tense = "present" if prime_aux == "is" else "past" if prime_aux == "was" else "unknown"
        prime_actual_tense = form_to_tense.get(prime_active_form, "unknown")
        if prime_expected_tense == "unknown" or prime_actual_tense == "unknown":
            unknown_tense_rows += 1
        prime_tense_mismatch_rows += int(prime_expected_tense != prime_actual_tense)

        # Target side tense audit.
        target_active_tokens = str(row["ta"]).strip().split()
        target_passive_tokens = str(row["tp"]).strip().split()
        target_active_form = target_active_tokens[2].lower()
        target_aux = target_passive_tokens[2].lower()
        target_expected_tense = "present" if target_aux == "is" else "past" if target_aux == "was" else "unknown"
        target_actual_tense = form_to_tense.get(target_active_form, "unknown")
        if target_expected_tense == "unknown" or target_actual_tense == "unknown":
            unknown_tense_rows += 1
        target_tense_mismatch_rows += int(target_expected_tense != target_actual_tense)

    total = len(source_frame)
    return {
        "row_count": int(total),
        "prime_source_row_indices_preview": prime_source_rows[:25],
        "same_verb_rows": int(same_verb_rows),
        "shared_noun_rows": int(shared_noun_rows),
        "shared_aux_rows": int(same_aux_rows),
        "same_det_family_rows": int(same_det_family_rows),
        "semantic_association_rows": int(semantic_association_rows),
        "prime_tense_mismatch_rows": int(prime_tense_mismatch_rows),
        "target_tense_mismatch_rows": int(target_tense_mismatch_rows),
        "unknown_tense_rows": int(unknown_tense_rows),
        "same_verb_rate": float(same_verb_rows / total),
        "shared_noun_rate": float(shared_noun_rows / total),
        "shared_aux_rate": float(same_aux_rows / total),
        "same_det_family_rate": float(same_det_family_rows / total),
        "semantic_association_rate": float(semantic_association_rows / total),
    }


def main() -> None:
    args = parse_args()
    source_csv = args.source_csv.resolve()
    source_frame = pd.read_csv(source_csv)
    source_frame.columns = [str(column).strip().lower() for column in source_frame.columns]

    required = {"pa", "pp", "ta", "tp"}
    missing = sorted(required.difference(source_frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in source CSV: {missing}")

    source_frame = source_frame[["pa", "pp", "ta", "tp"]]

    form_to_lemma, form_to_tense = load_verb_maps(args.verb_list.resolve())
    semantic_edges = load_semantic_edges(args.association_csv.resolve())

    target_signatures, prime_signatures = build_signatures(source_frame, form_to_lemma=form_to_lemma)
    assignment = build_assignment(
        target_signatures=target_signatures,
        prime_signatures=prime_signatures,
        semantic_edges=semantic_edges,
        seed=args.seed,
    )

    output_frame = build_output_frame(source_frame=source_frame, assignment=assignment)
    audit = audit_assignment(
        source_frame=source_frame,
        assignment=assignment,
        target_signatures=target_signatures,
        prime_signatures=prime_signatures,
        semantic_edges=semantic_edges,
        form_to_tense=form_to_tense,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(args.output_csv, index=False)

    summary = {
        "source_csv": str(source_csv),
        "output_csv": str(args.output_csv.resolve()),
        "summary_json": str(args.summary_json.resolve()),
        "verb_list": str(args.verb_list.resolve()),
        "association_csv": str(args.association_csv.resolve()),
        "seed": int(args.seed),
        "constraints": [
            "Prime and target must not share the active verb lemma.",
            "Prime and target must not share either noun.",
            "Prime and target passive auxiliaries must be opposite (is/was mismatch).",
            "Prime and target determiner families must be opposite (a/an vs the).",
            "Prime and target content words must not be connected in USF free association norms.",
            "Within each side, active tense must match passive auxiliary (is->present, was->past).",
        ],
        "audit": audit,
        "notes": [
            "Each output row keeps the original target ta/tp and replaces the prime pa/pp with a matched row from the source corpus.",
            "This file is the strict Sinclair-style corpus used for Experiment 1b, 2, and 3 core conditions.",
        ],
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if int(audit["same_verb_rows"]) != 0:
        raise RuntimeError("Audit failed: found rows with shared verb lemma.")
    if int(audit["shared_noun_rows"]) != 0:
        raise RuntimeError("Audit failed: found rows with shared nouns.")
    if int(audit["shared_aux_rows"]) != 0:
        raise RuntimeError("Audit failed: found rows with shared passive auxiliary.")
    if int(audit["same_det_family_rows"]) != 0:
        raise RuntimeError("Audit failed: found rows with shared determiner family.")
    if int(audit["semantic_association_rows"]) != 0:
        raise RuntimeError("Audit failed: found rows with USF semantic associations.")
    if int(audit["prime_tense_mismatch_rows"]) != 0 or int(audit["target_tense_mismatch_rows"]) != 0:
        raise RuntimeError("Audit failed: found tense mismatches between active and passive forms.")
    if int(audit["unknown_tense_rows"]) != 0:
        raise RuntimeError("Audit failed: unknown tense forms detected.")

    print(f"Saved lexically controlled corpus to {args.output_csv}")
    print(f"Summary written to {args.summary_json}")


if __name__ == "__main__":
    main()
