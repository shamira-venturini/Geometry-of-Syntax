import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOUNS = REPO_ROOT / "PrimeLM" / "vocabulary_lists" / "nounlist_usf_freq.csv"
DEFAULT_VERBS = REPO_ROOT / "PrimeLM" / "vocabulary_lists" / "verblist_T_usf_freq.csv"
DEFAULT_OUTPUT = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_constrained_counterbalanced.csv"
DEFAULT_SUMMARY = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_constrained_counterbalanced_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reduced transitive corpus for constrained generation where, for each included verb, "
            "every eligible noun appears equally often as AGENT and PATIENT. "
            "Rows enforce Sinclair-style controls except lexical overlap removal, which is done in script 17."
        )
    )
    parser.add_argument("--noun-list", type=Path, default=DEFAULT_NOUNS)
    parser.add_argument("--verb-list", type=Path, default=DEFAULT_VERBS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle final rows after construction.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--max-rank",
        type=int,
        default=5000,
        help="Keep only nouns/verbs with F_rank <= max-rank (COCA frequency control).",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        rows = []
        for row in reader:
            clean = {(k or "").strip().lstrip("\ufeff"): (v or "").strip() for k, v in row.items()}
            rows.append(clean)
    return rows


def rank_value(row: Dict[str, str], key: str = "F_rank") -> int:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return 10**9
    try:
        return int(float(raw))
    except ValueError:
        return 10**9


def filter_rows_by_rank(rows: Sequence[Dict[str, str]], max_rank: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    kept: List[Dict[str, str]] = []
    dropped: List[Dict[str, str]] = []
    for row in rows:
        if rank_value(row) <= max_rank:
            kept.append(dict(row))
        else:
            dropped.append(dict(row))
    return kept, dropped


def build_category_maps(noun_rows: Sequence[Dict[str, str]]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, str]]:
    by_category: Dict[str, Set[str]] = defaultdict(set)
    by_main: Dict[str, Set[str]] = defaultdict(set)
    determiners: Dict[str, str] = {}

    for row in noun_rows:
        noun = row["nouns"]
        by_category[row["category"].lower()].add(noun)
        by_main[row["cat_main"].lower()].add(noun)
        determiners[noun] = row["det_a"].lower()

    return by_category, by_main, determiners


def allowed_nouns(label: str, by_category: Dict[str, Set[str]], by_main: Dict[str, Set[str]]) -> Set[str]:
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


def sentence_active(det_agent: str, agent: str, active_verb: str, det_patient: str, patient: str) -> str:
    return f"{det_agent} {agent} {active_verb} {det_patient} {patient} ."


def sentence_passive(
    det_agent: str,
    agent: str,
    passive_verb: str,
    det_patient: str,
    patient: str,
    auxiliary: str,
) -> str:
    return f"{det_patient} {patient} {auxiliary} {passive_verb} by {det_agent} {agent} ."


def indefinite_or_definite(det_indef: str, mode: str) -> str:
    if mode == "indef":
        return det_indef
    if mode == "def":
        return "the"
    raise ValueError(f"Unknown determiner mode: {mode}")


def generate_rows(
    verbs: Sequence[Dict[str, str]],
    by_category: Dict[str, Set[str]],
    by_main: Dict[str, Set[str]],
    determiners: Dict[str, str],
) -> Tuple[List[List[str]], List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[List[str]] = []
    included: List[Dict[str, object]] = []
    excluded: List[Dict[str, object]] = []

    for verb in verbs:
        v = verb["V"]
        n1_pool = allowed_nouns(verb["N1"], by_category, by_main)
        n2_pool = allowed_nouns(verb["N2"], by_category, by_main)
        reversible = sorted(n1_pool & n2_pool)

        if len(reversible) < 2:
            excluded.append(
                {
                    "verb": v,
                    "reason": "No reversible noun set with at least two nouns under N1/N2 constraints.",
                    "N1": verb["N1"],
                    "N2": verb["N2"],
                    "eligible_reversible_nouns": len(reversible),
                }
            )
            continue

        n = len(reversible)
        prime_agent_shift = 0
        prime_patient_shift = 1
        target_agent_shift = n // 2
        target_patient_shift = (target_agent_shift + 1) % n

        for i in range(n):
            p_agent = reversible[(i + prime_agent_shift) % n]
            p_patient = reversible[(i + prime_patient_shift) % n]
            t_agent = reversible[(i + target_agent_shift) % n]
            t_patient = reversible[(i + target_patient_shift) % n]

            # Passive auxiliaries are always opposite within each row.
            if i % 2 == 0:
                p_aux, t_aux = "is", "was"
            else:
                p_aux, t_aux = "was", "is"

            # Determiner families are always opposite between prime and target.
            if i % 2 == 0:
                p_det_mode, t_det_mode = "indef", "def"
            else:
                p_det_mode, t_det_mode = "def", "indef"

            d_pa = indefinite_or_definite(determiners[p_agent], p_det_mode)
            d_pp = indefinite_or_definite(determiners[p_patient], p_det_mode)
            d_ta = indefinite_or_definite(determiners[t_agent], t_det_mode)
            d_tp = indefinite_or_definite(determiners[t_patient], t_det_mode)

            # Active tense must match passive auxiliary to preserve meaning.
            p_active_form = verb["pres_3s"] if p_aux == "is" else verb["past_A"]
            t_active_form = verb["pres_3s"] if t_aux == "is" else verb["past_A"]

            pa = sentence_active(d_pa, p_agent, p_active_form, d_pp, p_patient)
            pp = sentence_passive(d_pa, p_agent, verb["past_P"], d_pp, p_patient, p_aux)
            ta = sentence_active(d_ta, t_agent, t_active_form, d_tp, t_patient)
            tp = sentence_passive(d_ta, t_agent, verb["past_P"], d_tp, t_patient, t_aux)
            rows.append([pa, pp, ta, tp])

        included.append(
            {
                "verb": v,
                "N1": verb["N1"],
                "N2": verb["N2"],
                "eligible_reversible_nouns": n,
                "rows_added": n,
            }
        )

    return rows, included, excluded


def parse_active(sentence: str) -> Tuple[str, str, str, str, str]:
    tokens = sentence.strip().split()
    if len(tokens) != 6:
        raise ValueError(f"Unexpected active template: {sentence}")
    return tokens[0].lower(), tokens[1].lower(), tokens[2].lower(), tokens[3].lower(), tokens[4].lower()


def parse_passive(sentence: str) -> Tuple[str, str, str, str, str, str]:
    tokens = sentence.strip().split()
    if len(tokens) != 8:
        raise ValueError(f"Unexpected passive template: {sentence}")
    return (
        tokens[0].lower(),
        tokens[1].lower(),
        tokens[2].lower(),
        tokens[3].lower(),
        tokens[5].lower(),
        tokens[6].lower(),
    )


def det_family(det: str) -> str:
    token = det.lower().strip()
    if token == "the":
        return "def"
    if token in {"a", "an"}:
        return "indef"
    return "unknown"


def audit_balance(
    rows: Sequence[Sequence[str]],
    present_forms: Set[str],
    past_forms: Set[str],
) -> Dict[str, object]:
    counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    same_aux_rows = 0
    shared_noun_rows = 0
    same_det_family_rows = 0
    prime_tense_mismatch_rows = 0
    target_tense_mismatch_rows = 0
    unknown_det_rows = 0
    unknown_tense_rows = 0

    for row in rows:
        pa, pp, ta, tp = row

        pa_det_a, pa_agent, pa_verb, pa_det_p, pa_patient = parse_active(pa)
        ta_det_a, ta_agent, ta_verb, ta_det_p, ta_patient = parse_active(ta)
        pp_det_p, pp_patient, pp_aux, pp_part, pp_det_a, pp_agent = parse_passive(pp)
        tp_det_p, tp_patient, tp_aux, tp_part, tp_det_a, tp_agent = parse_passive(tp)

        # Consistency across active/passive realization for each side.
        if (pa_agent, pa_patient) != (pp_agent, pp_patient):
            raise ValueError(f"Prime agent/patient mismatch between pa and pp: {row}")
        if (ta_agent, ta_patient) != (tp_agent, tp_patient):
            raise ValueError(f"Target agent/patient mismatch between ta and tp: {row}")
        if pp_part != tp_part:
            # Within script 11 each row is same verb family, so participle should match.
            raise ValueError(f"Unexpected participle mismatch within row: {row}")

        same_aux_rows += int(pp_aux == tp_aux)
        shared_noun_rows += int(bool({pa_agent, pa_patient} & {ta_agent, ta_patient}))

        pa_family = det_family(pa_det_a)
        ta_family = det_family(ta_det_a)
        if pa_family == "unknown" or ta_family == "unknown":
            unknown_det_rows += 1
        same_det_family_rows += int(pa_family == ta_family)

        prime_expected_tense = "present" if pp_aux == "is" else "past" if pp_aux == "was" else "unknown"
        target_expected_tense = "present" if tp_aux == "is" else "past" if tp_aux == "was" else "unknown"
        prime_actual_tense = "present" if pa_verb in present_forms else "past" if pa_verb in past_forms else "unknown"
        target_actual_tense = "present" if ta_verb in present_forms else "past" if ta_verb in past_forms else "unknown"

        if prime_expected_tense == "unknown" or target_expected_tense == "unknown":
            unknown_tense_rows += 1
        if prime_actual_tense == "unknown" or target_actual_tense == "unknown":
            unknown_tense_rows += 1

        prime_tense_mismatch_rows += int(prime_expected_tense != prime_actual_tense)
        target_tense_mismatch_rows += int(target_expected_tense != target_actual_tense)

        # Balance tracking (verb form + noun role counts).
        counts[(pa_verb, pa_agent)]["agent"] += 1
        counts[(pa_verb, pa_patient)]["patient"] += 1
        counts[(ta_verb, ta_agent)]["agent"] += 1
        counts[(ta_verb, ta_patient)]["patient"] += 1
        counts[(pp_part, pp_agent)]["agent"] += 1
        counts[(pp_part, pp_patient)]["patient"] += 1
        counts[(tp_part, tp_agent)]["agent"] += 1
        counts[(tp_part, tp_patient)]["patient"] += 1

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

    total = len(rows)
    return {
        "tracked_verb_noun_pairs": len(counts),
        "imbalanced_pairs": imbalanced,
        "same_aux_rows": int(same_aux_rows),
        "same_aux_rate": float(same_aux_rows / total) if total else 0.0,
        "shared_noun_rows": int(shared_noun_rows),
        "shared_noun_rate": float(shared_noun_rows / total) if total else 0.0,
        "same_det_family_rows": int(same_det_family_rows),
        "same_det_family_rate": float(same_det_family_rows / total) if total else 0.0,
        "prime_tense_mismatch_rows": int(prime_tense_mismatch_rows),
        "target_tense_mismatch_rows": int(target_tense_mismatch_rows),
        "unknown_det_rows": int(unknown_det_rows),
        "unknown_tense_rows": int(unknown_tense_rows),
    }


def main() -> None:
    args = parse_args()

    noun_rows = load_rows(args.noun_list)
    verb_rows = load_rows(args.verb_list)

    filtered_nouns, dropped_nouns = filter_rows_by_rank(noun_rows, max_rank=args.max_rank)
    filtered_verbs, dropped_verbs = filter_rows_by_rank(verb_rows, max_rank=args.max_rank)

    by_category, by_main, determiners = build_category_maps(filtered_nouns)

    rows, included, excluded = generate_rows(filtered_verbs, by_category, by_main, determiners)

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pa", "pp", "ta", "tp"])
        writer.writerows(rows)

    present_forms = {str(row["pres_3s"]).lower() for row in filtered_verbs}
    past_forms = {str(row["past_A"]).lower() for row in filtered_verbs}
    audit = audit_balance(rows, present_forms=present_forms, past_forms=past_forms)

    summary = {
        "noun_source": str(args.noun_list),
        "verb_source": str(args.verb_list),
        "output_csv": str(args.output),
        "row_count": len(rows),
        "max_rank": int(args.max_rank),
        "noun_rows_kept": len(filtered_nouns),
        "noun_rows_dropped": len(dropped_nouns),
        "verb_rows_kept": len(filtered_verbs),
        "verb_rows_dropped": len(dropped_verbs),
        "included_verbs": included,
        "excluded_verbs": excluded,
        "audit": audit,
        "notes": [
            "Each included verb contributes one reduced block where every eligible noun appears equally often as AGENT and PATIENT.",
            "Within each row, passive prime (`pp`) and passive target (`tp`) use opposite auxiliaries (`is` vs `was`).",
            "Within each row, prime and target determiner families are opposite (`a/an` vs `the`).",
            "Within each row, active tense matches the passive auxiliary (`is`->present, `was`->past).",
            "Excluded verbs have no noun set that can satisfy both N1 and N2 role constraints simultaneously under the frequency cap.",
        ],
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if audit["imbalanced_pairs"]:
        raise RuntimeError("Balance audit failed: found verb-noun role imbalances.")
    if int(audit["same_aux_rows"]) != 0:
        raise RuntimeError("Auxiliary audit failed: found pp/tp rows with overlapping auxiliaries.")
    if int(audit["shared_noun_rows"]) != 0:
        raise RuntimeError("Noun-overlap audit failed: found shared nouns between prime and target.")
    if int(audit["same_det_family_rows"]) != 0:
        raise RuntimeError("Determiner audit failed: found rows where prime and target share determiner family.")
    if int(audit["prime_tense_mismatch_rows"]) != 0 or int(audit["target_tense_mismatch_rows"]) != 0:
        raise RuntimeError("Tense audit failed: active/passive tense mismatch detected.")
    if int(audit["unknown_det_rows"]) != 0 or int(audit["unknown_tense_rows"]) != 0:
        raise RuntimeError("Template audit failed: unknown determiner or tense rows detected.")

    print(f"Saved {len(rows)} rows to {args.output}")
    print(f"Included verbs: {len(included)} | Excluded verbs: {len(excluded)}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
