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
            "every eligible noun appears equally often as AGENT and PATIENT."
        )
    )
    parser.add_argument("--noun-list", type=Path, default=DEFAULT_NOUNS)
    parser.add_argument("--verb-list", type=Path, default=DEFAULT_VERBS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle final rows after construction.")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def load_nouns(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        rows = []
        for row in reader:
            clean = {(k or "").strip().lstrip("\ufeff"): (v or "").strip() for k, v in row.items()}
            rows.append(clean)
    return rows


def load_verbs(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        rows = []
        for row in reader:
            clean = {(k or "").strip().lstrip("\ufeff"): (v or "").strip() for k, v in row.items()}
            rows.append(clean)
    return rows


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

            # Keep auxiliary overlap controlled within each row:
            # prime/passive and target/passive always use opposite auxiliaries.
            if i % 2 == 0:
                p_aux, t_aux = "is", "was"
            else:
                p_aux, t_aux = "was", "is"

            d_pa = determiners[p_agent]
            d_pp = determiners[p_patient]
            d_ta = determiners[t_agent]
            d_tp = determiners[t_patient]

            pa = sentence_active(d_pa, p_agent, verb["pres_3s"], d_pp, p_patient)
            pp = sentence_passive(d_pa, p_agent, verb["past_P"], d_pp, p_patient, p_aux)
            ta = sentence_active(d_ta, t_agent, verb["pres_3s"], d_tp, t_patient)
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


def parse_active(sentence: str) -> Tuple[str, str, str]:
    tokens = sentence.strip().split()
    if len(tokens) != 6:
        raise ValueError(f"Unexpected active template: {sentence}")
    return tokens[1], tokens[2], tokens[4]


def parse_passive(sentence: str) -> Tuple[str, str, str]:
    tokens = sentence.strip().split()
    if len(tokens) != 8:
        raise ValueError(f"Unexpected passive template: {sentence}")
    return tokens[6], tokens[3], tokens[1]


def audit_balance(rows: Sequence[Sequence[str]]) -> Dict[str, object]:
    counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    same_aux_rows = 0

    for row in rows:
        pa, pp, ta, tp = row
        pp_tokens = pp.strip().split()
        tp_tokens = tp.strip().split()
        if len(pp_tokens) != 8 or len(tp_tokens) != 8:
            raise ValueError(f"Unexpected passive template in row: {row}")
        same_aux_rows += int(pp_tokens[2].lower() == tp_tokens[2].lower())

        for sentence, parser in ((pa, parse_active), (pp, parse_passive), (ta, parse_active), (tp, parse_passive)):
            agent, verb, patient = parser(sentence)
            counts[(verb, agent)]["agent"] += 1
            counts[(verb, patient)]["patient"] += 1

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
        "tracked_verb_noun_pairs": len(counts),
        "imbalanced_pairs": imbalanced,
        "same_aux_rows": int(same_aux_rows),
        "same_aux_rate": float(same_aux_rows / len(rows)) if rows else 0.0,
    }


def main() -> None:
    args = parse_args()

    noun_rows = load_nouns(args.noun_list)
    verb_rows = load_verbs(args.verb_list)
    by_category, by_main, determiners = build_category_maps(noun_rows)

    rows, included, excluded = generate_rows(verb_rows, by_category, by_main, determiners)

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pa", "pp", "ta", "tp"])
        writer.writerows(rows)

    audit = audit_balance(rows)

    summary = {
        "noun_source": str(args.noun_list),
        "verb_source": str(args.verb_list),
        "output_csv": str(args.output),
        "row_count": len(rows),
        "included_verbs": included,
        "excluded_verbs": excluded,
        "audit": audit,
        "notes": [
            "Each included verb contributes one reduced block where every eligible noun appears equally often as AGENT and PATIENT.",
            "Within each row, passive prime (`pp`) and passive target (`tp`) use opposite auxiliaries (`is` vs `was`).",
            "Excluded verbs have no noun set that can satisfy both N1 and N2 role constraints simultaneously.",
        ],
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if audit["imbalanced_pairs"]:
        raise RuntimeError("Balance audit failed: found verb-noun role imbalances.")
    if int(audit["same_aux_rows"]) != 0:
        raise RuntimeError("Auxiliary audit failed: found pp/tp rows with overlapping auxiliaries.")

    print(f"Saved {len(rows)} rows to {args.output}")
    print(f"Included verbs: {len(included)} | Excluded verbs: {len(excluded)}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
