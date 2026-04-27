import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from production_priming_common import REPO_ROOT, lexical_overlap_audit, normalize_transitive_frame


STRICT_CORE = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_strict_4cell_counterbalanced.csv"
JABBERWOCKY_POOL = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_gpt2_monosyllabic_strict_4cell.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_core_targets_jabberwocky_primes_2048.csv"
)
DEFAULT_SUMMARY = (
    REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_core_targets_jabberwocky_primes_2048_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic mixed corpus with CORE targets and Jabberwocky primes. "
            "Each row enforces passive auxiliary mismatch and determiner-family mismatch "
            "between prime and target."
        )
    )
    parser.add_argument("--core-csv", type=Path, default=STRICT_CORE)
    parser.add_argument("--jabberwocky-csv", type=Path, default=JABBERWOCKY_POOL)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--n-items", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def _parse_active(sentence: str) -> Tuple[str, str, str, str, str]:
    tokens = str(sentence).strip().split()
    if len(tokens) != 6:
        raise ValueError(f"Unexpected active sentence format: {sentence}")
    return tokens[0].lower(), tokens[1].lower(), tokens[2].lower(), tokens[3].lower(), tokens[4].lower()


def _parse_passive(sentence: str) -> Tuple[str, str, str, str, str, str]:
    tokens = str(sentence).strip().split()
    if len(tokens) != 8:
        raise ValueError(f"Unexpected passive sentence format: {sentence}")
    return tokens[0].lower(), tokens[1].lower(), tokens[2].lower(), tokens[3].lower(), tokens[5].lower(), tokens[6].lower()


def _passive_aux(sentence: str) -> str:
    return _parse_passive(sentence)[2]


def _det_family_from_active(sentence: str) -> str:
    det = _parse_active(sentence)[0]
    if det == "the":
        return "def"
    if det in {"a", "an"}:
        return "indef"
    raise ValueError(f"Unexpected determiner in active sentence: {sentence}")


def _active_tense(active_verb: str) -> str:
    token = str(active_verb).strip().lower()
    if token.endswith("s"):
        return "present"
    if token:
        return "past"
    return "unknown"


def _expected_tense(passive_aux: str) -> str:
    if passive_aux == "is":
        return "present"
    if passive_aux == "was":
        return "past"
    return "unknown"


def _build_prime_assignment(
    *,
    target_frame: pd.DataFrame,
    prime_frame: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, List[int]]:
    if len(prime_frame) < len(target_frame):
        raise ValueError(
            f"Prime frame too small for mixed assignment: {len(prime_frame)} < {len(target_frame)}"
        )

    prime_aux = prime_frame["pp"].map(_passive_aux)
    prime_det = prime_frame["pa"].map(_det_family_from_active)

    buckets: Dict[Tuple[str, str], List[int]] = {}
    for aux in ("is", "was"):
        for det in ("def", "indef"):
            mask = prime_aux.eq(aux) & prime_det.eq(det)
            buckets[(aux, det)] = prime_aux[mask].index.tolist()

    rng = np.random.default_rng(seed + 104729)
    for key in buckets:
        rng.shuffle(buckets[key])

    selected_prime_indices: List[int] = []
    for _, target_row in target_frame.iterrows():
        target_aux = _passive_aux(str(target_row["tp"]))
        target_det = _det_family_from_active(str(target_row["ta"]))
        required_aux = "was" if target_aux == "is" else "is"
        required_det = "def" if target_det == "indef" else "indef"
        pool = buckets[(required_aux, required_det)]
        if not pool:
            raise ValueError(
                "Cannot satisfy strict mixed assignment: exhausted rows for "
                f"aux='{required_aux}', det='{required_det}'."
            )
        selected_prime_indices.append(pool.pop())

    prime_sample = prime_frame.loc[selected_prime_indices].reset_index(drop=True)
    return prime_sample, selected_prime_indices


def _strict_audit(frame: pd.DataFrame) -> Dict[str, int]:
    violations = {
        "same_aux_rows": 0,
        "same_det_family_rows": 0,
        "shared_noun_rows": 0,
        "same_active_verb_rows": 0,
        "prime_tense_mismatch_rows": 0,
        "target_tense_mismatch_rows": 0,
    }

    for row in frame.itertuples(index=False):
        pa_det_a, pa_agent, pa_verb, _, pa_patient = _parse_active(str(row.pa))
        ta_det_a, ta_agent, ta_verb, _, ta_patient = _parse_active(str(row.ta))
        pp_aux = _passive_aux(str(row.pp))
        tp_aux = _passive_aux(str(row.tp))

        violations["same_aux_rows"] += int(pp_aux == tp_aux)
        violations["same_det_family_rows"] += int(
            _det_family_from_active(str(row.pa)) == _det_family_from_active(str(row.ta))
        )
        violations["shared_noun_rows"] += int(bool({pa_agent, pa_patient} & {ta_agent, ta_patient}))
        violations["same_active_verb_rows"] += int(pa_verb == ta_verb)
        violations["prime_tense_mismatch_rows"] += int(_active_tense(pa_verb) != _expected_tense(pp_aux))
        violations["target_tense_mismatch_rows"] += int(_active_tense(ta_verb) != _expected_tense(tp_aux))

        # Keep determiner parse validated on both sides.
        _ = pa_det_a, ta_det_a

    return violations


def main() -> None:
    args = parse_args()

    core_frame = normalize_transitive_frame(pd.read_csv(args.core_csv.resolve()))
    jabber_frame = normalize_transitive_frame(pd.read_csv(args.jabberwocky_csv.resolve()))

    if args.n_items > len(core_frame):
        raise ValueError(f"Requested n_items={args.n_items} exceeds CORE rows={len(core_frame)}")

    target_sample = core_frame.head(args.n_items).reset_index(drop=True)
    prime_sample, selected_prime_indices = _build_prime_assignment(
        target_frame=target_sample,
        prime_frame=jabber_frame,
        seed=args.seed,
    )

    mixed_frame = pd.DataFrame(
        {
            "pa": prime_sample["pa"].astype(str),
            "pp": prime_sample["pp"].astype(str),
            "ta": target_sample["ta"].astype(str),
            "tp": target_sample["tp"].astype(str),
        }
    )

    strict_audit = _strict_audit(mixed_frame)
    overlap_audit = lexical_overlap_audit(target_frame=target_sample, prime_frame=prime_sample)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    mixed_frame.to_csv(args.output_csv, index=False)

    summary = {
        "core_csv": str(args.core_csv.resolve()),
        "jabberwocky_csv": str(args.jabberwocky_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "summary_json": str(args.summary_json.resolve()),
        "n_items": int(args.n_items),
        "seed": int(args.seed),
        "selected_prime_source_row_indices_preview": [int(index) for index in selected_prime_indices[:25]],
        "strict_audit": strict_audit,
        "lexical_overlap_audit": overlap_audit,
        "constraints": [
            "Prime and target passive auxiliaries are always opposite (is/was mismatch).",
            "Prime and target determiner families are always opposite (a/an vs the).",
            "Prime and target keep no noun or active-verb overlap.",
            "Prime and target active tense matches their own passive auxiliary.",
        ],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    violated = {key: value for key, value in strict_audit.items() if int(value) > 0}
    if violated:
        raise RuntimeError(f"Mixed corpus strict audit failed: {violated}")

    print(f"Saved {args.output_csv}")
    print(f"Summary: {args.summary_json}")


if __name__ == "__main__":
    main()
