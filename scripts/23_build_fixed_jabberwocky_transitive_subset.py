import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from production_priming_common import REPO_ROOT, lexical_overlap_audit, normalize_transitive_frame


DEFAULT_SOURCE = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_bpe_filtered.csv"
DEFAULT_OUTPUT = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_bpe_filtered_2048.csv"
DEFAULT_SUMMARY = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_bpe_filtered_2048_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a fixed-size, fixed-seed Jabberwocky transitive subset and "
            "audit strict Sinclair-style row constraints."
        )
    )
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--n-items", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def parse_active(sentence: str):
    tokens = str(sentence).strip().split()
    if len(tokens) != 6:
        raise ValueError(f"Unexpected active sentence format: {sentence}")
    return tokens[0].lower(), tokens[1].lower(), tokens[2].lower(), tokens[3].lower(), tokens[4].lower()


def parse_passive(sentence: str):
    tokens = str(sentence).strip().split()
    if len(tokens) != 8:
        raise ValueError(f"Unexpected passive sentence format: {sentence}")
    return tokens[0].lower(), tokens[1].lower(), tokens[2].lower(), tokens[3].lower(), tokens[5].lower(), tokens[6].lower()


def det_family(det: str) -> str:
    token = det.lower().strip()
    if token == "the":
        return "def"
    if token in {"a", "an"}:
        return "indef"
    return "unknown"


def tense_match_for_nonce(active_verb: str, passive_aux: str) -> bool:
    if passive_aux == "is":
        return active_verb.endswith("s")
    if passive_aux == "was":
        return active_verb.endswith("ed")
    return False


def strict_row_audit(frame: pd.DataFrame) -> Dict[str, object]:
    same_aux_rows = 0
    shared_noun_rows = 0
    same_verb_rows = 0
    same_det_family_rows = 0
    prime_tense_mismatch_rows = 0
    target_tense_mismatch_rows = 0

    for row in frame.itertuples(index=False):
        pa_det_a, pa_agent, pa_verb, pa_det_p, pa_patient = parse_active(str(row.pa))
        ta_det_a, ta_agent, ta_verb, ta_det_p, ta_patient = parse_active(str(row.ta))
        _, _, pp_aux, pp_part, _, _ = parse_passive(str(row.pp))
        _, _, tp_aux, tp_part, _, _ = parse_passive(str(row.tp))

        same_aux_rows += int(pp_aux == tp_aux)
        shared_noun_rows += int(bool({pa_agent, pa_patient} & {ta_agent, ta_patient}))
        same_verb_rows += int(pa_verb == ta_verb)
        same_det_family_rows += int(det_family(pa_det_a) == det_family(ta_det_a))
        prime_tense_mismatch_rows += int(not tense_match_for_nonce(pa_verb, pp_aux))
        target_tense_mismatch_rows += int(not tense_match_for_nonce(ta_verb, tp_aux))

        # Within-row consistency for passive participles from nonce generator.
        if pp_part == "" or tp_part == "":
            raise ValueError("Unexpected empty passive participle in nonce row.")

    total = len(frame)
    return {
        "rows_evaluated": int(total),
        "same_active_verb_rows": int(same_verb_rows),
        "same_active_verb_rate": float(same_verb_rows / total) if total else 0.0,
        "shared_noun_rows": int(shared_noun_rows),
        "shared_noun_rate": float(shared_noun_rows / total) if total else 0.0,
        "same_aux_rows": int(same_aux_rows),
        "same_aux_rate": float(same_aux_rows / total) if total else 0.0,
        "same_det_family_rows": int(same_det_family_rows),
        "same_det_family_rate": float(same_det_family_rows / total) if total else 0.0,
        "prime_tense_mismatch_rows": int(prime_tense_mismatch_rows),
        "target_tense_mismatch_rows": int(target_tense_mismatch_rows),
    }


def main() -> None:
    args = parse_args()
    source_path = args.source_csv.resolve()
    source_frame = normalize_transitive_frame(pd.read_csv(source_path))

    if args.n_items > len(source_frame):
        raise ValueError(
            f"Requested n_items={args.n_items} exceeds source rows={len(source_frame)}."
        )

    rng = np.random.default_rng(args.seed)
    sampled_indices = rng.choice(len(source_frame), size=args.n_items, replace=False)
    sampled_frame = source_frame.iloc[sampled_indices].reset_index(drop=True)

    overlap_audit = lexical_overlap_audit(target_frame=sampled_frame, prime_frame=sampled_frame)
    strict_audit = strict_row_audit(sampled_frame)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled_frame.to_csv(args.output_csv, index=False)

    summary = {
        "source_csv": str(source_path),
        "output_csv": str(args.output_csv.resolve()),
        "summary_json": str(args.summary_json.resolve()),
        "n_items": int(args.n_items),
        "seed": int(args.seed),
        "sampling": "without_replacement",
        "sampled_source_row_indices_preview": [int(index) for index in sampled_indices[:25]],
        "within_row_lexical_audit": overlap_audit,
        "within_row_strict_audit": strict_audit,
        "notes": [
            "This file freezes the Jabberwocky prime pool at the strict-core matched size.",
            "Within-row strict audit enforces no noun/verb overlap, aux mismatch, determiner-family mismatch, and tense matching.",
        ],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if int(strict_audit["same_active_verb_rows"]) != 0:
        raise RuntimeError("Strict audit failed: same active verb across prime/target rows.")
    if int(strict_audit["shared_noun_rows"]) != 0:
        raise RuntimeError("Strict audit failed: shared nouns across prime/target rows.")
    if int(strict_audit["same_aux_rows"]) != 0:
        raise RuntimeError("Strict audit failed: overlapping passive auxiliaries.")
    if int(strict_audit["same_det_family_rows"]) != 0:
        raise RuntimeError("Strict audit failed: overlapping determiner families.")
    if int(strict_audit["prime_tense_mismatch_rows"]) != 0 or int(strict_audit["target_tense_mismatch_rows"]) != 0:
        raise RuntimeError("Strict audit failed: active/passive tense mismatch.")

    print(f"Saved fixed subset to {args.output_csv}")
    print(f"Summary: {args.summary_json}")


if __name__ == "__main__":
    main()
