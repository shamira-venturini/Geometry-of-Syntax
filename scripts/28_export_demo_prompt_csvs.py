import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from production_priming_common import (
    CORE_FILLER_SENTENCES,
    JABBERWOCKY_FILLER_SENTENCES,
    REPO_ROOT,
    lexical_overlap_audit,
    load_verb_lookup,
    normalize_transitive_frame,
    sample_condition_frames,
)


STRICT_CORE = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_constrained_counterbalanced_lexically_controlled.csv"
LEXICAL_OVERLAP_CORE = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_constrained_counterbalanced.csv"
JABBERWOCKY_2080 = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_bpe_filtered_2080.csv"
DEMO_MODULE_PATH = REPO_ROOT / "scripts" / "24_demo_prompt_completion_experiment.py"


def load_demo_module():
    spec = importlib.util.spec_from_file_location("demo_prompt_completion_experiment", DEMO_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load demo prompt module from {DEMO_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export wide CSVs of Experiment 2 prompts for core and Jabberwocky demo-prime conditions."
    )
    parser.add_argument(
        "--core-prime-mode",
        choices=("lexically_controlled", "lexical_overlap"),
        default="lexically_controlled",
        help="Use the repaired core corpus or the older lexical-overlap version.",
    )
    parser.add_argument("--max-items", type=int, default=2080)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--event-style",
        choices=("there_was_event", "involving_event"),
        default="involving_event",
    )
    parser.add_argument(
        "--role-style",
        choices=("responsible_affected", "did_to"),
        default="did_to",
    )
    parser.add_argument(
        "--quote-style",
        choices=("mary_answered", "said_mary"),
        default="mary_answered",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "corpora" / "transitive",
    )
    return parser.parse_args()


def build_prompt_export_frame(
    target_frame: pd.DataFrame,
    prime_frame: pd.DataFrame,
    filler_sentences: List[str],
    event_style: str,
    role_style: str,
    quote_style: str,
    seed: int,
    demo_module,
) -> pd.DataFrame:
    verb_lookup = load_verb_lookup()
    rows: List[Dict[str, object]] = []

    for item_index, target_row in target_frame.iterrows():
        prime_row = prime_frame.loc[item_index]
        target_bundle = demo_module.extract_bundle(str(target_row["ta"]), str(target_row["tp"]), verb_lookup=verb_lookup)
        prime_bundle = demo_module.extract_bundle(str(prime_row["pa"]), str(prime_row["pp"]), verb_lookup=verb_lookup)
        filler_sentence = filler_sentences[item_index % len(filler_sentences)]

        prompts = {}
        for prime_condition in ["active", "passive", "no_demo", "filler"]:
            if prime_condition == "active":
                prime_sentence = str(prime_row["pa"])
            elif prime_condition == "passive":
                prime_sentence = str(prime_row["pp"])
            else:
                prime_sentence = None

            prompts[prime_condition] = demo_module.build_prompt(
                target_bundle=target_bundle,
                prime_condition=prime_condition,
                prime_bundle=prime_bundle,
                prime_sentence=prime_sentence,
                filler_sentence=filler_sentence if prime_condition == "filler" else None,
                event_style=event_style,
                role_style=role_style,
                quote_style=quote_style,
            )

        rows.append(
            {
                "item_index": int(item_index),
                "prime_active_sentence": str(prime_row["pa"]),
                "prime_passive_sentence": str(prime_row["pp"]),
                "target_active": str(target_row["ta"]),
                "target_passive": str(target_row["tp"]),
                "filler_sentence": filler_sentence,
                "event_style": event_style,
                "role_style": role_style,
                "quote_style": quote_style,
                "prompt_active": prompts["active"],
                "prompt_passive": prompts["passive"],
                "prompt_no_demo": prompts["no_demo"],
                "prompt_filler": prompts["filler"],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    demo_module = load_demo_module()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    core_csv = STRICT_CORE if args.core_prime_mode == "lexically_controlled" else LEXICAL_OVERLAP_CORE
    target_frame = normalize_transitive_frame(pd.read_csv(core_csv))
    core_prime_frame = normalize_transitive_frame(pd.read_csv(core_csv))
    jabber_prime_frame = normalize_transitive_frame(pd.read_csv(JABBERWOCKY_2080))

    target_core_sample, core_prime_sample, core_alignment_mode = sample_condition_frames(
        target_frame=target_frame,
        prime_frame=core_prime_frame,
        max_items=args.max_items,
        seed=args.seed,
    )
    target_jabber_sample, jabber_prime_sample, jabber_alignment_mode = sample_condition_frames(
        target_frame=target_frame,
        prime_frame=jabber_prime_frame,
        max_items=args.max_items,
        seed=args.seed,
    )

    core_export = build_prompt_export_frame(
        target_frame=target_core_sample,
        prime_frame=core_prime_sample,
        filler_sentences=CORE_FILLER_SENTENCES,
        event_style=args.event_style,
        role_style=args.role_style,
        quote_style=args.quote_style,
        seed=args.seed,
        demo_module=demo_module,
    )
    jabber_export = build_prompt_export_frame(
        target_frame=target_jabber_sample,
        prime_frame=jabber_prime_sample,
        filler_sentences=JABBERWOCKY_FILLER_SENTENCES,
        event_style=args.event_style,
        role_style=args.role_style,
        quote_style=args.quote_style,
        seed=args.seed,
        demo_module=demo_module,
    )

    core_path = output_dir / f"experiment_2_core_demo_prompts_{args.core_prime_mode}.csv"
    jabber_path = output_dir / f"experiment_2_jabberwocky_demo_prompts_{args.core_prime_mode}.csv"
    summary_path = output_dir / f"experiment_2_demo_prompts_{args.core_prime_mode}_summary.json"

    core_export.to_csv(core_path, index=False)
    jabber_export.to_csv(jabber_path, index=False)

    summary = {
        "core_prime_mode": args.core_prime_mode,
        "event_style": args.event_style,
        "role_style": args.role_style,
        "quote_style": args.quote_style,
        "max_items": int(args.max_items),
        "seed": int(args.seed),
        "core_prompt_csv": str(core_path),
        "jabberwocky_prompt_csv": str(jabber_path),
        "core_alignment_mode": core_alignment_mode,
        "jabberwocky_alignment_mode": jabber_alignment_mode,
        "core_lexical_overlap_audit": lexical_overlap_audit(target_frame=target_core_sample, prime_frame=core_prime_sample),
        "jabberwocky_lexical_overlap_audit": lexical_overlap_audit(
            target_frame=target_jabber_sample,
            prime_frame=jabber_prime_sample,
        ),
        "n_core_rows": int(len(core_export)),
        "n_jabberwocky_rows": int(len(jabber_export)),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {core_path}")
    print(f"Saved {jabber_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
