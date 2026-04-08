import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

from production_priming_common import (
    CORE_FILLER_SENTENCES,
    JABBERWOCKY_FILLER_SENTENCES,
    LEXICALLY_CONTROLLED_CORE_CSV,
    REPO_ROOT,
    TargetBundle,
    batched_choice_log_probs,
    build_prompt,
    extract_bundle,
    get_device,
    lexical_overlap_audit,
    load_causal_lm_and_tokenizer,
    load_verb_lookup,
    normalize_transitive_frame,
    prompt_condition_order,
    prompt_templates,
    resolve_prime_sentence,
    role_sequence,
    sample_condition_frames,
    write_common_outputs,
)


DEFAULT_INPUT_CSV = LEXICALLY_CONTROLLED_CORE_CSV
DEFAULT_OUTPUT_DIR = REPO_ROOT / "behavioral_results" / "counterbalanced_completion_choice_controlled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the controlled counterbalanced completion-choice production experiment."
    )
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument(
        "--prime-csv",
        type=Path,
        default=None,
        help="Optional alternate prime source. If it differs in length, primes are sampled independently.",
    )
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
        "--prompt-template",
        choices=("word_list", "role_labeled", "cue_list", "another_event", "same_kind_event", "all", "elicited_all"),
        default="role_labeled",
    )
    parser.add_argument(
        "--prime-conditions",
        nargs="+",
        default=["active", "passive", "no_prime_eos", "filler"],
        help="Subset of active passive no_prime_eos no_prime_empty filler.",
    )
    parser.add_argument(
        "--filler-domain",
        choices=("auto", "core", "jabberwocky"),
        default="auto",
        help="Filler pool to use. auto infers from input/prime CSV path.",
    )
    parser.add_argument(
        "--role-order",
        choices=("fixed", "shuffle"),
        default="shuffle",
        help="Keep AGENT/PATIENT/VERB order fixed or shuffle it per item.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--sentence-stub",
        default="Sentence: the",
        help="Literal stub appended at the end of the prompt before the completion choice.",
    )
    return parser.parse_args()


def build_prompt_groups(
    target_frame: pd.DataFrame,
    prime_frame: pd.DataFrame,
    tokenizer,
    template_mode: str,
    prime_conditions: List[str],
    role_order_mode: str,
    seed: int,
    filler_sentences: List[str],
    sentence_stub: str,
) -> Tuple[List[Tuple[str, int, List[str], List[int]]], List[Dict[str, object]]]:
    verb_lookup = load_verb_lookup()
    prompt_groups: List[Tuple[str, int, List[str], List[int]]] = []
    row_metadata: List[Dict[str, object]] = []

    for item_index, target_row in target_frame.iterrows():
        prime_row = prime_frame.loc[item_index]
        bundle: TargetBundle = extract_bundle(target_row["ta"], target_row["tp"], verb_lookup=verb_lookup)
        item_rng = random.Random(seed + item_index * 104729)

        active_candidate = f" {bundle.agent_noun}"
        passive_candidate = f" {bundle.patient_noun}"
        candidates = [active_candidate, passive_candidate]
        candidate_lengths = [
            len(tokenizer(text, add_special_tokens=False)["input_ids"])
            for text in candidates
        ]

        for template_name in prompt_templates(template_mode):
            sequence_values = role_sequence(
                bundle=bundle,
                template_name=template_name,
                rng=item_rng,
                order_mode=role_order_mode,
            )
            for prime_condition in prime_conditions:
                prime_sentence = resolve_prime_sentence(
                    prime_condition=prime_condition,
                    prime_row=prime_row,
                    item_index=item_index,
                    filler_seed=seed,
                    filler_sentences=filler_sentences,
                )
                prompt = build_prompt(
                    prime_sentence=prime_sentence,
                    bundle=bundle,
                    template_name=template_name,
                    role_sequence_values=sequence_values,
                    sentence_stub=sentence_stub,
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
                        "prompt_template": template_name,
                        "prime_condition": prime_condition,
                        "prime_structure": prime_condition,
                        "prime_sentence": prime_sentence or "",
                        "target_active": str(target_row["ta"]),
                        "target_passive": str(target_row["tp"]),
                        "agent_det": bundle.agent_det,
                        "agent_noun": bundle.agent_noun,
                        "patient_det": bundle.patient_det,
                        "patient_noun": bundle.patient_noun,
                        "verb_lemma": bundle.verb_lemma,
                        "active_verb_form": bundle.active_verb_form,
                        "passive_verb_form": bundle.passive_verb_form,
                        "message_role_order_json": json.dumps(sequence_values),
                        "prompt": prompt,
                        "sentence_stub": sentence_stub,
                        "choice_target": "first_noun",
                    }
                )

    return prompt_groups, row_metadata


def infer_filler_domain(input_csv: Path, prime_csv: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    probe = f"{input_csv.name} {prime_csv.name}".lower()
    return "jabberwocky" if "jabberwocky" in probe else "core"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    input_csv = args.input_csv.resolve()
    target_frame = normalize_transitive_frame(pd.read_csv(input_csv))
    prime_csv = args.prime_csv.resolve() if args.prime_csv else input_csv
    prime_frame = normalize_transitive_frame(pd.read_csv(prime_csv))
    target_frame, prime_frame, prime_alignment_mode = sample_condition_frames(
        target_frame=target_frame,
        prime_frame=prime_frame,
        max_items=args.max_items,
        seed=args.seed,
    )
    overlap_audit = lexical_overlap_audit(target_frame=target_frame, prime_frame=prime_frame)

    device = get_device(args.device)
    tokenizer, model, resolved_dtype = load_causal_lm_and_tokenizer(
        model_name=args.model_name,
        device=device,
        local_files_only=args.local_files_only,
        torch_dtype_name=args.torch_dtype,
    )

    prime_conditions = prompt_condition_order(args.prime_conditions)
    filler_domain = infer_filler_domain(input_csv=input_csv, prime_csv=prime_csv, requested=args.filler_domain)
    filler_sentences = (
        JABBERWOCKY_FILLER_SENTENCES if filler_domain == "jabberwocky" else CORE_FILLER_SENTENCES
    )
    prompt_groups, row_metadata = build_prompt_groups(
        target_frame=target_frame,
        prime_frame=prime_frame,
        tokenizer=tokenizer,
        template_mode=args.prompt_template,
        prime_conditions=prime_conditions,
        role_order_mode=args.role_order,
        seed=args.seed,
        filler_sentences=filler_sentences,
        sentence_stub=args.sentence_stub,
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
        "input_csv": str(args.input_csv.resolve()),
        "prime_csv": str(prime_csv),
        "max_items": None if args.max_items is None else int(args.max_items),
        "batch_size": int(args.batch_size),
        "prompt_template": args.prompt_template,
        "prime_conditions": prime_conditions,
        "role_order": args.role_order,
        "sentence_stub": args.sentence_stub,
        "seed": int(args.seed),
        "device": device,
        "torch_dtype": str(resolved_dtype) if resolved_dtype is not None else "default",
        "local_files_only": bool(args.local_files_only),
        "n_rows": int(len(results)),
        "n_items": int(len(target_frame)),
        "prime_alignment_mode": prime_alignment_mode,
        "lexical_overlap_audit": overlap_audit,
        "filler_sentence_count": len(filler_sentences),
        "filler_domain": filler_domain,
        "task_type": "counterbalanced_completion_choice_controlled",
    }
    write_common_outputs(
        frame=results,
        output_dir=output_dir,
        title="Counterbalanced Completion-Choice Production Experiment",
        prime_condition_ordering=prime_conditions,
        extra_metadata=metadata,
    )


if __name__ == "__main__":
    main()
