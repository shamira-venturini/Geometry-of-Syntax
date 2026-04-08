import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer

from production_priming_common import (
    CORE_FILLER_SENTENCES,
    JABBERWOCKY_FILLER_SENTENCES,
    LEXICALLY_CONTROLLED_CORE_CSV,
    REPO_ROOT,
    TargetBundle,
    batched_choice_log_probs,
    build_prompt,
    classify_generated_structure,
    extract_bundle,
    get_device,
    lexical_overlap_audit,
    load_causal_lm_and_tokenizer,
    load_verb_lookup,
    normalize_generated_text,
    normalize_transitive_frame,
    prompt_condition_order,
    prompt_templates,
    resolve_prime_sentence,
    role_sequence,
    sample_condition_frames,
    write_common_outputs,
)


DEFAULT_INPUT_CSV = LEXICALLY_CONTROLLED_CORE_CSV
DEFAULT_OUTPUT_DIR = REPO_ROOT / "behavioral_results" / "counterbalanced_generation_choice"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the looser counterbalanced production experiment with open sentence prompts."
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
    parser.add_argument("--batch-size", type=int, default=64)
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
        choices=("word_list", "role_labeled", "all"),
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
    )
    parser.add_argument(
        "--generate-greedy",
        dest="generate_greedy",
        action="store_true",
        default=True,
        help="Also save greedy generations for the open prompt.",
    )
    parser.add_argument(
        "--no-generate-greedy",
        dest="generate_greedy",
        action="store_false",
        help="Skip greedy generations and only score active/passive continuations.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=13)
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
) -> Tuple[List[Tuple[str, int, List[str], List[int]]], List[Dict[str, object]], List[str]]:
    verb_lookup = load_verb_lookup()
    prompt_groups: List[Tuple[str, int, List[str], List[int]]] = []
    row_metadata: List[Dict[str, object]] = []
    prompts_for_generation: List[str] = []

    for item_index, target_row in target_frame.iterrows():
        prime_row = prime_frame.loc[item_index]
        bundle: TargetBundle = extract_bundle(target_row["ta"], target_row["tp"], verb_lookup=verb_lookup)
        item_rng = random.Random(seed + item_index * 104729)

        active_candidate = f" {str(target_row['ta']).strip()}"
        passive_candidate = f" {str(target_row['tp']).strip()}"
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
                    sentence_stub="Sentence:",
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
                        "sentence_stub": "Sentence:",
                        "choice_target": "full_sentence",
                    }
                )
                prompts_for_generation.append(prompt)

    return prompt_groups, row_metadata, prompts_for_generation


def infer_filler_domain(input_csv: Path, prime_csv: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    probe = f"{input_csv.name} {prime_csv.name}".lower()
    return "jabberwocky" if "jabberwocky" in probe else "core"


def first_sentence(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if "." not in stripped:
        return stripped
    head = stripped.split(".", 1)[0].strip()
    return f"{head} ."


def batched_greedy_generate(
    prompts: Sequence[str],
    model,
    tokenizer,
    device: str,
    batch_size: int,
    max_new_tokens: int,
) -> List[str]:
    outputs: List[str] = []
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = list(prompts[batch_start:batch_start + batch_size])
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_width = int(inputs.input_ids.shape[1])
        for row_index in range(len(batch_prompts)):
            new_tokens = generated[row_index, input_width:]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return outputs


def generation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    counts = (
        frame.groupby(["prompt_template", "prime_condition", "greedy_generation_class"], as_index=False)
        .agg(n_items=("item_index", "count"))
    )
    totals = (
        frame.groupby(["prompt_template", "prime_condition"], as_index=False)
        .agg(total_items=("item_index", "count"))
    )
    merged = counts.merge(totals, on=["prompt_template", "prime_condition"], how="left")
    merged["share"] = merged["n_items"] / merged["total_items"]
    return merged.sort_values(["prompt_template", "prime_condition", "greedy_generation_class"])


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
    tokenizer.padding_side = "right"

    prime_conditions = prompt_condition_order(args.prime_conditions)
    filler_domain = infer_filler_domain(input_csv=input_csv, prime_csv=prime_csv, requested=args.filler_domain)
    filler_sentences = (
        JABBERWOCKY_FILLER_SENTENCES if filler_domain == "jabberwocky" else CORE_FILLER_SENTENCES
    )
    prompt_groups, row_metadata, prompts_for_generation = build_prompt_groups(
        target_frame=target_frame,
        prime_frame=prime_frame,
        tokenizer=tokenizer,
        template_mode=args.prompt_template,
        prime_conditions=prime_conditions,
        role_order_mode=args.role_order,
        seed=args.seed,
        filler_sentences=filler_sentences,
    )
    batched_scores = batched_choice_log_probs(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt_groups=prompt_groups,
        batch_size=args.batch_size,
    )

    greedy_generations: List[str] = []
    if args.generate_greedy:
        generation_tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            local_files_only=args.local_files_only,
        )
        if generation_tokenizer.pad_token_id is None:
            generation_tokenizer.pad_token = generation_tokenizer.eos_token
        generation_tokenizer.padding_side = "left"
        greedy_generations = batched_greedy_generate(
            prompts=prompts_for_generation,
            model=model,
            tokenizer=generation_tokenizer,
            device=device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        greedy_generations = [""] * len(row_metadata)

    rows: List[Dict[str, object]] = []
    for metadata, candidate_log_probs, raw_generation in zip(row_metadata, batched_scores, greedy_generations):
        active_lp, passive_lp = candidate_log_probs
        chosen_structure = "passive" if passive_lp > active_lp else "active"
        generated_sentence = first_sentence(raw_generation)
        rows.append(
            {
                **metadata,
                "active_choice_logprob": active_lp,
                "passive_choice_logprob": passive_lp,
                "passive_minus_active_logprob": passive_lp - active_lp,
                "chosen_structure": chosen_structure,
                "passive_choice_indicator": 1.0 if chosen_structure == "passive" else 0.0,
                "greedy_generation_raw": raw_generation,
                "greedy_generation_first_sentence": generated_sentence,
                "greedy_generation_first_sentence_normalized": normalize_generated_text(generated_sentence),
                "greedy_generation_class": classify_generated_structure(
                    generated_sentence,
                    target_active=str(metadata["target_active"]),
                    target_passive=str(metadata["target_passive"]),
                ) if args.generate_greedy else "not_run",
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
        "task_type": "counterbalanced_generation_choice",
        "generate_greedy": bool(args.generate_greedy),
        "max_new_tokens": int(args.max_new_tokens),
    }
    write_common_outputs(
        frame=results,
        output_dir=output_dir,
        title="Counterbalanced Generation-Style Production Experiment",
        prime_condition_ordering=prime_conditions,
        extra_metadata=metadata,
    )

    if args.generate_greedy:
        generation_summary(results).to_csv(output_dir / "generation_summary.csv", index=False)


if __name__ == "__main__":
    main()
