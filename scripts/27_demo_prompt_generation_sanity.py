import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from production_priming_common import (
    JABBERWOCKY_FILLER_SENTENCES,
    CORE_FILLER_SENTENCES,
    LEXICALLY_CONTROLLED_CORE_CSV,
    REPO_ROOT,
    extract_bundle,
    get_device,
    lexical_overlap_audit,
    load_verb_lookup,
    normalize_generated_text,
    normalize_transitive_frame,
    sample_condition_frames,
)


DEFAULT_JABBERWOCKY_PRIMES = REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_bpe_filtered_2080.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "behavioral_results" / "experiment-2" / "demo_prompt_generation_sanity"
DEMO_SCRIPT_PATH = REPO_ROOT / "scripts" / "24_demo_prompt_completion_experiment.py"


def load_demo_module():
    spec = importlib.util.spec_from_file_location("demo_prompt_completion", DEMO_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load demo prompt module from {DEMO_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small greedy-generation sanity check for the demo-prompt production prompts."
    )
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--input-csv", type=Path, default=LEXICALLY_CONTROLLED_CORE_CSV)
    parser.add_argument("--prime-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-items", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer/model from the local Hugging Face cache only.",
    )
    parser.add_argument(
        "--prime-conditions",
        nargs="+",
        default=["active", "passive", "no_demo", "filler"],
    )
    parser.add_argument(
        "--filler-domain",
        choices=("auto", "core", "jabberwocky"),
        default="auto",
    )
    parser.add_argument(
        "--event-style",
        choices=("there_was_event", "involving_event", "all"),
        default="all",
    )
    parser.add_argument(
        "--role-style",
        choices=("responsible_affected", "did_to", "all"),
        default="all",
    )
    parser.add_argument(
        "--quote-style",
        choices=("mary_answered", "said_mary", "all"),
        default="all",
    )
    parser.add_argument("--max-new-tokens", type=int, default=18)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


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


def infer_filler_domain(input_csv: Path, prime_csv: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    probe = f"{input_csv.name} {prime_csv.name}".lower()
    return "jabberwocky" if "jabberwocky" in probe else "core"


def first_sentence_from_answer(raw_generation: str) -> str:
    text = raw_generation.replace("\n", " ").strip()
    if not text:
        return ""
    text = text.split('"', 1)[0].strip()
    for stop_symbol in [".", "?", "!"]:
        if stop_symbol in text:
            head = text.split(stop_symbol, 1)[0].strip()
            if head:
                return f"The {head} .".replace("The The ", "The ")
    return f"The {text}".replace("The The ", "The ").strip()


def classify_demo_generation(
    sentence: str,
    agent_noun: str,
    patient_noun: str,
    active_verb_form: str,
    passive_verb_form: str,
) -> str:
    normalized = normalize_generated_text(sentence)
    active_prefix = normalize_generated_text(f"the {agent_noun} {active_verb_form}")
    passive_prefix = normalize_generated_text(f"the {patient_noun} is {passive_verb_form} by")
    patient_start = normalize_generated_text(f"the {patient_noun}")
    agent_start = normalize_generated_text(f"the {agent_noun}")

    if normalized.startswith(active_prefix):
        return "active_like"
    if normalized.startswith(passive_prefix):
        return "passive_like"
    if normalized.startswith(agent_start):
        return "agent_start_other"
    if normalized.startswith(patient_start):
        return "patient_start_other"
    return "other"


def has_extra_text(raw_generation: str) -> bool:
    text = raw_generation.replace("\n", " ").strip()
    if not text:
        return False
    if '"' in text:
        trailing = text.split('"', 1)[1].strip()
        return bool(trailing)
    sentence_end = text.count(".") + text.count("?") + text.count("!")
    return sentence_end > 1


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_module = load_demo_module()
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    verb_lookup = load_verb_lookup()
    prime_conditions = demo_module.demo_prime_condition_order(list(args.prime_conditions))
    filler_domain = infer_filler_domain(input_csv=input_csv, prime_csv=prime_csv, requested=args.filler_domain)
    filler_sentences = JABBERWOCKY_FILLER_SENTENCES if filler_domain == "jabberwocky" else CORE_FILLER_SENTENCES

    rows: List[Dict[str, object]] = []
    prompts: List[str] = []
    prompt_metadata: List[Dict[str, object]] = []

    for item_index, target_row in target_frame.iterrows():
        prime_row = prime_frame.loc[item_index]
        target_bundle = extract_bundle(str(target_row["ta"]), str(target_row["tp"]), verb_lookup=verb_lookup)
        prime_bundle = extract_bundle(str(prime_row["pa"]), str(prime_row["pp"]), verb_lookup=verb_lookup)
        filler_sentence = filler_sentences[item_index % len(filler_sentences)]

        for event_style in demo_module.event_style_values(args.event_style):
            for role_style in demo_module.role_style_values(args.role_style):
                for quote_style in demo_module.quote_style_values(args.quote_style):
                    prompt_template_name = f"demo__{event_style}__{role_style}__{quote_style}"
                    for prime_condition in prime_conditions:
                        if prime_condition == "active":
                            prime_sentence = str(prime_row["pa"])
                        elif prime_condition == "passive":
                            prime_sentence = str(prime_row["pp"])
                        else:
                            prime_sentence = None

                        prompt = demo_module.build_prompt(
                            target_bundle=target_bundle,
                            prime_condition=prime_condition,
                            prime_bundle=prime_bundle,
                            prime_sentence=prime_sentence,
                            filler_sentence=filler_sentence if prime_condition == "filler" else None,
                            event_style=event_style,
                            role_style=role_style,
                            quote_style=quote_style,
                        )
                        prompts.append(prompt)
                        prompt_metadata.append(
                            {
                                "item_index": item_index,
                                "prompt_template": prompt_template_name,
                                "prime_condition": prime_condition,
                                "prompt": prompt,
                                "target_active": str(target_row["ta"]),
                                "target_passive": str(target_row["tp"]),
                                "agent_noun": target_bundle.agent_noun,
                                "patient_noun": target_bundle.patient_noun,
                                "active_verb_form": target_bundle.active_verb_form,
                                "passive_verb_form": target_bundle.passive_verb_form,
                                "quote_style": quote_style,
                                "event_style": event_style,
                                "role_style": role_style,
                            }
                        )

    raw_generations = batched_greedy_generate(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    for metadata, raw_generation in zip(prompt_metadata, raw_generations):
        generated_sentence = first_sentence_from_answer(raw_generation)
        generation_class = classify_demo_generation(
            sentence=generated_sentence,
            agent_noun=metadata["agent_noun"],
            patient_noun=metadata["patient_noun"],
            active_verb_form=metadata["active_verb_form"],
            passive_verb_form=metadata["passive_verb_form"],
        )
        rows.append(
            {
                **metadata,
                "raw_generation": raw_generation,
                "generated_first_sentence": generated_sentence,
                "generation_class": generation_class,
                "has_extra_text_after_first_sentence": has_extra_text(raw_generation),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "item_generations.csv", index=False)

    summary = (
        frame.groupby(["prompt_template", "prime_condition", "generation_class"], as_index=False)
        .agg(n_items=("item_index", "count"))
    )
    totals = (
        frame.groupby(["prompt_template", "prime_condition"], as_index=False)
        .agg(total_items=("item_index", "count"))
    )
    summary = summary.merge(totals, on=["prompt_template", "prime_condition"], how="left")
    summary["share"] = summary["n_items"] / summary["total_items"]
    summary.to_csv(output_dir / "generation_summary.csv", index=False)

    extra = (
        frame.groupby(["prompt_template", "prime_condition"], as_index=False)
        .agg(
            total_items=("item_index", "count"),
            extra_text_rate=("has_extra_text_after_first_sentence", "mean"),
        )
    )
    extra.to_csv(output_dir / "generation_quality_summary.csv", index=False)

    examples = (
        frame.groupby(["prompt_template", "prime_condition"], as_index=False)
        .head(2)[
            [
                "prompt_template",
                "prime_condition",
                "prompt",
                "raw_generation",
                "generated_first_sentence",
                "generation_class",
            ]
        ]
    )
    examples.to_csv(output_dir / "generation_examples.csv", index=False)

    metadata = {
        "model_name": args.model_name,
        "input_csv": str(input_csv),
        "prime_csv": str(prime_csv),
        "max_items": int(args.max_items),
        "batch_size": int(args.batch_size),
        "prime_conditions": prime_conditions,
        "filler_domain": filler_domain,
        "device": device,
        "prime_alignment_mode": prime_alignment_mode,
        "lexical_overlap_audit": overlap_audit,
        "event_style": args.event_style,
        "role_style": args.role_style,
        "quote_style": args.quote_style,
        "max_new_tokens": int(args.max_new_tokens),
        "seed": int(args.seed),
        "paradigm": "demo_prompt_generation_sanity",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# Demo Prompt Generation Sanity Check",
        "",
        "## Generation Summary",
        "",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Generation Quality",
        "",
        "```csv",
        extra.to_csv(index=False).strip(),
        "```",
        "",
        "## Examples",
        "",
        "```csv",
        examples.to_csv(index=False).strip(),
        "```",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
