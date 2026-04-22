import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch
from transformers import AutoTokenizer

from production_priming_common import (
    REPO_ROOT,
    classify_generated_structure,
    get_device,
    load_causal_lm_and_tokenizer,
    normalize_generated_text,
)


DEFAULT_PROMPT_CSV = (
    REPO_ROOT / "corpora" / "transitive" / "experiment_2_core_demo_prompts_lexically_controlled.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "behavioral_results" / "_smoke_demo_prompt_generation_audit"
PROMPT_COLUMN_TO_CONDITION = {
    "prompt_active": "active",
    "prompt_passive": "passive",
    "prompt_no_demo": "no_demo",
    "prompt_filler": "filler",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full-sentence continuations from Experiment 2 prompt CSVs."
    )
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--prompt-csv", type=Path, default=DEFAULT_PROMPT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument(
        "--prompt-columns",
        nargs="+",
        default=["prompt_active", "prompt_passive", "prompt_no_demo", "prompt_filler"],
        help="Subset of prompt_active prompt_passive prompt_no_demo prompt_filler.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=24)
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
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def first_sentence(text: str) -> str:
    stripped = text.strip().replace("\n", " ").replace('"', "").strip()
    if not stripped:
        return ""
    if "." not in stripped:
        return stripped
    head = stripped.split(".", 1)[0].strip()
    return f"{head} ."


def completion_to_answer(completion: str, stub: str = "The") -> str:
    joined = f"{stub}{completion}"
    return first_sentence(joined)


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


def validate_prompt_columns(prompt_columns: Sequence[str]) -> List[str]:
    invalid = sorted(set(prompt_columns).difference(PROMPT_COLUMN_TO_CONDITION))
    if invalid:
        raise ValueError(f"Unsupported prompt columns: {invalid}")
    if not prompt_columns:
        raise ValueError("At least one prompt column is required.")
    return list(prompt_columns)


def build_generation_rows(frame: pd.DataFrame, prompt_columns: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for _, record in frame.iterrows():
        for column in prompt_columns:
            rows.append(
                {
                    "item_index": int(record["item_index"]),
                    "prompt_column": column,
                    "prime_condition": PROMPT_COLUMN_TO_CONDITION[column],
                    "target_active": str(record["target_active"]),
                    "target_passive": str(record["target_passive"]),
                    "prompt": str(record[column]),
                }
            )
    return rows


def generation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    counts = (
        frame.groupby(["prompt_column", "prime_condition", "generation_class"], as_index=False)
        .agg(n_items=("item_index", "count"))
    )
    totals = (
        frame.groupby(["prompt_column", "prime_condition"], as_index=False)
        .agg(total_items=("item_index", "count"))
    )
    merged = counts.merge(totals, on=["prompt_column", "prime_condition"], how="left")
    merged["share"] = merged["n_items"] / merged["total_items"]
    return merged.sort_values(["prompt_column", "prime_condition", "generation_class"])


def quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["is_active_like"] = annotated["generation_class"].str.startswith("active")
    annotated["is_passive_like"] = annotated["generation_class"].str.startswith("passive")
    annotated["is_exact"] = annotated["generation_class"].isin(["active_exact", "passive_exact"])
    annotated["is_prefix"] = annotated["generation_class"].isin(["active_prefix", "passive_prefix"])
    annotated["is_structural"] = annotated["generation_class"].isin(
        ["active_structural", "passive_structural"]
    )
    annotated["is_congruent"] = (
        ((annotated["prime_condition"] == "active") & annotated["is_active_like"])
        | ((annotated["prime_condition"] == "passive") & annotated["is_passive_like"])
    )
    return (
        annotated.groupby(["prompt_column", "prime_condition"], as_index=False)
        .agg(
            n_items=("item_index", "count"),
            active_like_rate=("is_active_like", "mean"),
            passive_like_rate=("is_passive_like", "mean"),
            exact_rate=("is_exact", "mean"),
            prefix_rate=("is_prefix", "mean"),
            structural_rate=("is_structural", "mean"),
            congruent_rate=("is_congruent", "mean"),
        )
        .sort_values(["prompt_column", "prime_condition"])
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    prompt_frame = pd.read_csv(args.prompt_csv.resolve())
    if args.max_items is not None:
        prompt_frame = prompt_frame.head(int(args.max_items)).copy()
    prompt_columns = validate_prompt_columns(args.prompt_columns)
    generation_rows = build_generation_rows(prompt_frame, prompt_columns)
    prompts = [row["prompt"] for row in generation_rows]

    device = get_device(args.device)
    _, model, resolved_dtype = load_causal_lm_and_tokenizer(
        model_name=args.model_name,
        device=device,
        local_files_only=args.local_files_only,
        torch_dtype_name=args.torch_dtype,
    )
    generation_tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    if generation_tokenizer.pad_token_id is None:
        generation_tokenizer.pad_token = generation_tokenizer.eos_token
    generation_tokenizer.padding_side = "left"

    completions = batched_greedy_generate(
        prompts=prompts,
        model=model,
        tokenizer=generation_tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    rows: List[Dict[str, object]] = []
    for metadata, completion in zip(generation_rows, completions):
        generated_answer = completion_to_answer(completion, stub="The")
        rows.append(
            {
                **metadata,
                "greedy_completion_raw": completion,
                "greedy_answer_first_sentence": generated_answer,
                "greedy_answer_first_sentence_normalized": normalize_generated_text(generated_answer),
                "generation_class": classify_generated_structure(
                    generated_answer,
                    target_active=metadata["target_active"],
                    target_passive=metadata["target_passive"],
                ),
            }
        )

    results = pd.DataFrame(rows)
    summary = generation_summary(results)
    quality = quality_summary(results)

    results.to_csv(output_dir / "item_generations.csv", index=False)
    summary.to_csv(output_dir / "generation_summary.csv", index=False)
    quality.to_csv(output_dir / "generation_quality_summary.csv", index=False)

    preview = results[
        [
            "item_index",
            "prompt_column",
            "prime_condition",
            "target_active",
            "target_passive",
            "greedy_answer_first_sentence",
            "generation_class",
        ]
    ].copy()
    preview.to_csv(output_dir / "generation_examples.csv", index=False)

    metadata = {
        "model_name": args.model_name,
        "prompt_csv": str(args.prompt_csv.resolve()),
        "max_items": int(len(prompt_frame)),
        "prompt_columns": prompt_columns,
        "batch_size": int(args.batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "device": device,
        "torch_dtype": str(resolved_dtype) if resolved_dtype is not None else "default",
        "local_files_only": bool(args.local_files_only),
        "seed": int(args.seed),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# Demo Prompt Generation Audit",
        "",
        "Model metadata:",
        "```json",
        json.dumps(metadata, indent=2),
        "```",
        "",
        "Generation quality summary:",
        "```csv",
        quality.to_csv(index=False).strip(),
        "```",
        "",
        "Generation class summary:",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
