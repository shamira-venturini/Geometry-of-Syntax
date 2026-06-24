#!/usr/bin/env python3
"""Run Experiment 4 complex-NP prompt-generation role recovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from production_priming_common import get_device, load_causal_lm_and_tokenizer  # noqa: E402
from src.exp4_analysis import run_exp4_analysis  # noqa: E402
from src.exp4_scoring import evaluate_generated_answer  # noqa: E402


DEFAULT_PROMPT_CSV = (
    REPO_ROOT
    / "behavioral_results/generated_materials/experiment-4/complex_np"
    / "experiment_4_complex_np_core_role_recovery_prompts.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "behavioral_results/experiment-4/complex_np_prompt_generation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--model-condition", default=None)
    parser.add_argument("--prompt-csv", type=Path, default=DEFAULT_PROMPT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-items",
        type=int,
        default=2048,
        help="Maximum base item_index value to keep. Keeps all prompt variants for selected items.",
    )
    parser.add_argument(
        "--prime-conditions",
        nargs="+",
        default=["active", "passive", "filler", "no_prime"],
    )
    parser.add_argument(
        "--target-voices",
        nargs="+",
        default=["active", "passive"],
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=["doer", "acted_on"],
    )
    parser.add_argument(
        "--complexity-conditions",
        nargs="+",
        default=["agent_complex", "patient_complex", "both_complex"],
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--ceiling-threshold", type=float, default=0.95)
    return parser.parse_args()


def _validate_subset(frame: pd.DataFrame, column: str, requested: Sequence[str]) -> List[str]:
    values = [str(value) for value in requested if str(value).strip()]
    if not values:
        raise ValueError(f"At least one {column} value must be selected.")
    available = set(frame[column].astype(str))
    invalid = sorted(set(values).difference(available))
    if invalid:
        raise ValueError(f"Invalid {column} values {invalid}; available={sorted(available)}")
    return values


def load_prompt_rows(args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_csv(args.prompt_csv.resolve()).fillna("")

    if args.max_items is not None:
        frame = frame.loc[frame["item_index"].astype(int) < int(args.max_items)].copy()

    prime_conditions = _validate_subset(frame, "prime_condition", args.prime_conditions)
    target_voices = _validate_subset(frame, "target_voice", args.target_voices)
    question_types = _validate_subset(frame, "question_type", args.question_types)
    complexity_conditions = _validate_subset(frame, "complexity_condition", args.complexity_conditions)

    filtered = frame.loc[
        frame["prime_condition"].astype(str).isin(prime_conditions)
        & frame["target_voice"].astype(str).isin(target_voices)
        & frame["question_type"].astype(str).isin(question_types)
        & frame["complexity_condition"].astype(str).isin(complexity_conditions)
    ].copy()

    if filtered.empty:
        raise ValueError("Prompt selection produced zero rows.")
    return filtered.reset_index(drop=True)


def batched_greedy_generate(
    prompts: Sequence[str],
    model,
    tokenizer,
    device: str,
    batch_size: int,
    max_new_tokens: int,
) -> List[str]:
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs: List[str] = []
    effective_batch_size = max(1, int(batch_size))
    total_batches = (len(prompts) + effective_batch_size - 1) // effective_batch_size
    report_every = max(1, total_batches // 20)
    try:
        for batch_number, batch_start in enumerate(
            range(0, len(prompts), effective_batch_size),
            start=1,
        ):
            if batch_number == 1 or batch_number == total_batches or batch_number % report_every == 0:
                print(
                    f"Exp4 generation batch {batch_number}/{total_batches} "
                    f"({batch_start}/{len(prompts)} prompts)",
                    flush=True,
                )
            batch_prompts = list(prompts[batch_start : batch_start + effective_batch_size])
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max(1, int(max_new_tokens)),
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            input_width = int(inputs.input_ids.shape[1])
            for row_index in range(len(batch_prompts)):
                new_tokens = generated[row_index, input_width:]
                outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = old_padding_side
    return outputs


def _first_clause(text: str) -> str:
    compact = " ".join(str(text).strip().split())
    if not compact:
        return ""
    for separator in [".", "?", "!", ";", "\n"]:
        if separator in compact:
            compact = compact.split(separator, 1)[0].strip()
            break
    return compact


def make_item_level(
    *,
    prompt_frame: pd.DataFrame,
    completions: Sequence[str],
    model_name: str,
    model_condition: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (_, record), completion in zip(prompt_frame.iterrows(), completions):
        full_answer = _first_clause(f"{str(record['answer_prefix']).rstrip()} {str(completion).lstrip()}")
        evaluation = evaluate_generated_answer(
            generated_answer_raw=full_answer,
            correct_answer=str(record["correct_answer"]),
            foil_answer=str(record["foil_answer"]),
        )
        rows.append(
            {
                **record.to_dict(),
                "experiment_id": "experiment_4_complex_np_role_recovery",
                "model_name": model_name,
                "model_condition": model_condition,
                "generated_answer_raw": completion,
                "generated_answer_full": full_answer,
                "generated_answer_normalized": evaluation.generated_answer_normalized,
                "matched_label": evaluation.matched_label,
                "is_correct": bool(evaluation.is_correct),
                "is_foil": bool(evaluation.matched_label == "foil"),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    *,
    item_level: pd.DataFrame,
    output_dir: Path,
    ceiling_threshold: float,
    metadata: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    item_level.to_csv(output_dir / "item_level_results_exp4.csv", index=False)

    analysis_outputs = run_exp4_analysis(item_level, ceiling_threshold=ceiling_threshold)
    analysis_outputs["summary_by_prime_condition"].to_csv(
        output_dir / "summary_by_prime_condition_exp4.csv",
        index=False,
    )
    analysis_outputs["summary_by_target_voice"].to_csv(
        output_dir / "summary_by_target_voice_exp4.csv",
        index=False,
    )
    analysis_outputs["summary_by_lexicality"].to_csv(
        output_dir / "summary_by_lexicality_exp4.csv",
        index=False,
    )
    analysis_outputs["summary_baseline_vs_primed"].to_csv(
        output_dir / "summary_baseline_vs_primed_exp4.csv",
        index=False,
    )
    analysis_outputs["ceiling_diagnostics"].to_csv(
        output_dir / "ceiling_diagnostics_exp4.csv",
        index=False,
    )

    question_summary = (
        item_level.groupby(
            [
                "model_name",
                "model_condition",
                "lexicality_condition",
                "question_type",
                "prime_condition",
                "target_voice",
            ],
            as_index=False,
        )
        .agg(
            n_items=("prompt_id", "count"),
            accuracy=("is_correct", "mean"),
            correct_n=("matched_label", lambda values: int((values == "correct").sum())),
            foil_n=("matched_label", lambda values: int((values == "foil").sum())),
            ambiguous_n=("matched_label", lambda values: int((values == "ambiguous").sum())),
            other_n=("matched_label", lambda values: int((values == "other").sum())),
        )
        .sort_values(["lexicality_condition", "question_type", "prime_condition", "target_voice"])
    )
    question_summary.to_csv(output_dir / "summary_by_question_type_exp4.csv", index=False)

    (output_dir / "run_metadata_exp4.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))

    output_dir = args.output_dir.resolve()
    prompt_frame = load_prompt_rows(args)
    prompts = prompt_frame["prompt"].astype(str).tolist()
    print(
        "Experiment 4 selected "
        f"{len(prompt_frame)} prompt rows from {args.prompt_csv.resolve()} "
        f"(max_items={args.max_items}, batch_size={args.batch_size}).",
        flush=True,
    )

    device = get_device(args.device)
    _, model, resolved_dtype = load_causal_lm_and_tokenizer(
        model_name=args.model_name,
        device=device,
        local_files_only=args.local_files_only,
        torch_dtype_name=args.torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    completions = batched_greedy_generate(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=max(1, int(args.batch_size)),
        max_new_tokens=max(1, int(args.max_new_tokens)),
    )

    model_condition = str(args.model_condition).strip() if args.model_condition else Path(output_dir).name
    item_level = make_item_level(
        prompt_frame=prompt_frame,
        completions=completions,
        model_name=args.model_name,
        model_condition=model_condition,
    )

    metadata: Dict[str, object] = {
        "model_name": args.model_name,
        "model_condition": model_condition,
        "prompt_csv": str(args.prompt_csv.resolve()),
        "output_dir": str(output_dir),
        "n_prompt_rows": int(len(prompt_frame)),
        "max_items": int(args.max_items) if args.max_items is not None else None,
        "prime_conditions": list(args.prime_conditions),
        "target_voices": list(args.target_voices),
        "question_types": list(args.question_types),
        "complexity_conditions": list(args.complexity_conditions),
        "batch_size": int(args.batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "device": device,
        "torch_dtype": str(resolved_dtype) if resolved_dtype is not None else "default",
        "local_files_only": bool(args.local_files_only),
        "seed": int(args.seed),
        "ceiling_threshold": float(args.ceiling_threshold),
    }
    write_outputs(
        item_level=item_level,
        output_dir=output_dir,
        ceiling_threshold=float(args.ceiling_threshold),
        metadata=metadata,
    )
    print(f"Saved Experiment 4 complex-NP prompt-generation outputs to {output_dir}")


if __name__ == "__main__":
    main()
