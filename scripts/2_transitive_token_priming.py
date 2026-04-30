import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "behavioral_results" / "experiment-1" / "experiment-1a" / "transitive_token_profiles"
LOCAL_PAPER_INPUTS = {
    "CORE": REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_strict_4cell_counterbalanced.csv",
    "jabberwocky": REPO_ROOT / "corpora" / "transitive" / "jabberwocky_transitive_gpt2_monosyllabic_strict_4cell.csv",
}
PRIMELM_ROOT = REPO_ROOT / "PrimeLM" / "corpora"
PRESETS = {
    "paper_main": LOCAL_PAPER_INPUTS,
    "primelm_core": {
        "CORE": PRIMELM_ROOT / "CORE_transitive_15000sampled_10-1.csv",
        "ANOMALOUS": PRIMELM_ROOT / "ANOMALOUS_chomsky_transitive_15000sampled_10-1.csv",
    },
    "primelm_recency": {
        "RECENCY_5": PRIMELM_ROOT / "RECENCY_5_transitive_15000sampled_10-1.csv",
    },
    "primelm_cumulative": {
        "CUMULATIVE_5": PRIMELM_ROOT / "CUMULATIVE_5_transitive_15000sampled_10-1.csv",
    },
    "primelm_semsim": {
        "SEMSIM_Vonly": PRIMELM_ROOT / "SEMSIM_Vonly_transitive_15000sampled_10-1.csv",
        "SEMSIM_Nall": PRIMELM_ROOT / "SEMSIM_Nall_transitive_15000sampled_10-1.csv",
    },
}


@dataclass
class TargetScores:
    token_ids: List[int]
    token_strings: List[str]
    token_log_probs: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute jabberwocky_transitive structural priming traces for active/passive targets."
    )
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Torch dtype for model loading: auto, float32, float16, or bfloat16.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer/model from local Hugging Face cache only.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="paper_main",
        help="Named corpus bundle to analyze.",
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        default=[],
        help="Extra corpus spec in the form label=/abs/or/relative/path.csv. Can be repeated.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from completed per-condition checkpoints in output-dir.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore checkpoints and recompute all conditions from scratch.",
    )
    return parser.parse_args()


def normalize_transitive_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = frame.columns.str.strip().str.lower()
    expected = {"pa", "pp", "ta", "tp"}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame[list(sorted(expected))]


def get_device(user_device: Optional[str]) -> str:
    if user_device:
        return user_device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_torch_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    normalized = (dtype_name or "auto").strip().lower()
    if normalized in {"", "auto", "none", "default"}:
        return None
    lookup = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in lookup:
        raise ValueError(
            f"Unsupported torch dtype '{dtype_name}'. Use auto, float32, float16, or bfloat16."
        )
    return lookup[normalized]


def parse_input_specs(raw_specs: Sequence[str]) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for spec in raw_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --input-csv value: {spec}")
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip())
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        parsed[label] = path
    return parsed


def resolve_inputs(args: argparse.Namespace) -> Dict[str, Path]:
    inputs = dict(PRESETS[args.preset])
    inputs.update(parse_input_specs(args.input_csv))
    for label, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Input for {label} not found: {path}")
    return inputs


def token_start_for_word(tokenizer, text: str, word_index: int) -> int:
    words = text.strip().split()
    prefix = " " + " ".join(words[:word_index])
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    return len(prefix_ids)


def tokenize_target_with_offsets(tokenizer, text: str) -> Tuple[List[int], List[str], List[Tuple[int, int]]]:
    prefixed_text = " " + text
    encoded = tokenizer(
        prefixed_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = encoded["input_ids"]
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)
    offsets = [
        (max(0, start - 1), max(0, end - 1))
        for start, end in encoded["offset_mapping"]
    ]
    return token_ids, token_strings, offsets


def target_word_spans(text: str) -> List[Tuple[int, int]]:
    return [match.span() for match in re.finditer(r"\S+", text)]


def assign_tokens_to_words(
    text: str,
    offsets: Sequence[Tuple[int, int]],
) -> List[int]:
    spans = target_word_spans(text)
    word_indices: List[int] = []
    current_word = 0

    for start, end in offsets:
        while current_word < len(spans) and start >= spans[current_word][1]:
            current_word += 1

        if current_word >= len(spans):
            raise ValueError(f"Could not align token offset {(start, end)} to a word in: {text}")

        span_start, span_end = spans[current_word]
        if end <= span_start:
            raise ValueError(f"Token offset {(start, end)} falls before aligned word span {(span_start, span_end)} in: {text}")

        word_indices.append(current_word)

    return word_indices


def word_level_scores(
    tokenizer,
    text: str,
    token_ids: Sequence[int],
    token_pe: Sequence[float],
) -> List[Dict[str, object]]:
    offset_token_ids, _, offsets = tokenize_target_with_offsets(tokenizer, text)
    if list(token_ids) != list(offset_token_ids):
        raise ValueError(f"Target tokenization mismatch while building word-level scores for: {text}")

    word_indices = assign_tokens_to_words(text, offsets)
    words = text.strip().split()
    word_rows: List[Dict[str, object]] = []

    for word_index, word in enumerate(words):
        token_positions = [idx for idx, assigned_word in enumerate(word_indices) if assigned_word == word_index]
        if not token_positions:
            raise ValueError(f"Word {word_index} has no aligned tokens in: {text}")

        start = token_positions[0]
        end = token_positions[-1] + 1
        word_rows.append(
            {
                "word_index": word_index,
                "word": word,
                "token_start": start,
                "token_end": end,
                "token_count": end - start,
                "word_pe": float(sum(token_pe[start:end])),
                "word_pe_mean": float(sum(token_pe[start:end]) / (end - start)),
            }
        )

    return word_rows


def first_divergence_index(tokenizer, text_a: str, text_b: str) -> int:
    ids_a = tokenizer.encode(" " + text_a, add_special_tokens=False)
    ids_b = tokenizer.encode(" " + text_b, add_special_tokens=False)
    for idx, (tok_a, tok_b) in enumerate(zip(ids_a, ids_b)):
        if tok_a != tok_b:
            return idx
    return min(len(ids_a), len(ids_b))


def score_batch(
    tokenizer,
    model,
    device: str,
    batch_primes: Sequence[str],
    batch_targets: Sequence[str],
) -> List[TargetScores]:
    full_texts = [prime + " " + target for prime, target in zip(batch_primes, batch_targets)]
    inputs = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
        add_special_tokens=False,
        return_offsets_mapping=True,
    ).to(device)
    offset_mappings = inputs.pop("offset_mapping").cpu().tolist()

    with torch.no_grad():
        logits = model(**inputs).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs.input_ids[:, 1:].contiguous()
    log_probs_all = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    observed_log_probs = log_probs_all.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

    results: List[TargetScores] = []
    for row_idx, (prime, target) in enumerate(zip(batch_primes, batch_targets)):
        target_start = len(prime) + 1
        valid_token_count = int(inputs.attention_mask[row_idx].sum().item())
        target_token_positions = [
            position
            for position, (start, end) in enumerate(offset_mappings[row_idx][:valid_token_count])
            # Fast tokenizers may attach the separating space to the first target
            # token, so the token can start just before the target string itself.
            if end > target_start
        ]
        if not target_token_positions:
            raise ValueError(f"Could not locate target span in tokenized prompt: {target}")
        if target_token_positions[0] == 0:
            raise ValueError(f"Target begins at token position 0; cannot score first token for: {target}")

        score_positions = [position - 1 for position in target_token_positions]

        row_target_ids = inputs.input_ids[row_idx, target_token_positions].tolist()
        row_token_strings = tokenizer.convert_ids_to_tokens(row_target_ids)
        row_log_probs = observed_log_probs[row_idx, score_positions].tolist()

        results.append(
            TargetScores(
                token_ids=row_target_ids,
                token_strings=row_token_strings,
                token_log_probs=[float(value) for value in row_log_probs],
            )
        )

    return results


def build_item_record(
    condition: str,
    item_index: int,
    target_structure: str,
    target_text: str,
    paired_target_text: str,
    congruent_scores: TargetScores,
    incongruent_scores: TargetScores,
    tokenizer,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    if congruent_scores.token_ids != incongruent_scores.token_ids:
        raise ValueError(
            f"Target token mismatch for {condition} item {item_index} {target_structure}."
        )

    token_pe = [
        cong - inc
        for cong, inc in zip(congruent_scores.token_log_probs, incongruent_scores.token_log_probs)
    ]
    divergence_start = first_divergence_index(tokenizer, target_text, paired_target_text)
    structure_start = token_start_for_word(tokenizer, target_text, word_index=2)
    post_divergence_tokens = token_pe[divergence_start:]
    structure_region_tokens = token_pe[structure_start:]
    word_rows = word_level_scores(tokenizer, target_text, congruent_scores.token_ids, token_pe)
    critical_word_index = 2
    critical_word_row = word_rows[critical_word_index]

    item_record = {
        "condition": condition,
        "item_index": item_index,
        "target_structure": target_structure,
        "target_text": target_text,
        "paired_target_text": paired_target_text,
        "target_length": len(token_pe),
        "divergence_start": divergence_start,
        "structure_start": structure_start,
        "sentence_pe": float(sum(token_pe)),
        "sentence_pe_mean": float(sum(token_pe) / len(token_pe)),
        "critical_word_index": critical_word_index,
        "critical_word": critical_word_row["word"],
        "critical_word_token_start": critical_word_row["token_start"],
        "critical_word_token_end": critical_word_row["token_end"],
        "critical_word_token_count": critical_word_row["token_count"],
        "critical_word_pe": critical_word_row["word_pe"],
        "critical_word_pe_mean": critical_word_row["word_pe_mean"],
        "post_divergence_pe": float(sum(post_divergence_tokens)),
        "post_divergence_token_count": len(post_divergence_tokens),
        "post_divergence_pe_mean": float(sum(post_divergence_tokens) / len(post_divergence_tokens)),
        "pre_divergence_pe": float(sum(token_pe[:divergence_start])),
        "structure_region_pe": float(sum(structure_region_tokens)),
        "structure_region_token_count": len(structure_region_tokens),
        "structure_region_pe_mean": float(sum(structure_region_tokens) / len(structure_region_tokens)),
        "token_ids_json": json.dumps(congruent_scores.token_ids),
        "token_strings_json": json.dumps(congruent_scores.token_strings),
        "token_pe_json": json.dumps(token_pe),
        "congruent_log_probs_json": json.dumps(congruent_scores.token_log_probs),
        "incongruent_log_probs_json": json.dumps(incongruent_scores.token_log_probs),
    }

    token_records: List[Dict[str, object]] = []
    for token_index, token_id in enumerate(congruent_scores.token_ids):
        token_records.append(
            {
                "condition": condition,
                "item_index": item_index,
                "target_structure": target_structure,
                "target_text": target_text,
                "token_index": token_index,
                "token_id": token_id,
                "token_string": congruent_scores.token_strings[token_index],
                "congruent_log_prob": congruent_scores.token_log_probs[token_index],
                "incongruent_log_prob": incongruent_scores.token_log_probs[token_index],
                "token_pe": token_pe[token_index],
                "is_post_divergence": token_index >= divergence_start,
                "is_structure_region": token_index >= structure_start,
                "word_index": next(
                    row["word_index"]
                    for row in word_rows
                    if row["token_start"] <= token_index < row["token_end"]
                ),
                "is_critical_word": critical_word_row["token_start"] <= token_index < critical_word_row["token_end"],
            }
        )

    return item_record, token_records


def summarize_items(item_frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        item_frame.groupby(["condition", "target_structure"], as_index=False)
        .agg(
            n_items=("item_index", "count"),
            mean_sentence_pe=("sentence_pe", "mean"),
            sd_sentence_pe=("sentence_pe", "std"),
            mean_sentence_pe_mean=("sentence_pe_mean", "mean"),
            sd_sentence_pe_mean=("sentence_pe_mean", "std"),
            mean_critical_word_pe=("critical_word_pe", "mean"),
            sd_critical_word_pe=("critical_word_pe", "std"),
            mean_critical_word_pe_mean=("critical_word_pe_mean", "mean"),
            sd_critical_word_pe_mean=("critical_word_pe_mean", "std"),
            mean_post_divergence_pe=("post_divergence_pe", "mean"),
            sd_post_divergence_pe=("post_divergence_pe", "std"),
            mean_post_divergence_pe_mean=("post_divergence_pe_mean", "mean"),
            sd_post_divergence_pe_mean=("post_divergence_pe_mean", "std"),
            mean_pre_divergence_pe=("pre_divergence_pe", "mean"),
            sd_pre_divergence_pe=("pre_divergence_pe", "std"),
            mean_structure_region_pe=("structure_region_pe", "mean"),
            sd_structure_region_pe=("structure_region_pe", "std"),
            mean_structure_region_pe_mean=("structure_region_pe_mean", "mean"),
            sd_structure_region_pe_mean=("structure_region_pe_mean", "std"),
        )
    )
    return summary


def summarize_tokens(token_frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        token_frame.groupby(["condition", "target_structure", "token_index"], as_index=False)
        .agg(
            n_items=("item_index", "count"),
            mean_token_pe=("token_pe", "mean"),
            sd_token_pe=("token_pe", "std"),
            mean_congruent_log_prob=("congruent_log_prob", "mean"),
            mean_incongruent_log_prob=("incongruent_log_prob", "mean"),
            post_divergence_rate=("is_post_divergence", "mean"),
            structure_region_rate=("is_structure_region", "mean"),
        )
    )
    return summary


def build_word_frame(token_frame: pd.DataFrame) -> pd.DataFrame:
    word_frame = (
        token_frame.groupby(
            ["condition", "target_structure", "item_index", "target_text", "word_index"],
            as_index=False,
        )
        .agg(
            token_count=("token_index", "count"),
            word_pe=("token_pe", "sum"),
            word_pe_mean=("token_pe", "mean"),
            is_critical_word=("is_critical_word", "max"),
        )
    )
    word_frame["word"] = word_frame.apply(
        lambda row: str(row["target_text"]).split()[int(row["word_index"])],
        axis=1,
    )
    return word_frame


def summarize_words(word_frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        word_frame.groupby(["condition", "target_structure", "word_index"], as_index=False)
        .agg(
            n_items=("item_index", "count"),
            word=("word", "first"),
            mean_word_pe=("word_pe", "mean"),
            sd_word_pe=("word_pe", "std"),
            mean_word_pe_mean=("word_pe_mean", "mean"),
            sd_word_pe_mean=("word_pe_mean", "std"),
            critical_word_rate=("is_critical_word", "mean"),
        )
    )
    return summary


def condition_slug(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label.strip())
    return slug or "condition"


def condition_dir(output_dir: Path, label: str) -> Path:
    return output_dir / "conditions" / condition_slug(label)


def condition_paths(output_dir: Path, label: str) -> Dict[str, Path]:
    base = condition_dir(output_dir, label)
    return {
        "base": base,
        "item": base / "transitive_item_level_scores.csv",
        "token": base / "transitive_token_level_scores.csv",
        "word": base / "transitive_word_level_scores.csv",
        "item_summary": base / "transitive_item_summary.csv",
        "token_summary": base / "transitive_token_summary.csv",
        "word_summary": base / "transitive_word_summary.csv",
        "marker": base / "done.json",
    }


def condition_is_complete(paths: Dict[str, Path]) -> bool:
    required = ["item", "token", "word", "marker"]
    return all(paths[name].exists() for name in required)


def write_condition_outputs(
    condition: str,
    output_dir: Path,
    item_rows: List[Dict[str, object]],
    token_rows: List[Dict[str, object]],
) -> Dict[str, Path]:
    paths = condition_paths(output_dir, condition)
    paths["base"].mkdir(parents=True, exist_ok=True)

    item_frame = pd.DataFrame(item_rows)
    token_frame = pd.DataFrame(token_rows)
    word_frame = build_word_frame(token_frame)

    item_frame.to_csv(paths["item"], index=False)
    token_frame.to_csv(paths["token"], index=False)
    word_frame.to_csv(paths["word"], index=False)
    summarize_items(item_frame).to_csv(paths["item_summary"], index=False)
    summarize_tokens(token_frame).to_csv(paths["token_summary"], index=False)
    summarize_words(word_frame).to_csv(paths["word_summary"], index=False)

    marker_payload = {
        "condition": condition,
        "n_item_rows": int(len(item_frame)),
        "n_token_rows": int(len(token_frame)),
        "n_word_rows": int(len(word_frame)),
    }
    paths["marker"].write_text(json.dumps(marker_payload, indent=2) + "\n")
    return paths


def merge_condition_outputs(output_dir: Path, condition_labels: Sequence[str]) -> None:
    item_frames: List[pd.DataFrame] = []
    token_frames: List[pd.DataFrame] = []
    word_frames: List[pd.DataFrame] = []

    for label in condition_labels:
        paths = condition_paths(output_dir, label)
        if not condition_is_complete(paths):
            raise FileNotFoundError(f"Missing checkpoint outputs for condition: {label}")
        item_frames.append(pd.read_csv(paths["item"]))
        token_frames.append(pd.read_csv(paths["token"]))
        word_frames.append(pd.read_csv(paths["word"]))

    item_frame = pd.concat(item_frames, ignore_index=True)
    token_frame = pd.concat(token_frames, ignore_index=True)
    word_frame = pd.concat(word_frames, ignore_index=True)

    item_frame.to_csv(output_dir / "transitive_item_level_scores.csv", index=False)
    token_frame.to_csv(output_dir / "transitive_token_level_scores.csv", index=False)
    word_frame.to_csv(output_dir / "transitive_word_level_scores.csv", index=False)
    summarize_items(item_frame).to_csv(output_dir / "transitive_item_summary.csv", index=False)
    summarize_tokens(token_frame).to_csv(output_dir / "transitive_token_summary.csv", index=False)
    summarize_words(word_frame).to_csv(output_dir / "transitive_word_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    input_map = resolve_inputs(args)

    device = get_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
    }
    resolved_dtype = resolve_torch_dtype(args.torch_dtype)
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.to(device)
    model.eval()

    ordered_labels = list(input_map.keys())

    for condition, input_path in input_map.items():
        paths = condition_paths(output_dir, condition)
        if args.resume and condition_is_complete(paths):
            print(f"Skipping {condition} (checkpoint exists at {paths['base']})")
            continue

        frame = pd.read_csv(input_path)
        frame = normalize_transitive_frame(frame)
        if args.max_items is not None:
            frame = frame.iloc[: args.max_items].reset_index(drop=True)

        condition_item_rows: List[Dict[str, object]] = []
        condition_token_rows: List[Dict[str, object]] = []

        for batch_start in tqdm(
            range(0, len(frame), args.batch_size),
            desc=f"{condition} batches",
        ):
            batch = frame.iloc[batch_start: batch_start + args.batch_size]

            active_congruent = score_batch(
                tokenizer,
                model,
                device,
                batch["pa"].tolist(),
                batch["ta"].tolist(),
            )
            active_incongruent = score_batch(
                tokenizer,
                model,
                device,
                batch["pp"].tolist(),
                batch["ta"].tolist(),
            )

            passive_congruent = score_batch(
                tokenizer,
                model,
                device,
                batch["pp"].tolist(),
                batch["tp"].tolist(),
            )
            passive_incongruent = score_batch(
                tokenizer,
                model,
                device,
                batch["pa"].tolist(),
                batch["tp"].tolist(),
            )

            for offset, row in enumerate(batch.itertuples(index=True)):
                item_index = int(row.Index)

                active_item, active_tokens = build_item_record(
                    condition=condition,
                    item_index=item_index,
                    target_structure="active",
                    target_text=row.ta,
                    paired_target_text=row.tp,
                    congruent_scores=active_congruent[offset],
                    incongruent_scores=active_incongruent[offset],
                    tokenizer=tokenizer,
                )
                passive_item, passive_tokens = build_item_record(
                    condition=condition,
                    item_index=item_index,
                    target_structure="passive",
                    target_text=row.tp,
                    paired_target_text=row.ta,
                    congruent_scores=passive_congruent[offset],
                    incongruent_scores=passive_incongruent[offset],
                    tokenizer=tokenizer,
                )

                condition_item_rows.extend([active_item, passive_item])
                condition_token_rows.extend(active_tokens)
                condition_token_rows.extend(passive_tokens)

        write_condition_outputs(
            condition=condition,
            output_dir=output_dir,
            item_rows=condition_item_rows,
            token_rows=condition_token_rows,
        )

    merge_condition_outputs(output_dir, ordered_labels)


if __name__ == "__main__":
    main()
