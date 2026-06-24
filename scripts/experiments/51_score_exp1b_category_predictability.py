#!/usr/bin/env python3
"""Diagnose category vs lexical predictability in Exp1B passive baselines.

This is a compact follow-up scoring pass for the passive no-prime baseline.
For each passive target ROI, it scores the next-token distribution and
decomposes the observed token probability into:

    P(token) = P(category) * P(token | category)

The category sets are deliberately corpus-controlled. They are meant as a
diagnostic for the current active/passive materials, not as a full POS tagger.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.production_priming_common import (  # noqa: E402
    get_device,
    load_causal_lm_and_tokenizer,
)


DEFAULT_CORE_CSV = REPO_ROOT / "corpora/transitive/CORE_transitive_strict_4cell_counterbalanced.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "behavioral_results/experiment-1b_category_predictability"

PASSIVE_ROIS = ("patient_np", "aux", "participle", "by", "agent_det", "agent_np")
CATEGORY_ORDER = (
    "PUNCT",
    "DET",
    "AUX",
    "PREP",
    "PARTICIPLE",
    "FINITE_VERB",
    "NOUN",
    "ADJ",
    "ADV",
)
ROI_TO_CATEGORY = {
    "initial_det": "DET",
    "patient_np": "NOUN",
    "aux": "AUX",
    "participle": "PARTICIPLE",
    "by": "PREP",
    "agent_det": "DET",
    "agent_np": "NOUN",
    "punctuation": "PUNCT",
}

DETERMINERS = {"a", "an", "the"}
AUXILIARIES = {"am", "are", "is", "was", "were", "be", "been", "being"}
PREPOSITIONS = {
    "about",
    "above",
    "after",
    "around",
    "as",
    "at",
    "before",
    "below",
    "beside",
    "by",
    "during",
    "for",
    "from",
    "in",
    "inside",
    "into",
    "near",
    "of",
    "on",
    "onto",
    "over",
    "through",
    "to",
    "under",
    "with",
    "without",
}
ADVERBS = {"not", "never", "also", "just", "still", "then", "there", "here"}
ADJECTIVES = {
    "big",
    "small",
    "old",
    "young",
    "new",
    "good",
    "bad",
    "same",
    "other",
    "first",
    "last",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--model-condition", default=None)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CORE_CSV)
    parser.add_argument("--dataset-label", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-items", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--roi",
        nargs="+",
        default=list(PASSIVE_ROIS),
        help=f"Passive ROIs to export. Default: {' '.join(PASSIVE_ROIS)}",
    )
    return parser.parse_args()


def infer_dataset_label(path: Path) -> str:
    name = path.name.lower()
    if "jabberwocky" in name:
        return "jabberwocky"
    if "core" in name:
        return "core"
    return "unspecified"


def normalize_lexicality(dataset_label: str) -> str:
    if dataset_label == "core":
        return "core"
    if dataset_label == "jabberwocky":
        return "jabberwocky"
    return dataset_label


def sentence_words(text: object) -> List[str]:
    return re.findall(r"[a-z]+|[.?!]", str(text or "").lower())


def normalize_token_piece(token: object) -> str:
    text = str(token or "").strip().lower()
    text = text.replace("▁", "").replace("Ġ", "").replace("Ċ", "")
    text = text.replace('"', "").replace("'", "")
    if re.fullmatch(r"[.?!]+", text):
        return text[-1]
    text = re.sub(r"^[^a-z.?!]+|[^a-z.?!]+$", "", text)
    if re.fullmatch(r"[.?!]+", text):
        return text[-1]
    return text


def align_tokens_to_word_indices(tokens: Sequence[object], words: Sequence[str]) -> List[int]:
    word_indices: List[int] = []
    current_word = 0
    partial = ""
    partial_word_index: Optional[int] = None

    for token in tokens:
        piece = normalize_token_piece(token)
        if not piece:
            word_indices.append(-1)
            continue

        if piece in {".", "?", "!"}:
            search_start = partial_word_index if partial_word_index is not None else current_word
            match = next((idx for idx in range(max(0, search_start), len(words)) if words[idx] == piece), -1)
            word_indices.append(match)
            if match >= 0:
                current_word = match + 1
            partial = ""
            partial_word_index = None
            continue

        if partial_word_index is not None:
            candidate = words[partial_word_index]
            extended = partial + piece
            if candidate.startswith(extended):
                word_indices.append(partial_word_index)
                partial = extended
                if partial == candidate:
                    current_word = partial_word_index + 1
                    partial = ""
                    partial_word_index = None
                continue

        match = -1
        for idx in range(current_word, len(words)):
            word = words[idx]
            if word == piece or word.startswith(piece) or piece.startswith(word):
                match = idx
                break

        word_indices.append(match)
        if match >= 0:
            if words[match] == piece or piece.startswith(words[match]):
                current_word = match + 1
                partial = ""
                partial_word_index = None
            else:
                partial = piece
                partial_word_index = match

    return word_indices


def roi_label_for_passive_word(word_index: int) -> str:
    labels = {
        0: "initial_det",
        1: "patient_np",
        2: "aux",
        3: "participle",
        4: "by",
        5: "agent_det",
        6: "agent_np",
        7: "punctuation",
    }
    return labels.get(word_index, "other_passive")


def extract_material_words(frame: pd.DataFrame) -> Dict[str, set[str]]:
    nouns: set[str] = set()
    finite_verbs: set[str] = set()
    participles: set[str] = set()

    for column in ["pa", "ta"]:
        for text in frame[column].astype(str):
            words = sentence_words(text)
            if len(words) >= 5:
                nouns.update([words[1], words[4]])
                finite_verbs.add(words[2])

    for column in ["pp", "tp"]:
        for text in frame[column].astype(str):
            words = sentence_words(text)
            if len(words) >= 7:
                nouns.update([words[1], words[6]])
                participles.add(words[3])
                finite_verbs.add(words[3])

    return {
        "NOUN": {word for word in nouns if word not in DETERMINERS and word not in PREPOSITIONS},
        "FINITE_VERB": finite_verbs,
        "PARTICIPLE": participles,
    }


def candidate_first_token_ids(tokenizer, words: Iterable[str]) -> set[int]:
    token_ids: set[int] = set()
    for word in sorted({str(word).strip().lower() for word in words if str(word).strip()}):
        variants = [word, f" {word}"]
        if word in {".", "?", "!"}:
            variants = [word]
        for variant in variants:
            ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if ids:
                token_ids.add(int(ids[0]))
    return token_ids


def build_category_token_sets(tokenizer, frame: pd.DataFrame) -> Dict[str, set[int]]:
    material = extract_material_words(frame)
    category_words = {
        "PUNCT": {".", "?", "!"},
        "DET": DETERMINERS,
        "AUX": AUXILIARIES,
        "PREP": PREPOSITIONS,
        "PARTICIPLE": material["PARTICIPLE"],
        "FINITE_VERB": material["FINITE_VERB"],
        "NOUN": material["NOUN"],
        "ADJ": ADJECTIVES,
        "ADV": ADVERBS,
    }
    return {
        category: candidate_first_token_ids(tokenizer, words)
        for category, words in category_words.items()
    }


def build_disjoint_id_to_category(category_token_sets: Dict[str, set[int]]) -> Dict[int, str]:
    assigned: Dict[int, str] = {}
    for category in CATEGORY_ORDER:
        for token_id in category_token_sets.get(category, set()):
            assigned.setdefault(int(token_id), category)
    return assigned


def entropy(probabilities: Sequence[float]) -> float:
    total = 0.0
    for prob in probabilities:
        if prob > 0.0:
            total -= float(prob) * math.log(float(prob))
    return total


def category_distribution(
    probs: torch.Tensor,
    id_to_category: Dict[int, str],
) -> Dict[str, float]:
    masses = {category: 0.0 for category in CATEGORY_ORDER}
    for token_id, category in id_to_category.items():
        masses[category] += float(probs[int(token_id)].item())
    classified_mass = sum(masses.values())
    masses["OTHER"] = max(0.0, 1.0 - classified_mass)
    return masses


def conditional_entropy_for_category(
    probs: torch.Tensor,
    token_ids: set[int],
) -> float:
    if not token_ids:
        return float("nan")
    mass = float(probs[list(token_ids)].sum().item())
    if mass <= 0.0:
        return float("nan")
    conditional_probs = (probs[list(token_ids)] / mass).detach().cpu().float().numpy()
    return entropy([float(value) for value in conditional_probs])


def safe_log(value: float) -> float:
    if value <= 0.0 or not np.isfinite(value):
        return float("nan")
    return float(math.log(value))


def prepare_target_rows(frame: pd.DataFrame, max_items: Optional[int]) -> pd.DataFrame:
    if max_items is not None:
        frame = frame.iloc[: int(max_items)].copy()
    rows = []
    for item_index, row in frame.iterrows():
        rows.append(
            {
                "item_index": int(item_index),
                "target_passive": str(row["tp"]).strip(),
                "target_active": str(row["ta"]).strip(),
            }
        )
    return pd.DataFrame(rows)


def score_category_predictability(
    *,
    frame: pd.DataFrame,
    tokenizer,
    model,
    device: str,
    batch_size: int,
    category_token_sets: Dict[str, set[int]],
    selected_rois: set[str],
) -> pd.DataFrame:
    id_to_category = build_disjoint_id_to_category(category_token_sets)
    rows: List[Dict[str, object]] = []
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    records = frame.to_dict(orient="records")
    total_batches = (len(records) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(records), batch_size), start=1):
        batch = records[start: start + batch_size]
        input_rows: List[List[int]] = []
        continuation_ids_list: List[List[int]] = []
        uses_bos_list: List[bool] = []

        for row in batch:
            continuation = " " + str(row["target_passive"]).strip()
            continuation_ids = list(tokenizer(continuation, add_special_tokens=False)["input_ids"])
            if tokenizer.bos_token_id is not None:
                input_ids = [int(tokenizer.bos_token_id)] + continuation_ids
                uses_bos = True
            else:
                input_ids = continuation_ids
                uses_bos = False
            input_rows.append([int(token_id) for token_id in input_ids])
            continuation_ids_list.append([int(token_id) for token_id in continuation_ids])
            uses_bos_list.append(uses_bos)

        max_len = max(len(ids) for ids in input_rows)
        padded = [ids + [int(pad_token_id)] * (max_len - len(ids)) for ids in input_rows]
        attention = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in input_rows]
        input_tensor = torch.tensor(padded, dtype=torch.long, device=device)
        attention_tensor = torch.tensor(attention, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_tensor, attention_mask=attention_tensor).logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            log_probs_all = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            probs_all = torch.exp(log_probs_all)

        print(f"Category predictability batch {batch_number}/{total_batches}", flush=True)

        for row_offset, row in enumerate(batch):
            continuation_ids = continuation_ids_list[row_offset]
            target_text = str(row["target_passive"]).strip()
            target_words = sentence_words(target_text)
            token_strings = tokenizer.convert_ids_to_tokens(continuation_ids)

            if uses_bos_list[row_offset]:
                start_index = 0
                scored_ids = continuation_ids
                scored_tokens = token_strings
            else:
                # Tokenizers without BOS cannot score the first continuation token.
                start_index = 0
                scored_ids = continuation_ids[1:]
                scored_tokens = token_strings[1:]

            word_indices = align_tokens_to_word_indices(scored_tokens, target_words)
            first_token_for_word: Dict[int, int] = {}
            for token_index, word_index in enumerate(word_indices):
                if word_index >= 0 and word_index not in first_token_for_word:
                    first_token_for_word[word_index] = token_index

            for token_index, (token_id, token_string, word_index) in enumerate(
                zip(scored_ids, scored_tokens, word_indices)
            ):
                if word_index < 0:
                    continue
                if first_token_for_word.get(word_index) != token_index:
                    continue

                roi = roi_label_for_passive_word(word_index)
                if roi not in selected_rois:
                    continue
                target_category = ROI_TO_CATEGORY.get(roi, "OTHER")
                logits_index = start_index + token_index
                log_probs = log_probs_all[row_offset, logits_index]
                probs = probs_all[row_offset, logits_index]

                actual_logprob = float(log_probs[int(token_id)].item())
                actual_probability = float(probs[int(token_id)].item())
                masses = category_distribution(probs, id_to_category)
                target_category_probability = float(masses.get(target_category, 0.0))
                target_category_surprisal = -safe_log(target_category_probability)
                category_entropy = entropy(list(masses.values()))

                category_ids = category_token_sets.get(target_category, set())
                in_category = int(token_id) in category_ids
                if in_category and target_category_probability > 0.0:
                    within_probability = actual_probability / target_category_probability
                    within_surprisal = -safe_log(within_probability)
                else:
                    within_probability = float("nan")
                    within_surprisal = float("nan")
                within_entropy = conditional_entropy_for_category(probs, category_ids)

                same_word_token_indices = [
                    idx for idx, aligned_word_index in enumerate(word_indices)
                    if aligned_word_index == word_index
                ]
                word_logprob_sum = float(
                    sum(float(log_probs_all[row_offset, start_index + idx, int(scored_ids[idx])].item())
                        for idx in same_word_token_indices)
                )

                rows.append(
                    {
                        "item_index": int(row["item_index"]),
                        "target_passive": target_text,
                        "target_active": row.get("target_active", ""),
                        "roi": roi,
                        "word_index": int(word_index),
                        "word": target_words[word_index] if word_index < len(target_words) else "",
                        "token_index": int(token_index),
                        "token_id": int(token_id),
                        "token": token_string,
                        "target_category": target_category,
                        "actual_token_probability": actual_probability,
                        "actual_token_logprob": actual_logprob,
                        "actual_token_surprisal": -actual_logprob,
                        "target_category_probability": target_category_probability,
                        "target_category_logprob": safe_log(target_category_probability),
                        "target_category_surprisal": target_category_surprisal,
                        "category_entropy": category_entropy,
                        "category_effective_n": math.exp(category_entropy),
                        "within_category_probability": within_probability,
                        "within_category_logprob": safe_log(within_probability),
                        "within_category_surprisal": within_surprisal,
                        "within_category_entropy": within_entropy,
                        "within_category_effective_n": math.exp(within_entropy) if np.isfinite(within_entropy) else np.nan,
                        "target_token_in_category_set": in_category,
                        "category_candidate_token_count": len(category_ids),
                        "word_logprob_sum": word_logprob_sum,
                        "word_surprisal_sum": -word_logprob_sum,
                        "word_token_count": len(same_word_token_indices),
                        "category_probability_json": json.dumps(masses, ensure_ascii=True),
                    }
                )

    return pd.DataFrame(rows)


def summarize(item_level: pd.DataFrame) -> pd.DataFrame:
    if item_level.empty:
        return pd.DataFrame()
    return (
        item_level.groupby(
            ["model_condition", "dataset", "lexicality_condition", "roi", "target_category"],
            as_index=False,
        )
        .agg(
            n_items=("item_index", "count"),
            actual_token_logprob_mean=("actual_token_logprob", "mean"),
            actual_token_surprisal_mean=("actual_token_surprisal", "mean"),
            target_category_probability_mean=("target_category_probability", "mean"),
            target_category_surprisal_mean=("target_category_surprisal", "mean"),
            category_entropy_mean=("category_entropy", "mean"),
            category_effective_n_mean=("category_effective_n", "mean"),
            within_category_probability_mean=("within_category_probability", "mean"),
            within_category_surprisal_mean=("within_category_surprisal", "mean"),
            within_category_entropy_mean=("within_category_entropy", "mean"),
            within_category_effective_n_mean=("within_category_effective_n", "mean"),
            word_surprisal_sum_mean=("word_surprisal_sum", "mean"),
            target_token_in_category_set_rate=("target_token_in_category_set", "mean"),
        )
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_label = args.dataset_label or infer_dataset_label(input_csv)
    selected_rois = set(args.roi)

    frame = pd.read_csv(input_csv)
    frame.columns = frame.columns.str.strip().str.lower()
    required = {"pa", "pp", "ta", "tp"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required corpus columns: {sorted(missing)}")

    target_rows = prepare_target_rows(frame, max_items=args.max_items)
    device = get_device(args.device)
    tokenizer, model, resolved_dtype = load_causal_lm_and_tokenizer(
        model_name=args.model_name,
        device=device,
        local_files_only=args.local_files_only,
        torch_dtype_name=args.torch_dtype,
    )

    category_token_sets = build_category_token_sets(tokenizer, frame)
    item_level = score_category_predictability(
        frame=target_rows,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=max(1, int(args.batch_size)),
        category_token_sets=category_token_sets,
        selected_rois=selected_rois,
    )
    item_level["model_name"] = args.model_name
    item_level["model_condition"] = args.model_condition or args.model_name
    item_level["dataset"] = dataset_label
    item_level["lexicality_condition"] = normalize_lexicality(dataset_label)
    item_level["experiment"] = "experiment_1b"
    item_level["task_type"] = "passive_baseline_category_predictability"
    item_level["prime_condition"] = "no_prime"
    item_level["target_voice"] = "passive"
    item_level["prompt_format_used"] = "plain_text"
    item_level["category_set_policy"] = "corpus_controlled_first_token_sets_with_other_mass"

    item_path = output_dir / "category_predictability_item_level.csv"
    summary_path = output_dir / "category_predictability_summary.csv"
    metadata_path = output_dir / "run_metadata_category_predictability.json"
    category_path = output_dir / "category_token_set_summary.csv"

    item_level.to_csv(item_path, index=False)
    summarize(item_level).to_csv(summary_path, index=False)

    category_summary = pd.DataFrame(
        [
            {
                "category": category,
                "candidate_token_count": len(token_ids),
            }
            for category, token_ids in category_token_sets.items()
        ]
    )
    category_summary.to_csv(category_path, index=False)

    metadata = {
        "model_name": args.model_name,
        "model_condition": args.model_condition or args.model_name,
        "dataset_label": dataset_label,
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "max_items": args.max_items,
        "batch_size": args.batch_size,
        "device": device,
        "torch_dtype": str(resolved_dtype),
        "selected_rois": sorted(selected_rois),
        "category_order": list(CATEGORY_ORDER),
        "category_set_policy": "corpus_controlled_first_token_sets_with_other_mass",
        "notes": (
            "Category probabilities are first-token candidate-set diagnostics. "
            "Closed-class ROIs are close to full-category estimates; noun and "
            "participle categories are restricted to the controlled corpus vocabulary."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote: {item_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {category_path}")
    print(f"Wrote: {metadata_path}")


if __name__ == "__main__":
    main()
