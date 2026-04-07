import argparse
import csv
import difflib
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


WORD_RE = re.compile(r"[A-Za-z']+")
ITEM_ID_RE = re.compile(r"^(\d+)([A-Za-z]+)$")


@dataclass(frozen=True)
class ScoreTask:
    prefix: str
    candidate: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score target-word fit in sentence and dialogue contexts.")
    parser.add_argument("--input-csv", type=Path, default=Path("transprob_df1.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("transprob_df1_scored.csv"))
    parser.add_argument("--model-name", default="gpt2-medium")
    parser.add_argument("--bert-model-name", default="bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def pick_device(user_device: Optional[str]) -> str:
    if user_device:
        return user_device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def split_item_id(item_id: str) -> Tuple[str, str]:
    match = ITEM_ID_RE.match(item_id.strip())
    if not match:
        raise ValueError(f"Unexpected item_id format: {item_id}")
    return match.group(1), match.group(2)


def longest_shared_prefix(text_a: str, text_b: str) -> str:
    prefix = text_a[: len(text_a) if text_a == text_b else len(os.path.commonprefix([text_a, text_b]))]
    if prefix and not prefix[-1].isspace():
        whitespace_positions = [idx for idx, char in enumerate(prefix) if char.isspace()]
        prefix = prefix[: whitespace_positions[-1] + 1] if whitespace_positions else ""
    return prefix


def extract_surface_candidate(text: str, prefix: str) -> str:
    remainder = text[len(prefix) :]
    match = WORD_RE.match(remainder)
    if not match:
        raise ValueError(f"Could not extract a candidate word after prefix from: {text!r}")
    return match.group(0)


def make_score_task(prefix: str, surface_form: str) -> ScoreTask:
    trimmed_prefix = prefix.rstrip()
    boundary = prefix[len(trimmed_prefix) :]
    return ScoreTask(prefix=trimmed_prefix, candidate=boundary + surface_form)


def similarity_score(target_word: str, candidate_surface: str) -> float:
    target = target_word.lower()
    candidate = candidate_surface.lower()
    if target == candidate:
        return 10.0
    if candidate.startswith(target) or target.startswith(candidate):
        return 5.0
    return difflib.SequenceMatcher(None, target, candidate).ratio()


def assign_surface_forms(target_words: Sequence[str], surface_forms: Sequence[str]) -> Dict[str, str]:
    if len(target_words) != 2 or len(surface_forms) != 2:
        raise ValueError("assign_surface_forms expects exactly two target words and two surface forms")

    tw0, tw1 = target_words
    sf0, sf1 = surface_forms

    direct = similarity_score(tw0, sf0) + similarity_score(tw1, sf1)
    crossed = similarity_score(tw0, sf1) + similarity_score(tw1, sf0)

    if direct == crossed:
        raise ValueError(
            f"Could not disambiguate target/surface mapping for targets {target_words} and surfaces {surface_forms}"
        )

    if direct > crossed:
        return {tw0: sf0, tw1: sf1}
    return {tw0: sf1, tw1: sf0}


def prepare_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_pair_metadata(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        pair_id, _ = split_item_id(row["item_id"])
        grouped[pair_id].append(row)

    pair_metadata: Dict[str, Dict[str, str]] = {}
    for pair_id, pair_rows in grouped.items():
        if len(pair_rows) != 2:
            raise ValueError(f"Expected exactly two rows for pair {pair_id}, found {len(pair_rows)}")

        sentence_prefix = longest_shared_prefix(pair_rows[0]["sentence"], pair_rows[1]["sentence"])
        dialogue_prefix = longest_shared_prefix(pair_rows[0]["dialogue"], pair_rows[1]["dialogue"])
        sentence_surfaces = [
            extract_surface_candidate(pair_rows[0]["sentence"], sentence_prefix),
            extract_surface_candidate(pair_rows[1]["sentence"], sentence_prefix),
        ]
        surface_map = assign_surface_forms(
            [pair_rows[0]["target_word"], pair_rows[1]["target_word"]],
            sentence_surfaces,
        )

        pair_metadata[pair_id] = {
            "sentence_prefix": sentence_prefix,
            "dialogue_prefix": dialogue_prefix,
            pair_rows[0]["target_word"]: surface_map[pair_rows[0]["target_word"]],
            pair_rows[1]["target_word"]: surface_map[pair_rows[1]["target_word"]],
        }

    return pair_metadata


def build_tasks(
    rows: Sequence[Dict[str, str]],
    pair_metadata: Dict[str, Dict[str, str]],
) -> List[ScoreTask]:
    tasks = set()
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        pair_id, _ = split_item_id(row["item_id"])
        grouped[pair_id].append(row)

    for pair_id, pair_rows in grouped.items():
        meta = pair_metadata[pair_id]
        target_words = [row["target_word"] for row in pair_rows]
        surface_forms = [meta[word] for word in target_words]
        for prefix_key in ("sentence_prefix", "dialogue_prefix"):
            prefix = meta[prefix_key]
            for surface_form in surface_forms:
                tasks.add(make_score_task(prefix, surface_form))

    return sorted(tasks, key=lambda task: (task.prefix, task.candidate))


def batch_score_tasks(
    tasks: Sequence[ScoreTask],
    tokenizer,
    model,
    device: str,
    batch_size: int,
) -> Dict[ScoreTask, Tuple[float, int]]:
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id
    if bos_token_id is None:
        raise ValueError("Tokenizer must provide either bos_token_id or eos_token_id")

    task_scores: Dict[ScoreTask, Tuple[float, int]] = {}
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        input_rows: List[List[int]] = []
        attention_rows: List[List[int]] = []
        candidate_starts: List[int] = []
        candidate_lengths: List[int] = []
        max_len = 0

        for task in batch:
            prefix_ids = tokenizer.encode(task.prefix, add_special_tokens=False)
            full_ids = tokenizer.encode(task.prefix + task.candidate, add_special_tokens=False)
            candidate_start = len(prefix_ids)
            candidate_len = len(full_ids) - candidate_start
            if candidate_len <= 0:
                raise ValueError(f"Candidate tokenization is empty for {task}")

            input_ids = [bos_token_id] + full_ids
            max_len = max(max_len, len(input_ids))
            input_rows.append(input_ids)
            candidate_starts.append(candidate_start)
            candidate_lengths.append(candidate_len)

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must provide either pad_token_id or eos_token_id")

        for idx, input_ids in enumerate(input_rows):
            pad_len = max_len - len(input_ids)
            input_rows[idx] = input_ids + [pad_id] * pad_len
            attention_rows.append([1] * len(input_ids) + [0] * pad_len)

        input_tensor = torch.tensor(input_rows, dtype=torch.long, device=device)
        attention_tensor = torch.tensor(attention_rows, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_tensor, attention_mask=attention_tensor).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_tensor[:, 1:].contiguous()
        log_probs_all = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        observed_log_probs = log_probs_all.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

        for row_idx, task in enumerate(batch):
            start_idx = candidate_starts[row_idx]
            end_idx = start_idx + candidate_lengths[row_idx]
            token_log_probs = observed_log_probs[row_idx, start_idx:end_idx]
            task_scores[task] = (float(token_log_probs.sum().item()), candidate_lengths[row_idx])

    return task_scores


def score_rows(
    rows: Sequence[Dict[str, str]],
    pair_metadata: Dict[str, Dict[str, str]],
    task_scores: Dict[ScoreTask, Tuple[float, int]],
    bert_fit_scores: Dict[str, float],
) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        pair_id, _ = split_item_id(row["item_id"])
        grouped[pair_id].append(row)

    scored_rows: List[Dict[str, str]] = []
    for pair_id, pair_rows in grouped.items():
        if len(pair_rows) != 2:
            raise ValueError(f"Expected exactly two rows for pair {pair_id}, found {len(pair_rows)}")

        meta = pair_metadata[pair_id]
        target_words = [pair_rows[0]["target_word"], pair_rows[1]["target_word"]]
        competitor_map = {
            target_words[0]: target_words[1],
            target_words[1]: target_words[0],
        }

        sentence_prefix = meta["sentence_prefix"]
        dialogue_prefix = meta["dialogue_prefix"]

        for row in pair_rows:
            _, status = split_item_id(row["item_id"])
            target_word = row["target_word"]
            competitor_word = competitor_map[target_word]
            target_surface = meta[target_word]
            competitor_surface = meta[competitor_word]

            sentence_logprob, sentence_token_count = task_scores[make_score_task(sentence_prefix, target_surface)]
            dialogue_logprob, dialogue_token_count = task_scores[make_score_task(dialogue_prefix, target_surface)]
            sentence_comp_logprob, _ = task_scores[make_score_task(sentence_prefix, competitor_surface)]
            dialogue_comp_logprob, _ = task_scores[make_score_task(dialogue_prefix, competitor_surface)]

            scored_row = dict(row)
            scored_row.update(
                {
                    "pair_id": pair_id,
                    "target_status": "matched" if status.upper() == "C" else "mismatched",
                    "competitor_word": competitor_word,
                    "scored_target_form": target_surface,
                    "scored_competitor_form": competitor_surface,
                    "sentence_prefix": sentence_prefix,
                    "dialogue_prefix": dialogue_prefix,
                    "sentence_logprob": f"{sentence_logprob:.6f}",
                    "dialogue_logprob": f"{dialogue_logprob:.6f}",
                    "sentence_surprisal": f"{-sentence_logprob:.6f}",
                    "dialogue_surprisal": f"{-dialogue_logprob:.6f}",
                    "sentence_delta_logprob": f"{sentence_logprob - sentence_comp_logprob:.6f}",
                    "dialogue_delta_logprob": f"{dialogue_logprob - dialogue_comp_logprob:.6f}",
                    "sentence_target_token_count": str(sentence_token_count),
                    "dialogue_target_token_count": str(dialogue_token_count),
                    "bert_full_context_fit": f"{bert_fit_scores[row['item_id']]:.6f}",
                }
            )
            scored_rows.append(scored_row)

    return scored_rows


def write_rows(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_model_and_tokenizer(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def load_masked_model_and_tokenizer(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def find_surface_span(text: str, prefix: str, surface_form: str) -> Tuple[int, int]:
    start = len(prefix)
    end = start + len(surface_form)
    if text[start:end].lower() == surface_form.lower():
        return start, end

    pattern = re.compile(r"(?i)(?<![A-Za-z])" + re.escape(surface_form) + r"(?![A-Za-z])")
    match = pattern.search(text, pos=max(0, start - 1))
    if not match:
        raise ValueError(f"Could not find surface form {surface_form!r} in text {text!r}")
    return match.start(), match.end()


def bert_target_pll(
    text: str,
    span_start: int,
    span_end: int,
    tokenizer,
    model,
    device: str,
) -> float:
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    offsets = encoded["offset_mapping"][0].tolist()

    target_positions: List[int] = []
    for idx, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        if token_start < span_end and token_end > span_start:
            target_positions.append(idx)

    if not target_positions:
        raise ValueError(f"Could not align BERT tokens to target span {(span_start, span_end)} in {text!r}")

    masked_rows = []
    for position in target_positions:
        masked_ids = input_ids.clone()
        masked_ids[position] = tokenizer.mask_token_id
        masked_rows.append(masked_ids)

    batch_input_ids = torch.stack(masked_rows).to(device)
    batch_attention = attention_mask.unsqueeze(0).repeat(len(target_positions), 1).to(device)

    with torch.no_grad():
        logits = model(input_ids=batch_input_ids, attention_mask=batch_attention).logits

    total_log_prob = 0.0
    for batch_index, token_position in enumerate(target_positions):
        token_id = int(input_ids[token_position].item())
        token_log_probs = torch.nn.functional.log_softmax(logits[batch_index, token_position], dim=-1)
        total_log_prob += float(token_log_probs[token_id].item())

    return total_log_prob


def compute_bert_full_context_fit(
    rows: Sequence[Dict[str, str]],
    pair_metadata: Dict[str, Dict[str, str]],
    tokenizer,
    model,
    device: str,
) -> Dict[str, float]:
    fit_scores: Dict[str, float] = {}
    for row in rows:
        pair_id, _ = split_item_id(row["item_id"])
        meta = pair_metadata[pair_id]
        surface_form = meta[row["target_word"]]
        span_start, span_end = find_surface_span(row["dialogue"], meta["dialogue_prefix"], surface_form)
        fit_scores[row["item_id"]] = bert_target_pll(
            text=row["dialogue"],
            span_start=span_start,
            span_end=span_end,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )
    return fit_scores


def main() -> None:
    args = parse_args()
    rows = prepare_rows(args.input_csv)
    pair_metadata = build_pair_metadata(rows)

    device = pick_device(args.device)
    tasks = build_tasks(rows, pair_metadata)
    tokenizer, model = load_model_and_tokenizer(args.model_name, device)

    try:
        task_scores = batch_score_tasks(tasks, tokenizer, model, device, args.batch_size)
    except RuntimeError:
        if device != "mps":
            raise
        tokenizer, model = load_model_and_tokenizer(args.model_name, "cpu")
        task_scores = batch_score_tasks(tasks, tokenizer, model, "cpu", args.batch_size)

    bert_tokenizer, bert_model = load_masked_model_and_tokenizer(args.bert_model_name, device)
    try:
        bert_fit_scores = compute_bert_full_context_fit(rows, pair_metadata, bert_tokenizer, bert_model, device)
    except RuntimeError:
        if device != "mps":
            raise
        bert_tokenizer, bert_model = load_masked_model_and_tokenizer(args.bert_model_name, "cpu")
        bert_fit_scores = compute_bert_full_context_fit(rows, pair_metadata, bert_tokenizer, bert_model, "cpu")

    scored_rows = score_rows(rows, pair_metadata, task_scores, bert_fit_scores)
    write_rows(args.output_csv, scored_rows)


if __name__ == "__main__":
    main()
