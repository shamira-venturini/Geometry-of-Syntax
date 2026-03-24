import argparse
import csv
import itertools
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_15000sampled_10-1.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "behavioral_results" / "core_generation_priming_pilot"
WORD_ORDER_KEYS = ("agent", "patient", "verb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small generation-based structural priming pilot on CORE transitives."
    )
    parser.add_argument("--model-name", default="EleutherAI/pythia-410m")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--samples-per-prime", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--word-order-mode",
        choices=("fixed", "all"),
        default="fixed",
        help="Whether to keep the original agent-patient-verb prompt order or sweep all permutations.",
    )
    return parser.parse_args()


def get_device(user_device: Optional[str]) -> str:
    if user_device:
        return user_device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_transitive_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = frame.columns.str.strip().str.lower()
    expected = {"pa", "pp", "ta", "tp"}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame[["pa", "pp", "ta", "tp"]]


def extract_bundle(active_target: str, passive_target: str) -> Dict[str, str]:
    active = active_target.strip().split()
    passive = passive_target.strip().split()
    if len(active) < 5 or len(passive) < 7:
        raise ValueError(f"Unexpected target formats: {active_target} / {passive_target}")
    return {
        "agent_noun": active[1],
        "patient_noun": active[4],
        "active_verb_form": active[2],
        "passive_verb_form": passive[3],
    }


def get_word_order_specs(mode: str) -> List[str]:
    if mode == "fixed":
        return ["agent-patient-verb"]
    return ["-".join(order) for order in itertools.permutations(WORD_ORDER_KEYS)]


def ordered_prompt_words(bundle: Dict[str, str], order_label: str) -> List[str]:
    active_form = bundle["active_verb_form"]
    passive_form = bundle["passive_verb_form"]
    verb_instruction = active_form if active_form == passive_form else f"{active_form}/{passive_form}"
    mapping = {
        "agent": bundle["agent_noun"],
        "patient": bundle["patient_noun"],
        "verb": verb_instruction,
    }
    return [mapping[key] for key in order_label.split("-")]


def build_prompt(prime_sentence: str, bundle: Dict[str, str], order_label: str) -> str:
    prompt_words = ordered_prompt_words(bundle, order_label)
    return (
        f"Prime sentence: {prime_sentence.strip()}\n"
        f"Use these words in one sentence: {', '.join(prompt_words)}\n"
        f"Sentence: the"
    )


def clean_generated_sentence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return text
    match = re.search(r"(.+?[.!?])(?:\s|$)", text)
    if match:
        return match.group(1).strip()
    return text.split("\n", 1)[0].strip()


def compose_generated_sentence(continuation: str) -> str:
    if continuation.startswith(" "):
        return clean_generated_sentence("the" + continuation)
    return clean_generated_sentence(f"the {continuation}")


def first_word(text: str) -> Optional[str]:
    match = re.search(r"\b([A-Za-z']+)\b", text)
    if not match:
        return None
    return match.group(1).lower()


def classify_first_word_choice(first_generated_word: Optional[str], bundle: Dict[str, str]) -> str:
    if not first_generated_word:
        return "other"
    if first_generated_word == bundle["agent_noun"].lower():
        return "active"
    if first_generated_word == bundle["patient_noun"].lower():
        return "passive"
    return "other"


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+|[.!?]", text.lower())


def find_all(tokens: List[str], needle: str) -> List[int]:
    return [idx for idx, token in enumerate(tokens) if token == needle.lower()]


def classify_generation(sentence: str, bundle: Dict[str, str]) -> str:
    tokens = tokenize_words(sentence)
    noun_a = bundle["agent_noun"].lower()
    noun_b = bundle["patient_noun"].lower()
    active_form = bundle["active_verb_form"].lower()
    passive_form = bundle["passive_verb_form"].lower()

    noun_a_positions = find_all(tokens, noun_a)
    noun_b_positions = find_all(tokens, noun_b)
    active_positions = find_all(tokens, active_form)
    passive_positions = find_all(tokens, passive_form)
    by_positions = find_all(tokens, "by")

    if not noun_a_positions or not noun_b_positions:
        return "other"

    noun_positions = sorted(noun_a_positions + noun_b_positions)

    for first_noun_pos in noun_positions:
        for passive_pos in passive_positions:
            if passive_pos <= first_noun_pos:
                continue
            window = tokens[max(first_noun_pos, passive_pos - 3):passive_pos]
            if not any(token in {"is", "was", "were", "be", "been", "being", "are"} for token in window):
                continue
            for by_pos in by_positions:
                if by_pos <= passive_pos:
                    continue
                if any(second_noun_pos > by_pos for second_noun_pos in noun_positions if second_noun_pos != first_noun_pos):
                    return "passive"

    for first_noun_pos in noun_positions:
        for verb_pos in active_positions + passive_positions:
            if verb_pos <= first_noun_pos:
                continue
            if any(second_noun_pos > verb_pos for second_noun_pos in noun_positions if second_noun_pos != first_noun_pos):
                return "active"

    return "other"


def generate_once(
    tokenizer,
    model,
    device: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    continuation = generated[0][encoded.input_ids.shape[1]:]
    return tokenizer.decode(continuation, skip_special_tokens=True)


def write_summary(frame: pd.DataFrame, output_dir: Path) -> None:
    if frame.empty:
        raise ValueError("Pilot produced no rows.")

    def build_summary_table(column_name: str, group_columns: List[str]) -> pd.DataFrame:
        summary_rows: List[Dict[str, object]] = []
        for group_key, subset in frame.groupby(group_columns):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            counts = subset[column_name].value_counts()
            total = len(subset)
            valid = int(counts.get("active", 0) + counts.get("passive", 0))
            row = dict(zip(group_columns, group_key))
            row.update(
                {
                    "n_generations": total,
                    "n_active": int(counts.get("active", 0)),
                    "n_passive": int(counts.get("passive", 0)),
                    "n_other": int(counts.get("other", 0)),
                    "valid_rate": valid / total,
                    "active_rate_all": counts.get("active", 0) / total,
                    "passive_rate_all": counts.get("passive", 0) / total,
                    "active_rate_valid": (counts.get("active", 0) / valid) if valid else None,
                    "passive_rate_valid": (counts.get("passive", 0) / valid) if valid else None,
                }
            )
            summary_rows.append(row)
        return pd.DataFrame(summary_rows).sort_values(group_columns)

    choice_summary = build_summary_table("choice_structure", ["prime_structure"])
    choice_summary.to_csv(output_dir / "choice_summary.csv", index=False)

    sentence_summary = build_summary_table("generated_structure", ["prime_structure"])
    sentence_summary.to_csv(output_dir / "sentence_summary.csv", index=False)

    choice_summary_by_order = build_summary_table("choice_structure", ["word_order", "prime_structure"])
    choice_summary_by_order.to_csv(output_dir / "choice_summary_by_order.csv", index=False)

    sentence_summary_by_order = build_summary_table("generated_structure", ["word_order", "prime_structure"])
    sentence_summary_by_order.to_csv(output_dir / "sentence_summary_by_order.csv", index=False)

    order_main_effects = build_summary_table("choice_structure", ["word_order"])
    order_main_effects.to_csv(output_dir / "order_main_effects.csv", index=False)

    active_prime = choice_summary.loc[choice_summary["prime_structure"] == "active"]
    passive_prime = choice_summary.loc[choice_summary["prime_structure"] == "passive"]
    comparison = {}
    if not active_prime.empty and not passive_prime.empty:
        comparison = {
            "passive_rate_shift_all": float(passive_prime["passive_rate_all"].iloc[0] - active_prime["passive_rate_all"].iloc[0]),
            "active_rate_shift_all": float(active_prime["active_rate_all"].iloc[0] - passive_prime["active_rate_all"].iloc[0]),
            "passive_rate_shift_valid": (
                float(passive_prime["passive_rate_valid"].iloc[0] - active_prime["passive_rate_valid"].iloc[0])
                if pd.notna(active_prime["passive_rate_valid"].iloc[0]) and pd.notna(passive_prime["passive_rate_valid"].iloc[0])
                else None
            ),
        }

    comparison_rows: List[Dict[str, object]] = []
    for word_order, subset in choice_summary_by_order.groupby("word_order"):
        active_order = subset.loc[subset["prime_structure"] == "active"]
        passive_order = subset.loc[subset["prime_structure"] == "passive"]
        if active_order.empty or passive_order.empty:
            continue
        comparison_rows.append(
            {
                "word_order": word_order,
                "active_rate_all_after_active_prime": float(active_order["active_rate_all"].iloc[0]),
                "active_rate_all_after_passive_prime": float(passive_order["active_rate_all"].iloc[0]),
                "passive_rate_all_after_active_prime": float(active_order["passive_rate_all"].iloc[0]),
                "passive_rate_all_after_passive_prime": float(passive_order["passive_rate_all"].iloc[0]),
                "active_rate_shift_all": float(active_order["active_rate_all"].iloc[0] - passive_order["active_rate_all"].iloc[0]),
                "passive_rate_shift_all": float(passive_order["passive_rate_all"].iloc[0] - active_order["passive_rate_all"].iloc[0]),
                "valid_rate_after_active_prime": float(active_order["valid_rate"].iloc[0]),
                "valid_rate_after_passive_prime": float(passive_order["valid_rate"].iloc[0]),
            }
        )
    comparison_by_order = pd.DataFrame(comparison_rows).sort_values("word_order")
    comparison_by_order.to_csv(output_dir / "comparison_by_order.csv", index=False)

    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))

    report_lines = [
        "# CORE Generation Priming Pilot",
        "",
        "## First-word choice summary",
        "",
        "```csv",
        choice_summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Full-sentence summary",
        "",
        "```csv",
        sentence_summary.to_csv(index=False).strip(),
        "```",
        "",
        "## First-word choice summary by prompt order",
        "",
        "```csv",
        choice_summary_by_order.to_csv(index=False).strip(),
        "```",
        "",
        "## Prompt-order comparisons",
        "",
        "```csv",
        comparison_by_order.to_csv(index=False).strip(),
        "```",
        "",
        "## Comparison",
        "",
        "```json",
        json.dumps(comparison, indent=2),
        "```",
        "",
        "Interpretation:",
        "- `choice_summary.csv` is the primary pilot output. It scores the first generated noun after `Sentence: the` as the structural commitment point.",
        "- `sentence_summary.csv` is a stricter secondary output based on whole-sentence active/passive coding.",
        "- `choice_summary_by_order.csv` and `comparison_by_order.csv` isolate prompt-order effects across all word permutations.",
        "- `passive_rate_shift_all` compares passive choice rate after passive primes vs active primes.",
        "- `active_rate_shift_all` compares active choice rate after active primes vs passive primes.",
        "- `*_valid` rates restrict to outputs classified as active or passive.",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_csv = args.input_csv.resolve()
    frame = normalize_transitive_frame(pd.read_csv(input_csv))
    frame = frame.sample(n=min(args.max_items, len(frame)), random_state=args.seed).reset_index(drop=True)
    word_orders = get_word_order_specs(args.word_order_mode)

    device = get_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    rows: List[Dict[str, object]] = []
    for item_index, row in frame.iterrows():
        bundle = extract_bundle(row["ta"], row["tp"])
        for word_order_index, word_order in enumerate(word_orders):
            prompts = {
                "active": build_prompt(row["pa"], bundle, word_order),
                "passive": build_prompt(row["pp"], bundle, word_order),
            }

            for prime_structure, prompt in prompts.items():
                for sample_index in range(args.samples_per_prime):
                    seed_offset = (
                        args.seed
                        + item_index * 1009
                        + word_order_index * 97
                        + sample_index * 17
                        + (0 if prime_structure == "active" else 1)
                    )
                    torch.manual_seed(seed_offset)
                    continuation = generate_once(
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    sentence = compose_generated_sentence(continuation)
                    generated_first_word = first_word(continuation)
                    choice_structure = classify_first_word_choice(generated_first_word, bundle)
                    classification = classify_generation(sentence, bundle)
                    rows.append(
                        {
                            "item_index": item_index,
                            "prime_structure": prime_structure,
                            "word_order": word_order,
                            "sample_index": sample_index,
                            "prime_sentence": row["pa"] if prime_structure == "active" else row["pp"],
                            "target_active": row["ta"],
                            "target_passive": row["tp"],
                            "agent_noun": bundle["agent_noun"],
                            "patient_noun": bundle["patient_noun"],
                            "active_verb_form": bundle["active_verb_form"],
                            "passive_verb_form": bundle["passive_verb_form"],
                            "prompt": prompt,
                            "raw_continuation": continuation,
                            "first_generated_word": generated_first_word,
                            "choice_structure": choice_structure,
                            "generated_sentence": sentence,
                            "generated_structure": classification,
                        }
                    )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "generations.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    metadata = {
        "model_name": args.model_name,
        "input_csv": str(input_csv),
        "max_items": int(args.max_items),
        "samples_per_prime": int(args.samples_per_prime),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "seed": int(args.seed),
        "device": device,
        "word_order_mode": args.word_order_mode,
        "word_orders": word_orders,
        "n_rows": int(len(results)),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    write_summary(results, output_dir)


if __name__ == "__main__":
    main()
