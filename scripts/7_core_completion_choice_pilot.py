import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_15000sampled_10-1.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "behavioral_results" / "core_completion_choice_pilot"
N_RESAMPLES = 10000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a constrained completion-choice structural priming pilot on CORE transitives."
    )
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument(
        "--prime-csv",
        type=Path,
        default=None,
        help="Optional alternate prime source. Must have aligned pa/pp rows with input-csv targets.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--prompt-template",
        choices=("word_list", "role_labeled", "all"),
        default="all",
        help="Prompt family to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=13)
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


def prompt_templates(mode: str) -> List[str]:
    if mode == "all":
        return ["word_list", "role_labeled"]
    return [mode]


def verb_spec(bundle: Dict[str, str]) -> str:
    active = bundle["active_verb_form"]
    passive = bundle["passive_verb_form"]
    return active if active == passive else f"{active}/{passive}"


def build_prompt(prime_sentence: str, bundle: Dict[str, str], template_name: str) -> str:
    if template_name == "word_list":
        return (
            f"Prime sentence: {prime_sentence.strip()}\n"
            f"Use these words in one sentence: {bundle['agent_noun']}, {bundle['patient_noun']}, {verb_spec(bundle)}\n"
            f"Sentence: the"
        )

    if template_name == "role_labeled":
        return (
            f"Prime sentence: {prime_sentence.strip()}\n"
            f"Event roles: AGENT={bundle['agent_noun']}; PATIENT={bundle['patient_noun']}; VERB={verb_spec(bundle)}\n"
            f"Sentence: the"
        )

    raise ValueError(f"Unknown prompt template: {template_name}")


def continuation_log_probs(
    tokenizer,
    model,
    device: str,
    prompt: str,
    continuations: List[str],
) -> List[float]:
    full_texts = [prompt + continuation for continuation in continuations]
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    with torch.no_grad():
        logits = model(**inputs).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs.input_ids[:, 1:].contiguous()
    log_probs_all = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    observed_log_probs = log_probs_all.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

    results: List[float] = []
    prompt_len = len(prompt_ids)
    for row_idx, full_text in enumerate(full_texts):
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if full_ids[:prompt_len] != prompt_ids:
            raise ValueError("Prompt tokenization mismatch while scoring completion choices.")
        continuation_ids = full_ids[prompt_len:]
        if not continuation_ids:
            results.append(0.0)
            continue
        start_idx = prompt_len - 1
        end_idx = start_idx + len(continuation_ids)
        results.append(float(observed_log_probs[row_idx, start_idx:end_idx].sum().item()))
    return results


def batched_choice_log_probs(
    tokenizer,
    model,
    device: str,
    prompt_groups: List[Tuple[str, int, List[str], List[int]]],
    batch_size: int,
) -> List[List[float]]:
    all_scores: List[List[float]] = []
    for batch_start in range(0, len(prompt_groups), batch_size):
        batch_groups = prompt_groups[batch_start:batch_start + batch_size]
        full_texts: List[str] = []
        prompt_lens: List[int] = []
        continuation_lens: List[int] = []

        for prompt, prompt_len, continuations, continuation_len_group in batch_groups:
            for continuation, continuation_len in zip(continuations, continuation_len_group):
                full_texts.append(prompt + continuation)
                prompt_lens.append(prompt_len)
                continuation_lens.append(continuation_len)

        inputs = tokenizer(full_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs.input_ids[:, 1:].contiguous()
        log_probs_all = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        observed_log_probs = log_probs_all.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

        for group_index in range(len(batch_groups)):
            group_scores: List[float] = []
            for continuation_index in range(2):
                row_index = group_index * 2 + continuation_index
                start_idx = prompt_lens[row_index] - 1
                end_idx = start_idx + continuation_lens[row_index]
                group_scores.append(float(observed_log_probs[row_index, start_idx:end_idx].sum().item()))
            all_scores.append(group_scores)

    return all_scores


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> Tuple[float, float]:
    chunk_size = 250
    mean_chunks: List[np.ndarray] = []
    for chunk_start in range(0, N_RESAMPLES, chunk_size):
        current = min(chunk_size, N_RESAMPLES - chunk_start)
        samples = rng.choice(values, size=(current, len(values)), replace=True)
        mean_chunks.append(samples.mean(axis=1))
    means = np.concatenate(mean_chunks)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip_pvalue(values: np.ndarray, rng: np.random.Generator) -> float:
    observed = float(values.mean())
    chunk_size = 250
    exceed_count = 0
    total = 0
    sign_values = np.array([-1.0, 1.0])
    for chunk_start in range(0, N_RESAMPLES, chunk_size):
        current = min(chunk_size, N_RESAMPLES - chunk_start)
        signs = rng.choice(sign_values, size=(current, len(values)))
        permuted = (signs * values).mean(axis=1)
        exceed_count += int((np.abs(permuted) >= abs(observed)).sum())
        total += current
    return float(exceed_count / total)


def paired_stats(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for template_name, subset in frame.groupby("prompt_template"):
        paired = subset.pivot(index="item_index", columns="prime_structure")
        passive_choice_delta = (
            paired[("chosen_structure", "passive")].eq("passive").astype(float)
            - paired[("chosen_structure", "active")].eq("passive").astype(float)
        ).to_numpy()
        logprob_delta = (
            paired[("passive_minus_active_logprob", "passive")]
            - paired[("passive_minus_active_logprob", "active")]
        ).to_numpy()

        a = int(((paired[("chosen_structure", "active")] == "active") & (paired[("chosen_structure", "passive")] == "active")).sum())
        b = int(((paired[("chosen_structure", "active")] == "active") & (paired[("chosen_structure", "passive")] == "passive")).sum())
        c = int(((paired[("chosen_structure", "active")] == "passive") & (paired[("chosen_structure", "passive")] == "active")).sum())
        d = int(((paired[("chosen_structure", "active")] == "passive") & (paired[("chosen_structure", "passive")] == "passive")).sum())
        mcnemar_result = mcnemar([[a, b], [c, d]], exact=True)

        for metric_name, values in [
            ("passive_choice_delta", passive_choice_delta),
            ("logprob_delta", logprob_delta),
        ]:
            mean_diff = float(values.mean())
            sd_diff = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            t_stat, t_p = stats.ttest_1samp(values, popmean=0.0)
            ci_low, ci_high = bootstrap_mean_ci(values, rng)
            rows.append(
                {
                    "prompt_template": template_name,
                    "metric": metric_name,
                    "n_items": int(len(values)),
                    "mean_diff": mean_diff,
                    "sd_diff": sd_diff,
                    "effect_size_dz": float(mean_diff / sd_diff) if sd_diff else 0.0,
                    "t_stat": float(t_stat),
                    "t_p_two_sided": float(t_p),
                    "perm_p_two_sided": sign_flip_pvalue(values, rng),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "mcnemar_b": b if metric_name == "passive_choice_delta" else None,
                    "mcnemar_c": c if metric_name == "passive_choice_delta" else None,
                    "mcnemar_p_exact": float(mcnemar_result.pvalue) if metric_name == "passive_choice_delta" else None,
                }
            )
    return pd.DataFrame(rows).sort_values(["prompt_template", "metric"])


def write_outputs(frame: pd.DataFrame, output_dir: Path) -> None:
    if frame.empty:
        raise ValueError("Completion-choice pilot produced no rows.")
    rng = np.random.default_rng(13)

    summary_rows: List[Dict[str, object]] = []
    for (template_name, prime_structure), subset in frame.groupby(["prompt_template", "prime_structure"]):
        n_items = len(subset)
        n_active = int((subset["chosen_structure"] == "active").sum())
        n_passive = int((subset["chosen_structure"] == "passive").sum())
        summary_rows.append(
            {
                "prompt_template": template_name,
                "prime_structure": prime_structure,
                "n_items": n_items,
                "n_active_choice": n_active,
                "n_passive_choice": n_passive,
                "active_choice_rate": n_active / n_items,
                "passive_choice_rate": n_passive / n_items,
                "mean_passive_minus_active_logprob": float(subset["passive_minus_active_logprob"].mean()),
                "sd_passive_minus_active_logprob": float(subset["passive_minus_active_logprob"].std(ddof=1)) if n_items > 1 else 0.0,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["prompt_template", "prime_structure"])
    summary.to_csv(output_dir / "summary.csv", index=False)

    comparison_rows: List[Dict[str, object]] = []
    for template_name, subset in summary.groupby("prompt_template"):
        active_prime = subset.loc[subset["prime_structure"] == "active"]
        passive_prime = subset.loc[subset["prime_structure"] == "passive"]
        if active_prime.empty or passive_prime.empty:
            continue
        comparison_rows.append(
            {
                "prompt_template": template_name,
                "passive_choice_rate_after_active_prime": float(active_prime["passive_choice_rate"].iloc[0]),
                "passive_choice_rate_after_passive_prime": float(passive_prime["passive_choice_rate"].iloc[0]),
                "passive_choice_rate_shift": float(passive_prime["passive_choice_rate"].iloc[0] - active_prime["passive_choice_rate"].iloc[0]),
                "active_choice_rate_after_active_prime": float(active_prime["active_choice_rate"].iloc[0]),
                "active_choice_rate_after_passive_prime": float(passive_prime["active_choice_rate"].iloc[0]),
                "active_choice_rate_shift": float(active_prime["active_choice_rate"].iloc[0] - passive_prime["active_choice_rate"].iloc[0]),
                "mean_logprob_shift": float(
                    passive_prime["mean_passive_minus_active_logprob"].iloc[0]
                    - active_prime["mean_passive_minus_active_logprob"].iloc[0]
                ),
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("prompt_template")
    comparison.to_csv(output_dir / "comparison.csv", index=False)

    stats_table = paired_stats(frame, rng)
    stats_table.to_csv(output_dir / "stats.csv", index=False)

    report_lines = [
        "# CORE Completion-Choice Priming Pilot",
        "",
        "## Summary",
        "",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Comparison",
        "",
        "```csv",
        comparison.to_csv(index=False).strip(),
        "```",
        "",
        "## Paired Significance Tests",
        "",
        "```csv",
        stats_table.to_csv(index=False).strip(),
        "```",
        "",
        "Interpretation:",
        "- `passive_choice_rate_shift` is the key priming statistic in this pilot.",
        "- Positive values mean passive primes increase passive first-noun choices relative to active primes.",
        "- `mean_logprob_shift` compares patient-vs-agent noun preference after passive versus active primes.",
        "- `passive_choice_delta` is the paired item-level passive-choice difference (passive-prime minus active-prime).",
        "- `logprob_delta` is the paired item-level shift in patient-vs-agent noun preference.",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    input_csv = args.input_csv.resolve()
    target_frame = normalize_transitive_frame(pd.read_csv(input_csv))
    prime_csv = args.prime_csv.resolve() if args.prime_csv else input_csv
    prime_frame = normalize_transitive_frame(pd.read_csv(prime_csv))
    if len(target_frame) != len(prime_frame):
        raise ValueError(f"Prime and target corpora must have same number of rows: {len(prime_frame)} vs {len(target_frame)}")

    sampled_indices = (
        target_frame.sample(n=min(args.max_items, len(target_frame)), random_state=args.seed).index.to_list()
    )
    target_frame = target_frame.loc[sampled_indices].reset_index(drop=True)
    prime_frame = prime_frame.loc[sampled_indices].reset_index(drop=True)

    device = get_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    prompt_groups: List[Tuple[str, int, List[str], List[int]]] = []
    row_metadata: List[Dict[str, object]] = []
    for item_index, target_row in target_frame.iterrows():
        prime_row = prime_frame.loc[item_index]
        bundle = extract_bundle(target_row["ta"], target_row["tp"])
        candidates = {
            "active": f" {bundle['agent_noun']}",
            "passive": f" {bundle['patient_noun']}",
        }
        candidate_lengths = {
            name: len(tokenizer(text, add_special_tokens=False)["input_ids"])
            for name, text in candidates.items()
        }

        for template_name in prompt_templates(args.prompt_template):
            prompts = {
                "active": build_prompt(prime_row["pa"], bundle, template_name),
                "passive": build_prompt(prime_row["pp"], bundle, template_name),
            }

            for prime_structure, prompt in prompts.items():
                prompt_groups.append(
                    (
                        prompt,
                        len(tokenizer(prompt, add_special_tokens=False)["input_ids"]),
                        [candidates["active"], candidates["passive"]],
                        [candidate_lengths["active"], candidate_lengths["passive"]],
                    )
                )
                row_metadata.append(
                    {
                        "item_index": item_index,
                        "prompt_template": template_name,
                        "prime_structure": prime_structure,
                        "prime_sentence": prime_row["pa"] if prime_structure == "active" else prime_row["pp"],
                        "target_active": target_row["ta"],
                        "target_passive": target_row["tp"],
                        "agent_noun": bundle["agent_noun"],
                        "patient_noun": bundle["patient_noun"],
                        "active_verb_form": bundle["active_verb_form"],
                        "passive_verb_form": bundle["passive_verb_form"],
                        "prompt": prompt,
                    }
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
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "item_scores.csv", index=False)
    metadata = {
        "model_name": args.model_name,
        "input_csv": str(input_csv),
        "prime_csv": str(prime_csv),
        "max_items": int(args.max_items),
        "batch_size": int(args.batch_size),
        "prompt_template": args.prompt_template,
        "seed": int(args.seed),
        "device": device,
        "n_rows": int(len(results)),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    write_outputs(results, output_dir)


if __name__ == "__main__":
    main()
