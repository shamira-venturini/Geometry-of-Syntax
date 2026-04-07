import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
VERB_LIST_PATH = REPO_ROOT / "PrimeLM" / "vocabulary_lists" / "verblist_T_usf_freq.csv"
N_RESAMPLES = 10000
CORE_FILLER_SENTENCES = [
    "The lantern glowed near sunset .",
    "The traveler rested beside the river .",
    "The notebook stayed on the shelf .",
    "The window rattled during the storm .",
    "The singer smiled after rehearsal .",
    "The painter waited near the doorway .",
    "The engine cooled before dawn .",
    "The market opened after sunrise .",
    "The blanket dried on the line .",
    "The planet shimmered above the valley .",
    "The baker laughed during breakfast .",
    "The signal flickered across the screen .",
    "The tourist wandered through the plaza .",
    "The kettle whistled in the kitchen .",
    "The garden brightened after rain .",
    "The jacket hung behind the chair .",
    "The radio crackled in the attic .",
    "The witness paused near the doorway .",
    "The package arrived before noon .",
    "The sculpture stood in the hallway .",
    "The captain waited beside the harbor .",
    "The teacher listened during assembly .",
    "The violin echoed in the theater .",
    "The pillow rested on the sofa .",
]
JABBERWOCKY_FILLER_SENTENCES = [
    "The noster glimmed near varset .",
    "The trassel fented beside murven .",
    "The krelbin staved on dralfin .",
    "The prindle rasped during forven .",
    "The slinter maved after bralken .",
    "The vornet drissed near malden .",
    "The krimble flened before narven .",
    "The tharner plested after zolven .",
    "The drasken yelped on fralden .",
    "The glarnet shummed above torven .",
    "The prasket whaved during morden .",
    "The flinder trassed across jorven .",
    "The claven morged through selven .",
    "The drabble quisted in varlen .",
    "The snorbel gredded after folven .",
    "The prantel staved behind nulven .",
    "The glimmer farned in jasken .",
    "The nasker plodded near hulven .",
    "The drimlet vorked before talven .",
    "The plinter drossed in marven .",
    "The clorven wepted beside prasken .",
    "The vornel stonned during nalven .",
    "The frindle glemmed in yorsen .",
    "The trasket draved on molven .",
]
# Backward-compatible alias used by older scripts.
DEFAULT_FILLER_SENTENCES = CORE_FILLER_SENTENCES


@dataclass(frozen=True)
class TargetBundle:
    agent_det: str
    agent_noun: str
    patient_det: str
    patient_noun: str
    active_verb_form: str
    passive_verb_form: str
    verb_lemma: str


def get_device(user_device: Optional[str]) -> str:
    if user_device:
        requested = str(user_device).strip().lower()
        if requested == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            warnings.warn("Requested device 'mps' is unavailable; falling back to 'cpu'.")
            return "cpu"
        if requested.startswith("cuda"):
            if torch.cuda.is_available():
                return requested
            warnings.warn(f"Requested device '{requested}' is unavailable; falling back to 'cpu'.")
            return "cpu"
        return requested
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


def load_verb_lookup(path: Path = VERB_LIST_PATH) -> Dict[Tuple[str, str], str]:
    frame = pd.read_csv(path, sep=";")
    lookup: Dict[Tuple[str, str], str] = {}
    for row in frame.itertuples(index=False):
        lookup[(str(row.pres_3s).strip().lower(), str(row.past_P).strip().lower())] = str(row.V).strip().lower()
    return lookup


def infer_lemma(active_form: str, passive_form: str, verb_lookup: Dict[Tuple[str, str], str]) -> str:
    key = (active_form.lower(), passive_form.lower())
    if key in verb_lookup:
        return verb_lookup[key]
    if active_form.endswith("ies") and len(active_form) > 3:
        return active_form[:-3] + "y"
    if active_form.endswith("es") and len(active_form) > 2:
        return active_form[:-2]
    if active_form.endswith("s") and len(active_form) > 1:
        return active_form[:-1]
    return active_form


def parse_active_target(text: str) -> Tuple[str, str, str, str, str]:
    tokens = text.strip().split()
    if len(tokens) != 6:
        raise ValueError(f"Unexpected active target format: {text}")
    return tokens[0], tokens[1], tokens[2], tokens[3], tokens[4]


def parse_passive_target(text: str) -> Tuple[str, str, str, str, str]:
    tokens = text.strip().split()
    if len(tokens) != 8:
        raise ValueError(f"Unexpected passive target format: {text}")
    return tokens[0], tokens[1], tokens[3], tokens[5], tokens[6]


def extract_bundle(
    active_target: str,
    passive_target: str,
    verb_lookup: Dict[Tuple[str, str], str],
) -> TargetBundle:
    agent_det, agent_noun, active_verb_form, patient_det, patient_noun = parse_active_target(active_target)
    passive_det, passive_patient, passive_verb_form, passive_agent_det, passive_agent = parse_passive_target(passive_target)
    if (patient_det, patient_noun) != (passive_det, passive_patient):
        raise ValueError(f"Passive patient mismatch: {active_target} / {passive_target}")
    if (agent_det, agent_noun) != (passive_agent_det, passive_agent):
        raise ValueError(f"Passive agent mismatch: {active_target} / {passive_target}")
    verb_lemma = infer_lemma(active_verb_form, passive_verb_form, verb_lookup)
    return TargetBundle(
        agent_det=agent_det.lower(),
        agent_noun=agent_noun.lower(),
        patient_det=patient_det.lower(),
        patient_noun=patient_noun.lower(),
        active_verb_form=active_verb_form.lower(),
        passive_verb_form=passive_verb_form.lower(),
        verb_lemma=verb_lemma.lower(),
    )


def sample_condition_frames(
    target_frame: pd.DataFrame,
    prime_frame: pd.DataFrame,
    max_items: Optional[int],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    rng = np.random.default_rng(seed)
    target_n = len(target_frame) if max_items is None else min(max_items, len(target_frame))
    target_indices = rng.choice(len(target_frame), size=target_n, replace=False)
    target_sample = target_frame.iloc[target_indices].reset_index(drop=True)

    if len(prime_frame) == len(target_frame):
        prime_sample = prime_frame.iloc[target_indices].reset_index(drop=True)
        return target_sample, prime_sample, "paired_indices"

    if len(prime_frame) < target_n:
        raise ValueError(
            f"Prime corpus has too few rows to sample {target_n} items independently: {len(prime_frame)}"
        )
    prime_indices = rng.choice(len(prime_frame), size=target_n, replace=False)
    prime_sample = prime_frame.iloc[prime_indices].reset_index(drop=True)
    return target_sample, prime_sample, "independent_sample"


def prompt_templates(mode: str) -> List[str]:
    if mode == "all":
        return ["word_list", "role_labeled"]
    return [mode]


def role_sequence(bundle: TargetBundle, template_name: str, rng: random.Random, order_mode: str) -> List[str]:
    if template_name == "word_list":
        elements = [
            bundle.agent_noun,
            bundle.patient_noun,
            bundle.verb_lemma,
        ]
    elif template_name == "role_labeled":
        elements = [
            f"AGENT={bundle.agent_noun}",
            f"PATIENT={bundle.patient_noun}",
            f"VERB={bundle.verb_lemma}",
        ]
    else:
        raise ValueError(f"Unknown prompt template: {template_name}")

    if order_mode == "fixed":
        return elements
    if order_mode == "shuffle":
        shuffled = list(elements)
        rng.shuffle(shuffled)
        return shuffled
    raise ValueError(f"Unknown role order mode: {order_mode}")


def build_prompt(
    prime_sentence: Optional[str],
    bundle: TargetBundle,
    template_name: str,
    role_sequence_values: Sequence[str],
    sentence_stub: str,
) -> str:
    lines: List[str] = []
    if prime_sentence:
        lines.append(f"Prime sentence: {prime_sentence.strip()}")

    if template_name == "word_list":
        lines.append(f"Use these words in one sentence: {', '.join(role_sequence_values)}")
    elif template_name == "role_labeled":
        lines.append(f"Event roles: {'; '.join(role_sequence_values)}")
    else:
        raise ValueError(f"Unknown prompt template: {template_name}")

    lines.append(sentence_stub)
    return "\n".join(lines)


def prompt_condition_order(raw_conditions: Sequence[str]) -> List[str]:
    allowed = {"active", "passive", "no_prime", "no_prime_eos", "no_prime_empty", "filler"}
    alias_map = {
        "no_prime": "no_prime_eos",
    }
    conditions = [condition.strip() for condition in raw_conditions if condition.strip()]
    conditions = [alias_map.get(condition, condition) for condition in conditions]
    invalid = sorted(set(conditions).difference(allowed))
    if invalid:
        raise ValueError(f"Unsupported prime conditions: {invalid}")
    if not conditions:
        raise ValueError("At least one prime condition is required.")
    return conditions


def filler_prime_for_item(item_index: int, seed: int, filler_sentences: Sequence[str]) -> str:
    if not filler_sentences:
        raise ValueError("Filler baseline requested without filler sentences.")
    rng = random.Random(seed + item_index * 7919)
    options = list(filler_sentences)
    rng.shuffle(options)
    return options[0]


def resolve_prime_sentence(
    prime_condition: str,
    prime_row: pd.Series,
    item_index: int,
    filler_seed: int,
    filler_sentences: Sequence[str],
) -> Optional[str]:
    if prime_condition == "active":
        return str(prime_row["pa"])
    if prime_condition == "passive":
        return str(prime_row["pp"])
    if prime_condition in {"no_prime", "no_prime_eos", "no_prime_empty"}:
        return None
    if prime_condition == "filler":
        return filler_prime_for_item(item_index=item_index, seed=filler_seed, filler_sentences=filler_sentences)
    raise ValueError(f"Unsupported prime condition: {prime_condition}")


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

        row_offset = 0
        for _, _, continuations, _ in batch_groups:
            group_scores: List[float] = []
            for continuation_index in range(len(continuations)):
                prompt_len = int(prompt_lens[row_offset])
                continuation_len = int(continuation_lens[row_offset])

                # With empty prompt, GPT-style causal scoring has no previous token for
                # the first continuation token, so we score from token 2 onward.
                if prompt_len == 0:
                    start_idx = 0
                    scored_len = max(0, continuation_len - 1)
                else:
                    start_idx = prompt_len - 1
                    scored_len = continuation_len

                end_idx = start_idx + scored_len
                group_scores.append(float(observed_log_probs[row_offset, start_idx:end_idx].sum().item()))
                row_offset += 1
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


def pairwise_stats(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    prime_condition_ordering: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    metrics = [
        ("passive_choice_indicator", "passive_choice_delta"),
        ("passive_minus_active_logprob", "logprob_delta"),
    ]

    for template_name, subset in frame.groupby("prompt_template"):
        pivot = subset.pivot(index="item_index", columns="prime_condition")
        available_conditions = [condition for condition in prime_condition_ordering if condition in pivot.columns.get_level_values(1)]
        for i, condition_a in enumerate(available_conditions):
            for condition_b in available_conditions[i + 1:]:
                for value_column, metric_name in metrics:
                    values = (
                        pivot[(value_column, condition_b)].astype(float).to_numpy()
                        - pivot[(value_column, condition_a)].astype(float).to_numpy()
                    )
                    mean_diff = float(values.mean())
                    sd_diff = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    if len(values) > 1:
                        t_stat, t_p = stats.ttest_1samp(values, popmean=0.0)
                        t_stat_value = float(t_stat)
                        t_p_value = float(t_p)
                    else:
                        t_stat_value = float("nan")
                        t_p_value = float("nan")
                    ci_low, ci_high = bootstrap_mean_ci(values, rng)
                    row: Dict[str, object] = {
                        "prompt_template": template_name,
                        "metric": metric_name,
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "n_items": int(len(values)),
                        "mean_diff_b_minus_a": mean_diff,
                        "sd_diff": sd_diff,
                        "effect_size_dz": float(mean_diff / sd_diff) if sd_diff else 0.0,
                        "t_stat": t_stat_value,
                        "t_p_two_sided": t_p_value,
                        "perm_p_two_sided": sign_flip_pvalue(values, rng),
                        "bootstrap_ci95_low": ci_low,
                        "bootstrap_ci95_high": ci_high,
                        "mcnemar_b": None,
                        "mcnemar_c": None,
                        "mcnemar_p_exact": None,
                    }

                    if metric_name == "passive_choice_delta":
                        a_choices = pivot[("chosen_structure", condition_a)]
                        b_choices = pivot[("chosen_structure", condition_b)]
                        a = int(((a_choices == "active") & (b_choices == "active")).sum())
                        b = int(((a_choices == "active") & (b_choices == "passive")).sum())
                        c = int(((a_choices == "passive") & (b_choices == "active")).sum())
                        d = int(((a_choices == "passive") & (b_choices == "passive")).sum())
                        mcnemar_result = mcnemar([[a, b], [c, d]], exact=True)
                        row["mcnemar_b"] = b
                        row["mcnemar_c"] = c
                        row["mcnemar_p_exact"] = float(mcnemar_result.pvalue)

                    rows.append(row)

    return pd.DataFrame(rows).sort_values(["prompt_template", "condition_a", "condition_b", "metric"])


def summarize_results(frame: pd.DataFrame) -> pd.DataFrame:
    summary_rows: List[Dict[str, object]] = []
    for (template_name, prime_condition), subset in frame.groupby(["prompt_template", "prime_condition"]):
        n_items = len(subset)
        n_active = int((subset["chosen_structure"] == "active").sum())
        n_passive = int((subset["chosen_structure"] == "passive").sum())
        summary_rows.append(
            {
                "prompt_template": template_name,
                "prime_condition": prime_condition,
                "n_items": n_items,
                "n_active_choice": n_active,
                "n_passive_choice": n_passive,
                "active_choice_rate": n_active / n_items,
                "passive_choice_rate": n_passive / n_items,
                "mean_passive_minus_active_logprob": float(subset["passive_minus_active_logprob"].mean()),
                "sd_passive_minus_active_logprob": float(
                    subset["passive_minus_active_logprob"].std(ddof=1)
                ) if n_items > 1 else 0.0,
            }
        )
    return pd.DataFrame(summary_rows).sort_values(["prompt_template", "prime_condition"])


def build_contrast_table(
    summary: pd.DataFrame,
    prime_condition_ordering: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for template_name, subset in summary.groupby("prompt_template"):
        available = {
            row["prime_condition"]: row
            for row in subset.to_dict(orient="records")
        }
        conditions = [condition for condition in prime_condition_ordering if condition in available]
        for i, condition_a in enumerate(conditions):
            for condition_b in conditions[i + 1:]:
                row_a = available[condition_a]
                row_b = available[condition_b]
                rows.append(
                    {
                        "prompt_template": template_name,
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "passive_choice_rate_a": float(row_a["passive_choice_rate"]),
                        "passive_choice_rate_b": float(row_b["passive_choice_rate"]),
                        "passive_choice_rate_diff_b_minus_a": float(
                            row_b["passive_choice_rate"] - row_a["passive_choice_rate"]
                        ),
                        "active_choice_rate_a": float(row_a["active_choice_rate"]),
                        "active_choice_rate_b": float(row_b["active_choice_rate"]),
                        "active_choice_rate_diff_b_minus_a": float(
                            row_b["active_choice_rate"] - row_a["active_choice_rate"]
                        ),
                        "mean_logprob_a": float(row_a["mean_passive_minus_active_logprob"]),
                        "mean_logprob_b": float(row_b["mean_passive_minus_active_logprob"]),
                        "mean_logprob_diff_b_minus_a": float(
                            row_b["mean_passive_minus_active_logprob"] - row_a["mean_passive_minus_active_logprob"]
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(["prompt_template", "condition_a", "condition_b"])


def write_common_outputs(
    frame: pd.DataFrame,
    output_dir: Path,
    title: str,
    prime_condition_ordering: Sequence[str],
    extra_metadata: Optional[Dict[str, object]] = None,
) -> None:
    if frame.empty:
        raise ValueError("Experiment produced no item rows.")
    rng = np.random.default_rng(13)

    summary = summarize_results(frame)
    contrasts = build_contrast_table(summary, prime_condition_ordering=prime_condition_ordering)
    stats_table = pairwise_stats(frame, rng=rng, prime_condition_ordering=prime_condition_ordering)

    frame.to_csv(output_dir / "item_scores.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    contrasts.to_csv(output_dir / "comparison.csv", index=False)
    stats_table.to_csv(output_dir / "stats.csv", index=False)

    report_lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Pairwise Condition Comparisons",
        "",
        "```csv",
        contrasts.to_csv(index=False).strip(),
        "```",
        "",
        "## Paired Significance Tests",
        "",
        "```csv",
        stats_table.to_csv(index=False).strip(),
        "```",
        "",
        "Interpretation:",
        "- `passive_choice_rate` is the share of items where the passive option outranked the active option.",
        "- `mean_passive_minus_active_logprob` is the mean passive-vs-active structural preference score.",
        "- In `comparison.csv` and `stats.csv`, differences are always `condition_b - condition_a`.",
        "- `passive_choice_delta` tests paired shifts in passive choice rates across prime conditions.",
        "- `logprob_delta` tests paired shifts in passive-vs-active preference across prime conditions.",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines))

    if extra_metadata is not None:
        (output_dir / "metadata.json").write_text(json.dumps(extra_metadata, indent=2))


def normalize_generated_text(text: str) -> str:
    normalized = text.strip().lower().replace(".", " . ")
    return " ".join(normalized.split())


def classify_generated_structure(text: str, target_active: str, target_passive: str) -> str:
    normalized = normalize_generated_text(text)
    active_norm = normalize_generated_text(target_active)
    passive_norm = normalize_generated_text(target_passive)
    if normalized == active_norm:
        return "active_exact"
    if normalized == passive_norm:
        return "passive_exact"
    if normalized.startswith(active_norm):
        return "active_prefix"
    if normalized.startswith(passive_norm):
        return "passive_prefix"
    return "other"
