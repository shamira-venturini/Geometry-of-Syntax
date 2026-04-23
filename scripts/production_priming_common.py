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
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
VERB_LIST_PATH = REPO_ROOT / "PrimeLM" / "vocabulary_lists" / "verblist_T_usf_freq.csv"
LEXICALLY_CONTROLLED_CORE_CSV = (
    REPO_ROOT / "corpora" / "transitive" / "CORE_transitive_constrained_counterbalanced_lexically_controlled.csv"
)
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

PRIME_CONDITION_ALIASES = {
    "active_prime": "active",
    "passive_prime": "passive",
    "filler_prime": "filler",
    "no_prime": "no_prime",
    "no_prime_eos": "no_prime",
    "no_prime_empty": "no_prime",
    "no_demo": "no_prime",
    "none": "no_prime",
}


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


def resolve_torch_dtype(dtype_name: Optional[str], device: str) -> Optional[torch.dtype]:
    normalized = (dtype_name or "auto").strip().lower()
    if normalized in {"", "auto"}:
        if device.startswith("cuda"):
            return torch.float16
        return None
    if normalized in {"none", "default"}:
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


def load_causal_lm_and_tokenizer(
    model_name: str,
    device: str,
    local_files_only: bool,
    torch_dtype_name: Optional[str] = "auto",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
    }
    resolved_dtype = resolve_torch_dtype(torch_dtype_name, device)
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()
    return tokenizer, model, resolved_dtype


def normalize_transitive_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = frame.columns.str.strip().str.lower()
    expected = {"pa", "pp", "ta", "tp"}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame[["pa", "pp", "ta", "tp"]]


def lexical_overlap_audit(target_frame: pd.DataFrame, prime_frame: pd.DataFrame) -> Dict[str, object]:
    if len(target_frame) != len(prime_frame):
        raise ValueError(
            f"Target and prime frames must be same length for lexical audit: {len(target_frame)} vs {len(prime_frame)}"
        )

    same_verb_rows = 0
    shared_noun_rows = 0
    shared_both_nouns_rows = 0

    for target_row, prime_row in zip(target_frame.itertuples(index=False), prime_frame.itertuples(index=False)):
        _, target_agent, target_verb, _, target_patient = parse_active_target(str(target_row.ta))
        _, prime_agent, prime_verb, _, prime_patient = parse_active_target(str(prime_row.pa))
        target_nouns = {target_agent.lower(), target_patient.lower()}
        prime_nouns = {prime_agent.lower(), prime_patient.lower()}
        overlap = target_nouns & prime_nouns
        same_verb_rows += int(target_verb.lower() == prime_verb.lower())
        shared_noun_rows += int(bool(overlap))
        shared_both_nouns_rows += int(target_nouns == prime_nouns)

    total_rows = len(target_frame)
    return {
        "rows_evaluated": int(total_rows),
        "same_active_verb_rows": int(same_verb_rows),
        "same_active_verb_rate": float(same_verb_rows / total_rows),
        "shared_noun_rows": int(shared_noun_rows),
        "shared_noun_rate": float(shared_noun_rows / total_rows),
        "shared_both_nouns_rows": int(shared_both_nouns_rows),
        "shared_both_nouns_rate": float(shared_both_nouns_rows / total_rows),
    }


def load_verb_lookup(path: Path = VERB_LIST_PATH) -> Dict[Tuple[str, str], str]:
    if not path.exists():
        warnings.warn(
            f"Verb lookup file not found at {path}; falling back to heuristic lemma inference.",
        )
        return {}
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
    if mode == "elicited_all":
        return ["cue_list", "another_event", "same_kind_event"]
    return [mode]


def role_sequence(bundle: TargetBundle, template_name: str, rng: random.Random, order_mode: str) -> List[str]:
    if template_name in {"word_list", "cue_list", "another_event", "same_kind_event"}:
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
        if template_name in {"cue_list", "another_event", "same_kind_event"}:
            lines.append(prime_sentence.strip())
            lines.append("")
        else:
            lines.append(f"Prime sentence: {prime_sentence.strip()}")

    if template_name == "word_list":
        lines.append(f"Use these words in one sentence: {', '.join(role_sequence_values)}")
    elif template_name == "role_labeled":
        lines.append(f"Event roles: {'; '.join(role_sequence_values)}")
    elif template_name == "cue_list":
        lines.append(", ".join(role_sequence_values))
    elif template_name == "another_event":
        lines.append(f"Another event: {', '.join(role_sequence_values)}.")
    elif template_name == "same_kind_event":
        lines.append(f"Different people, same kind of event: {', '.join(role_sequence_values)}.")
    else:
        raise ValueError(f"Unknown prompt template: {template_name}")

    lines.append(sentence_stub)
    return "\n".join(lines)


def canonical_prime_condition(condition: str) -> str:
    normalized = condition.strip().lower()
    return PRIME_CONDITION_ALIASES.get(normalized, normalized)


def prompt_condition_order(raw_conditions: Sequence[str]) -> List[str]:
    allowed = {"active", "passive", "no_prime", "filler"}
    conditions = [canonical_prime_condition(condition) for condition in raw_conditions if condition.strip()]
    invalid = sorted(set(conditions).difference(allowed))
    if invalid:
        raise ValueError(f"Unsupported prime conditions: {invalid}")
    if not conditions:
        raise ValueError("At least one prime condition is required.")
    ordered: List[str] = []
    seen = set()
    for condition in conditions:
        if condition not in seen:
            ordered.append(condition)
            seen.add(condition)
    return ordered


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
    condition = canonical_prime_condition(prime_condition)
    if condition == "active":
        return str(prime_row["pa"])
    if condition == "passive":
        return str(prime_row["pp"])
    if condition == "no_prime":
        return None
    if condition == "filler":
        return filler_prime_for_item(item_index=item_index, seed=filler_seed, filler_sentences=filler_sentences)
    raise ValueError(f"Unsupported prime condition: {prime_condition}")


def batched_choice_log_probs(
    tokenizer,
    model,
    device: str,
    prompt_groups: List[Tuple[str, int, List[str], List[int]]],
    batch_size: int,
) -> List[List[float]]:
    detailed_scores = batched_choice_detailed_scores(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt_groups=prompt_groups,
        batch_size=batch_size,
    )
    return [
        [float(choice["total_logprob"]) for choice in group]
        for group in detailed_scores
    ]


def batched_choice_detailed_scores(
    tokenizer,
    model,
    device: str,
    prompt_groups: List[Tuple[str, int, List[str], List[int]]],
    batch_size: int,
) -> List[List[Dict[str, object]]]:
    """Score each continuation and return per-token diagnostics.

    The scoring convention matches batched_choice_log_probs, including the
    no-prime behavior where an empty prompt cannot score the first continuation
    token (so token 2 onward is used).
    """
    all_scores: List[List[Dict[str, object]]] = []
    for batch_start in range(0, len(prompt_groups), batch_size):
        batch_groups = prompt_groups[batch_start:batch_start + batch_size]
        full_texts: List[str] = []
        prompt_lens: List[int] = []
        continuation_texts: List[str] = []
        continuation_ids_list: List[List[int]] = []

        for prompt, prompt_len, continuations, _ in batch_groups:
            for continuation in continuations:
                continuation_ids = list(tokenizer(continuation, add_special_tokens=False)["input_ids"])
                full_texts.append(prompt + continuation)
                prompt_lens.append(int(prompt_len))
                continuation_texts.append(str(continuation))
                continuation_ids_list.append(continuation_ids)

        inputs = tokenizer(full_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs.input_ids[:, 1:].contiguous()
        log_probs_all = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        observed_log_probs = log_probs_all.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

        row_offset = 0
        for _, _, continuations, _ in batch_groups:
            group_scores: List[Dict[str, object]] = []
            for _ in range(len(continuations)):
                prompt_len = int(prompt_lens[row_offset])
                continuation_text = continuation_texts[row_offset]
                continuation_ids = continuation_ids_list[row_offset]

                # With empty prompt, GPT-style causal scoring has no previous token for
                # the first continuation token, so we score from token 2 onward.
                if prompt_len == 0:
                    start_idx = 0
                    candidate_token_ids = continuation_ids[1:]
                else:
                    start_idx = prompt_len - 1
                    candidate_token_ids = continuation_ids

                scored_len = int(len(candidate_token_ids))
                end_idx = start_idx + scored_len
                token_logprobs = [
                    float(value)
                    for value in observed_log_probs[row_offset, start_idx:end_idx].tolist()
                ]

                if len(token_logprobs) != scored_len:
                    raise ValueError(
                        "Candidate token/logprob length mismatch while scoring continuations: "
                        f"tokens={scored_len} logprobs={len(token_logprobs)}"
                    )

                total_logprob = float(sum(token_logprobs))
                mean_logprob = float(total_logprob / max(1, scored_len))

                group_scores.append(
                    {
                        "candidate_text": continuation_text,
                        "total_logprob": total_logprob,
                        "mean_logprob": mean_logprob,
                        "token_count": scored_len,
                        "candidate_token_ids": [int(token_id) for token_id in candidate_token_ids],
                        "candidate_tokens": tokenizer.convert_ids_to_tokens(candidate_token_ids),
                        "candidate_token_logprobs": token_logprobs,
                    }
                )
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
    if "passive_minus_active_logprob_sum" in frame.columns:
        metrics.append(("passive_minus_active_logprob_sum", "logprob_sum_delta"))

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


def one_sample_summary(values: np.ndarray, rng: np.random.Generator) -> Dict[str, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    n = int(len(clean))
    if n == 0:
        return {
            "n_items": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "t_stat": float("nan"),
            "t_p_two_sided": float("nan"),
            "perm_p_two_sided": float("nan"),
            "bootstrap_ci95_low": float("nan"),
            "bootstrap_ci95_high": float("nan"),
        }
    if n == 1:
        return {
            "n_items": 1,
            "mean": float(clean.mean()),
            "sd": 0.0,
            "t_stat": float("nan"),
            "t_p_two_sided": float("nan"),
            "perm_p_two_sided": float("nan"),
            "bootstrap_ci95_low": float(clean[0]),
            "bootstrap_ci95_high": float(clean[0]),
        }

    mean_value = float(clean.mean())
    sd_value = float(clean.std(ddof=1))
    t_stat, t_p = stats.ttest_1samp(clean, popmean=0.0)
    ci_low, ci_high = bootstrap_mean_ci(clean, rng)
    return {
        "n_items": n,
        "mean": mean_value,
        "sd": sd_value,
        "t_stat": float(t_stat),
        "t_p_two_sided": float(t_p),
        "perm_p_two_sided": sign_flip_pvalue(clean, rng),
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
    }


def compute_sinclair_pe_tables(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sinclair et al. style PE tables:
      - active target PE = logP(active target | active prime) - logP(active target | passive prime)
      - passive target PE = logP(passive target | passive prime) - logP(passive target | active prime)
    """
    required = {"item_index", "prompt_template", "prime_condition", "active_choice_logprob", "passive_choice_logprob"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(), pd.DataFrame()

    include_sum = {"active_choice_logprob_sum", "passive_choice_logprob_sum"}.issubset(frame.columns)
    agg_map: Dict[str, str] = {
        "active_choice_logprob": "mean",
        "passive_choice_logprob": "mean",
    }
    if include_sum:
        agg_map["active_choice_logprob_sum"] = "mean"
        agg_map["passive_choice_logprob_sum"] = "mean"

    collapsed = (
        frame.groupby(["prompt_template", "item_index", "prime_condition"], as_index=False)
        .agg(agg_map)
    )

    item_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []

    for template_name, subset in collapsed.groupby("prompt_template"):
        active_lp = subset.pivot(index="item_index", columns="prime_condition", values="active_choice_logprob")
        passive_lp = subset.pivot(index="item_index", columns="prime_condition", values="passive_choice_logprob")

        needed = {"active", "passive"}
        if not needed.issubset(set(active_lp.columns)) or not needed.issubset(set(passive_lp.columns)):
            continue

        item_table = pd.DataFrame(
            {
                "item_index": active_lp.index.astype(int),
                "logp_active_target_given_active_prime": active_lp["active"].astype(float),
                "logp_active_target_given_passive_prime": active_lp["passive"].astype(float),
                "logp_passive_target_given_active_prime": passive_lp["active"].astype(float),
                "logp_passive_target_given_passive_prime": passive_lp["passive"].astype(float),
            }
        ).reset_index(drop=True)

        item_table["pe_active_target_logprob_same_minus_other"] = (
            item_table["logp_active_target_given_active_prime"]
            - item_table["logp_active_target_given_passive_prime"]
        )
        item_table["pe_passive_target_logprob_same_minus_other"] = (
            item_table["logp_passive_target_given_passive_prime"]
            - item_table["logp_passive_target_given_active_prime"]
        )
        item_table["pe_logprob_imbalance_passive_minus_active"] = (
            item_table["pe_passive_target_logprob_same_minus_other"]
            - item_table["pe_active_target_logprob_same_minus_other"]
        )

        if include_sum:
            active_sum = subset.pivot(index="item_index", columns="prime_condition", values="active_choice_logprob_sum")
            passive_sum = subset.pivot(index="item_index", columns="prime_condition", values="passive_choice_logprob_sum")
            if needed.issubset(set(active_sum.columns)) and needed.issubset(set(passive_sum.columns)):
                item_table["logp_sum_active_target_given_active_prime"] = active_sum.loc[
                    item_table["item_index"], "active"
                ].astype(float).to_numpy()
                item_table["logp_sum_active_target_given_passive_prime"] = active_sum.loc[
                    item_table["item_index"], "passive"
                ].astype(float).to_numpy()
                item_table["logp_sum_passive_target_given_active_prime"] = passive_sum.loc[
                    item_table["item_index"], "active"
                ].astype(float).to_numpy()
                item_table["logp_sum_passive_target_given_passive_prime"] = passive_sum.loc[
                    item_table["item_index"], "passive"
                ].astype(float).to_numpy()

                item_table["pe_active_target_logprob_sum_same_minus_other"] = (
                    item_table["logp_sum_active_target_given_active_prime"]
                    - item_table["logp_sum_active_target_given_passive_prime"]
                )
                item_table["pe_passive_target_logprob_sum_same_minus_other"] = (
                    item_table["logp_sum_passive_target_given_passive_prime"]
                    - item_table["logp_sum_passive_target_given_active_prime"]
                )
                item_table["pe_logprob_sum_imbalance_passive_minus_active"] = (
                    item_table["pe_passive_target_logprob_sum_same_minus_other"]
                    - item_table["pe_active_target_logprob_sum_same_minus_other"]
                )

        item_table = item_table.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        if item_table.empty:
            continue
        item_table.insert(0, "prompt_template", template_name)
        item_rows.append(item_table)

        active_stats = one_sample_summary(
            item_table["pe_active_target_logprob_same_minus_other"].to_numpy(dtype=float),
            rng=rng,
        )
        passive_stats = one_sample_summary(
            item_table["pe_passive_target_logprob_same_minus_other"].to_numpy(dtype=float),
            rng=rng,
        )
        imbalance_stats = one_sample_summary(
            item_table["pe_logprob_imbalance_passive_minus_active"].to_numpy(dtype=float),
            rng=rng,
        )

        row: Dict[str, object] = {
            "prompt_template": template_name,
            "n_items": int(len(item_table)),
            "pe_active_target_logprob_same_minus_other": active_stats["mean"],
            "pe_active_target_logprob_ci95_low": active_stats["bootstrap_ci95_low"],
            "pe_active_target_logprob_ci95_high": active_stats["bootstrap_ci95_high"],
            "pe_active_target_logprob_p": active_stats["t_p_two_sided"],
            "pe_active_target_logprob_perm_p": active_stats["perm_p_two_sided"],
            "pe_passive_target_logprob_same_minus_other": passive_stats["mean"],
            "pe_passive_target_logprob_ci95_low": passive_stats["bootstrap_ci95_low"],
            "pe_passive_target_logprob_ci95_high": passive_stats["bootstrap_ci95_high"],
            "pe_passive_target_logprob_p": passive_stats["t_p_two_sided"],
            "pe_passive_target_logprob_perm_p": passive_stats["perm_p_two_sided"],
            "pe_logprob_imbalance_passive_minus_active": imbalance_stats["mean"],
            "pe_logprob_imbalance_ci95_low": imbalance_stats["bootstrap_ci95_low"],
            "pe_logprob_imbalance_ci95_high": imbalance_stats["bootstrap_ci95_high"],
            "pe_logprob_imbalance_p": imbalance_stats["t_p_two_sided"],
            "pe_logprob_imbalance_perm_p": imbalance_stats["perm_p_two_sided"],
        }

        if "pe_active_target_logprob_sum_same_minus_other" in item_table.columns:
            active_sum_stats = one_sample_summary(
                item_table["pe_active_target_logprob_sum_same_minus_other"].to_numpy(dtype=float),
                rng=rng,
            )
            passive_sum_stats = one_sample_summary(
                item_table["pe_passive_target_logprob_sum_same_minus_other"].to_numpy(dtype=float),
                rng=rng,
            )
            imbalance_sum_stats = one_sample_summary(
                item_table["pe_logprob_sum_imbalance_passive_minus_active"].to_numpy(dtype=float),
                rng=rng,
            )
            row.update(
                {
                    "pe_active_target_logprob_sum_same_minus_other": active_sum_stats["mean"],
                    "pe_active_target_logprob_sum_ci95_low": active_sum_stats["bootstrap_ci95_low"],
                    "pe_active_target_logprob_sum_ci95_high": active_sum_stats["bootstrap_ci95_high"],
                    "pe_active_target_logprob_sum_p": active_sum_stats["t_p_two_sided"],
                    "pe_active_target_logprob_sum_perm_p": active_sum_stats["perm_p_two_sided"],
                    "pe_passive_target_logprob_sum_same_minus_other": passive_sum_stats["mean"],
                    "pe_passive_target_logprob_sum_ci95_low": passive_sum_stats["bootstrap_ci95_low"],
                    "pe_passive_target_logprob_sum_ci95_high": passive_sum_stats["bootstrap_ci95_high"],
                    "pe_passive_target_logprob_sum_p": passive_sum_stats["t_p_two_sided"],
                    "pe_passive_target_logprob_sum_perm_p": passive_sum_stats["perm_p_two_sided"],
                    "pe_logprob_sum_imbalance_passive_minus_active": imbalance_sum_stats["mean"],
                    "pe_logprob_sum_imbalance_ci95_low": imbalance_sum_stats["bootstrap_ci95_low"],
                    "pe_logprob_sum_imbalance_ci95_high": imbalance_sum_stats["bootstrap_ci95_high"],
                    "pe_logprob_sum_imbalance_p": imbalance_sum_stats["t_p_two_sided"],
                    "pe_logprob_sum_imbalance_perm_p": imbalance_sum_stats["perm_p_two_sided"],
                }
            )

        summary_rows.append(row)

    if not item_rows:
        return pd.DataFrame(), pd.DataFrame()
    item_frame = pd.concat(item_rows, ignore_index=True)
    summary_frame = pd.DataFrame(summary_rows).sort_values(["prompt_template"]).reset_index(drop=True)
    return item_frame, summary_frame


def summarize_results(frame: pd.DataFrame) -> pd.DataFrame:
    summary_rows: List[Dict[str, object]] = []
    include_sum_metric = "passive_minus_active_logprob_sum" in frame.columns
    for (template_name, prime_condition), subset in frame.groupby(["prompt_template", "prime_condition"]):
        n_items = len(subset)
        n_active = int((subset["chosen_structure"] == "active").sum()) if "chosen_structure" in subset.columns else 0
        n_passive = int((subset["chosen_structure"] == "passive").sum()) if "chosen_structure" in subset.columns else 0
        row = {
            "prompt_template": template_name,
            "prime_condition": prime_condition,
            "n_items": n_items,
            "n_active_choice": n_active,
            "n_passive_choice": n_passive,
            "active_choice_rate": n_active / n_items if n_items else 0.0,
            "passive_choice_rate": n_passive / n_items if n_items else 0.0,
            "mean_passive_minus_active_logprob": float(subset["passive_minus_active_logprob"].mean()),
            "sd_passive_minus_active_logprob": float(
                subset["passive_minus_active_logprob"].std(ddof=1)
            ) if n_items > 1 else 0.0,
        }
        if include_sum_metric:
            row["mean_passive_minus_active_logprob_sum"] = float(subset["passive_minus_active_logprob_sum"].mean())
            row["sd_passive_minus_active_logprob_sum"] = float(
                subset["passive_minus_active_logprob_sum"].std(ddof=1)
            ) if n_items > 1 else 0.0
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values(["prompt_template", "prime_condition"])


def build_contrast_table(
    summary: pd.DataFrame,
    prime_condition_ordering: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    include_sum_metric = "mean_passive_minus_active_logprob_sum" in summary.columns
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
                if include_sum_metric:
                    rows[-1]["mean_logprob_sum_a"] = float(row_a["mean_passive_minus_active_logprob_sum"])
                    rows[-1]["mean_logprob_sum_b"] = float(row_b["mean_passive_minus_active_logprob_sum"])
                    rows[-1]["mean_logprob_sum_diff_b_minus_a"] = float(
                        row_b["mean_passive_minus_active_logprob_sum"] - row_a["mean_passive_minus_active_logprob_sum"]
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
    pe_item_table, pe_summary = compute_sinclair_pe_tables(frame=frame, rng=rng)

    frame.to_csv(output_dir / "item_scores.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    contrasts.to_csv(output_dir / "comparison.csv", index=False)
    stats_table.to_csv(output_dir / "stats.csv", index=False)
    if not pe_item_table.empty:
        pe_item_table.to_csv(output_dir / "sinclair_pe_item_scores.csv", index=False)
    if not pe_summary.empty:
        pe_summary.to_csv(output_dir / "sinclair_pe_summary.csv", index=False)

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
    ]
    if not pe_summary.empty:
        report_lines.extend(
            [
                "## Sinclair-Style PE (Same Minus Other)",
                "",
                "```csv",
                pe_summary.to_csv(index=False).strip(),
                "```",
                "",
            ]
        )
    report_lines.extend(
        [
        "Interpretation:",
        "- `passive_choice_rate` is the share of items where the passive option outranked the active option.",
        "- `mean_passive_minus_active_logprob` is the mean passive-vs-active structural preference score.",
        "- In `comparison.csv` and `stats.csv`, differences are always `condition_b - condition_a`.",
        "- `passive_choice_delta` tests paired shifts in passive choice rates across prime conditions.",
        "- `logprob_delta` tests paired shifts in passive-vs-active preference across prime conditions.",
        "- `sinclair_pe_summary.csv` reports paper-style PE: same-prime minus other-prime for each target form.",
        "- `pe_active_target_logprob_same_minus_other = logP(active|active prime) - logP(active|passive prime)`.",
        "- `pe_passive_target_logprob_same_minus_other = logP(passive|passive prime) - logP(passive|active prime)`.",
    ]
    )
    if "mean_passive_minus_active_logprob_sum" in summary.columns:
        report_lines.append(
            "- `mean_passive_minus_active_logprob_sum` is the summed passive-vs-active preference score."
        )
    if "logprob_sum_delta" in stats_table.get("metric", pd.Series(dtype=str)).astype(str).tolist():
        report_lines.append(
            "- `logprob_sum_delta` tests paired shifts in summed passive-vs-active preference across prime conditions."
        )
    (output_dir / "report.md").write_text("\n".join(report_lines))

    if extra_metadata is not None:
        (output_dir / "metadata.json").write_text(json.dumps(extra_metadata, indent=2))


def normalize_generated_text(text: str) -> str:
    normalized = text.strip().lower().replace(".", " . ")
    return " ".join(normalized.split())


DETERMINERS = {"a", "an", "the"}
BE_FORMS = {"am", "is", "are", "was", "were", "be", "been", "being"}


def _drop_final_period(tokens: List[str]) -> List[str]:
    if tokens and tokens[-1] == ".":
        return tokens[:-1]
    return tokens


def _last_content_token(tokens: Sequence[str]) -> Optional[str]:
    for token in reversed(tokens):
        if token not in DETERMINERS:
            return token
    return tokens[-1] if tokens else None


def _rough_verb_key(token: str) -> str:
    token = token.lower()
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ied") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        return stem
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        return stem
    if token.endswith("es") and len(token) > 4:
        if token.endswith(("ches", "shes", "sses", "xes", "zes", "oes", "ges")):
            return token[:-2]
        return token[:-1]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _extract_target_signature(target_active: str, target_passive: str) -> Optional[Dict[str, object]]:
    active_tokens = _drop_final_period(normalize_generated_text(target_active).split())
    passive_tokens = _drop_final_period(normalize_generated_text(target_passive).split())
    if "by" not in passive_tokens:
        return None

    by_index = passive_tokens.index("by")
    be_positions = [index for index, token in enumerate(passive_tokens[:by_index]) if token in BE_FORMS]
    if not be_positions:
        return None

    be_index = be_positions[-1]
    patient_phrase = passive_tokens[:be_index]
    agent_phrase = passive_tokens[by_index + 1:]
    if not patient_phrase or not agent_phrase or be_index + 1 >= by_index:
        return None

    passive_verb = passive_tokens[be_index + 1]
    active_verb = None
    if (
        len(active_tokens) > len(agent_phrase) + len(patient_phrase)
        and active_tokens[: len(agent_phrase)] == agent_phrase
        and active_tokens[-len(patient_phrase):] == patient_phrase
    ):
        active_verb = active_tokens[len(agent_phrase)]

    expected_verb_keys = {_rough_verb_key(passive_verb)}
    if active_verb is not None:
        expected_verb_keys.add(_rough_verb_key(active_verb))

    agent_head = _last_content_token(agent_phrase)
    patient_head = _last_content_token(patient_phrase)
    if agent_head is None or patient_head is None:
        return None

    return {
        "agent_head": agent_head,
        "patient_head": patient_head,
        "expected_verb_keys": expected_verb_keys,
    }


def _first_index(tokens: Sequence[str], target: str) -> Optional[int]:
    for index, token in enumerate(tokens):
        if token == target:
            return index
    return None


def _contains_expected_verb(tokens: Sequence[str], expected_verb_keys: Sequence[str]) -> bool:
    return any(_rough_verb_key(token) in expected_verb_keys for token in tokens)


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

    signature = _extract_target_signature(target_active=target_active, target_passive=target_passive)
    if signature is None:
        return "other"

    tokens = _drop_final_period(normalized.split())
    agent_index = _first_index(tokens, str(signature["agent_head"]))
    patient_index = _first_index(tokens, str(signature["patient_head"]))
    expected_verb_keys = signature["expected_verb_keys"]
    if agent_index is None or patient_index is None:
        return "other"

    if agent_index < patient_index:
        span = tokens[agent_index + 1:patient_index]
        if "by" not in span and _contains_expected_verb(span, expected_verb_keys):
            return "active_structural"

    if patient_index < agent_index:
        middle = tokens[patient_index + 1:agent_index]
        if "by" in middle:
            by_index = patient_index + 1 + middle.index("by")
            pre_by = tokens[patient_index + 1:by_index]
            if any(token in BE_FORMS for token in pre_by) and _contains_expected_verb(pre_by, expected_verb_keys):
                return "passive_structural"

    return "other"
