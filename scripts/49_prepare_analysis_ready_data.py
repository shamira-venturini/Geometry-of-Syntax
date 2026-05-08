#!/usr/bin/env python3
"""Build analysis-ready CSVs for the R mixed-model notebook."""

from __future__ import annotations

import json
import re
import ast
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "behavioral_results"
OUTPUT_DIR = RESULTS_ROOT / "analysis_ready"


MODEL_METADATA: Dict[str, Dict[str, str]] = {
    "colab_gpt_2_large": {
        "model_size": "774M",
        "model_family": "gpt2",
        "model_instruct": "base",
        "model_condition": "774M_base",
        "model_label": "GPT-2 large",
    },
    "colab_Llama_32_3B": {
        "model_size": "3B",
        "model_family": "llama",
        "model_instruct": "base",
        "model_condition": "3B_base",
        "model_label": "Llama 3.2 3B",
    },
    "colab_Llama_32_3B_Instruct": {
        "model_size": "3B",
        "model_family": "llama",
        "model_instruct": "instruct",
        "model_condition": "3B_instruct",
        "model_label": "Llama 3.2 3B Instruct",
    },
    "colab_Llama_31_8B": {
        "model_size": "8B",
        "model_family": "llama",
        "model_instruct": "base",
        "model_condition": "8B_base",
        "model_label": "Llama 3.1 8B",
    },
    "colab_Llama_31_8B_Instruct": {
        "model_size": "8B",
        "model_family": "llama",
        "model_instruct": "instruct",
        "model_condition": "8B_instruct",
        "model_label": "Llama 3.1 8B Instruct",
    },
    "colab_Gemma_2_9B": {
        "model_size": "9B",
        "model_family": "gemma",
        "model_instruct": "base",
        "model_condition": "9B_base",
        "model_label": "Gemma 2 9B",
    },
}


def infer_model_run(path: Path) -> str:
    for part in path.parts:
        if part.startswith("colab_"):
            return part
    return "unknown_model"


def model_metadata(model_run: str) -> Dict[str, str]:
    return {
        "source_model_run": model_run,
        **MODEL_METADATA.get(
            model_run,
            {
                "model_size": "unspecified",
                "model_family": "unspecified",
                "model_instruct": "unspecified",
                "model_condition": "unspecified",
                "model_label": model_run,
            },
        ),
    }


def normalize_lexicality(value: object, dataset: object = "") -> str:
    text = str(value or "").strip().lower()
    data = str(dataset or "").strip().lower()
    if text in {"real", "core"}:
        return "core"
    if text in {"nonce", "jabberwocky"}:
        return "jabberwocky"
    if "core_targets_jabberwocky_primes" in data:
        return "mixed"
    if data == "core":
        return "core"
    if data == "jabberwocky":
        return "jabberwocky"
    return text or data or "unspecified"


def infer_dataset_from_path(path: Path) -> str:
    parts = list(path.parts)
    for candidate in ("core_targets_jabberwocky_primes", "jabberwocky", "core"):
        if candidate in parts:
            return candidate
    text = str(path)
    if "core_targets_jabberwocky_primes" in text:
        return "core_targets_jabberwocky_primes"
    if "jabberwocky" in text:
        return "jabberwocky"
    if "core" in text:
        return "core"
    return "unspecified"


def add_common_columns(frame: pd.DataFrame, *, path: Path, experiment: str, task_type: str) -> pd.DataFrame:
    out = frame.copy()
    model_run = infer_model_run(path)
    for key, value in model_metadata(model_run).items():
        out[key] = value
    out["source_path"] = str(path.relative_to(REPO_ROOT))
    out["experiment"] = experiment
    out["task_type"] = task_type
    out["dataset"] = infer_dataset_from_path(path)
    if "lexicality_condition" in out.columns:
        out["lexicality_condition"] = [
            normalize_lexicality(value, dataset)
            for value, dataset in zip(out["lexicality_condition"], out["dataset"])
        ]
    else:
        out["lexicality_condition"] = [normalize_lexicality("", dataset) for dataset in out["dataset"]]
    if "item_order" not in out.columns and "item_index" in out.columns:
        out["item_order"] = out["item_index"]
    if "item_id" not in out.columns and "item_index" in out.columns:
        out["item_id"] = out.apply(
            lambda row: f"{experiment}_{row.get('dataset', 'dataset')}_{int(row['item_index']):05d}"
            if pd.notna(row["item_index"])
            else "",
            axis=1,
        )
    return out


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def finalize_teacher_forced(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "passive_minus_active_logprob" in out.columns:
        out["logodds_passive"] = numeric_column(out, "passive_minus_active_logprob")
    elif {"passive_choice_logprob", "active_choice_logprob"}.issubset(out.columns):
        out["logodds_passive"] = (
            numeric_column(out, "passive_choice_logprob")
            - numeric_column(out, "active_choice_logprob")
        )
    else:
        out["logodds_passive"] = np.nan

    if "passive_minus_active_logprob_sum" in out.columns:
        out["logodds_passive_sum"] = numeric_column(out, "passive_minus_active_logprob_sum")
    elif {"passive_choice_logprob_sum", "active_choice_logprob_sum"}.issubset(out.columns):
        out["logodds_passive_sum"] = (
            numeric_column(out, "passive_choice_logprob_sum")
            - numeric_column(out, "active_choice_logprob_sum")
        )
    else:
        out["logodds_passive_sum"] = np.nan

    if {"passive_target_token_count", "active_target_token_count"}.issubset(out.columns):
        out["passive_minus_active_token_count"] = (
            numeric_column(out, "passive_target_token_count")
            - numeric_column(out, "active_target_token_count")
        )
        out["token_count_diff"] = out["passive_minus_active_token_count"]
    else:
        out["passive_minus_active_token_count"] = np.nan
        out["token_count_diff"] = np.nan
    return out


def read_csvs(paths: Iterable[Path], *, experiment: str, task_type: str) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = add_common_columns(frame, path=path, experiment=experiment, task_type=task_type)
        frames.append(frame)
    return frames


def exp1b_paths() -> List[Path]:
    paths = set(RESULTS_ROOT.glob("**/experiment-1*/**/experiment-1b/**/item_scores.csv"))
    paths.update(RESULTS_ROOT.glob("**/experiment-1b/**/item_scores.csv"))
    return sorted(paths)


def exp3_paths() -> List[Path]:
    return sorted(RESULTS_ROOT.glob("**/experiment-3/**/item_level_results.csv"))


def exp2_reviewed_paths() -> List[Path]:
    paths = sorted(RESULTS_ROOT.glob("**/experiment-2/**/item_generations_reviewed.csv"))
    # Keep one run per model/dataset when duplicate historical folders exist.
    priority_patterns = [
        "generation_experiment_2_Llama_32_3B_lexically_controlled",
        "experiment-2_generation_audit_lexically_controlled",
        "/experiment-2/core",
        "/experiment-2/core_targets_jabberwocky_primes",
    ]
    selected: Dict[tuple[str, str], Path] = {}
    selected_priority: Dict[tuple[str, str], int] = {}
    for path in paths:
        key = (infer_model_run(path), infer_dataset_from_path(path))
        text = str(path)
        priority = next(
            (idx for idx, pattern in enumerate(priority_patterns) if pattern in text),
            len(priority_patterns),
        )
        if key not in selected or priority < selected_priority[key]:
            selected[key] = path
            selected_priority[key] = priority
    return sorted(selected.values())


def exp4_comprehension_paths() -> List[Path]:
    return sorted(RESULTS_ROOT.glob("**/experiment-4/**/item_level_results_exp4.csv"))


def exp4_pe_paths() -> List[Path]:
    return sorted(RESULTS_ROOT.glob("**/experiment-4/**/sinclair_target_pe/sinclair_pe_item_level_exp4.csv"))


def build_item_level_master() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    frames.extend(read_csvs(exp1b_paths(), experiment="experiment_1b", task_type="fixed_target_processing"))
    frames.extend(read_csvs(exp3_paths(), experiment="experiment_3", task_type="event_to_syntax_mapping"))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = finalize_teacher_forced(out)
    return out


def build_baseline_shifts(item_level: pd.DataFrame) -> pd.DataFrame:
    if item_level.empty or "prime_condition" not in item_level.columns:
        return pd.DataFrame()
    required = ["source_model_run", "experiment", "task_type", "lexicality_condition", "item_id"]
    available = [column for column in required if column in item_level.columns]
    optional = [column for column in ["role_order", "dataset"] if column in item_level.columns]
    group_cols = available + optional
    rows: List[Dict[str, object]] = []
    for keys, group in item_level.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        values = group.groupby("prime_condition", dropna=False)["logodds_passive"].mean()
        if "no_prime" not in values.index:
            continue
        baseline = float(values.loc["no_prime"])
        for prime_type in ("active", "passive"):
            if prime_type not in values.index:
                continue
            row = {**base}
            exemplar = group.iloc[0]
            for column in [
                "model_name",
                "model_condition",
                "model_size",
                "model_family",
                "model_instruct",
                "model_label",
                "item_index",
                "item_order",
                "passive_minus_active_token_count",
                "token_count_diff",
            ]:
                if column in group.columns:
                    row[column] = exemplar.get(column)
            row["prime_type"] = prime_type
            row["baseline_condition"] = "no_prime"
            row["baseline_logodds_passive"] = baseline
            row["primed_logodds_passive"] = float(values.loc[prime_type])
            row["shift_from_no_prime"] = row["primed_logodds_passive"] - baseline
            rows.append(row)
    return pd.DataFrame(rows)


def parse_serialized_list(value: object) -> List[object]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    text = str(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
    return parsed if isinstance(parsed, list) else []


def normalize_candidate_voice(label: object, fallback: str) -> str:
    text = str(label or "").lower()
    if "active" in text:
        return "active"
    if "passive" in text:
        return "passive"
    return fallback


def candidate_words(candidate_text: object) -> List[str]:
    text = str(candidate_text or "").strip().lower()
    text = text.replace('"', "")
    return re.findall(r"[a-z]+|[.?!]", text)


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
    """Greedily align scored token pieces to candidate words.

    This handles BOS/no-prime cases where the first determiner may be absent
    from the scored continuation and subword tokenization where a word can span
    multiple tokens.
    """
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


def roi_label_for_word(target_voice: str, word_index: int) -> str:
    if word_index < 0:
        return "unaligned"
    if target_voice == "passive":
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
    if target_voice == "active":
        labels = {
            0: "initial_det",
            1: "agent_np",
            2: "verb",
            3: "patient_det",
            4: "patient_np",
            5: "punctuation",
        }
        return labels.get(word_index, "other_active")
    return "other"


def build_token_level_probabilities(item_level: pd.DataFrame) -> pd.DataFrame:
    if item_level.empty:
        return pd.DataFrame()

    required = {
        "candidate_a_tokens",
        "candidate_a_token_logprobs",
        "candidate_b_tokens",
        "candidate_b_token_logprobs",
    }
    if not required.issubset(item_level.columns):
        return pd.DataFrame()

    metadata_cols = [
        "experiment",
        "task_type",
        "dataset",
        "source_model_run",
        "model_condition",
        "model_size",
        "model_family",
        "model_instruct",
        "model_label",
        "model_name",
        "lexicality_condition",
        "prime_condition",
        "target_voice",
        "item_id",
        "item_index",
        "item_order",
        "role_order",
        "prompt_format_used",
        "source_path",
    ]
    metadata_cols = [column for column in metadata_cols if column in item_level.columns]

    rows: List[Dict[str, object]] = []
    candidate_specs = [
        ("candidate_a", "active"),
        ("candidate_b", "passive"),
    ]

    for row_number, row in item_level.iterrows():
        base = {column: row.get(column) for column in metadata_cols}
        base["analysis_row_id"] = int(row_number)

        for prefix, fallback_voice in candidate_specs:
            tokens = parse_serialized_list(row.get(f"{prefix}_tokens"))
            token_logprobs = parse_serialized_list(row.get(f"{prefix}_token_logprobs"))
            token_ids = parse_serialized_list(row.get(f"{prefix}_token_ids"))
            if not tokens or not token_logprobs:
                continue

            scored_len = min(len(tokens), len(token_logprobs))
            target_voice = normalize_candidate_voice(row.get(f"{prefix}_label"), fallback_voice)
            candidate_text = row.get(f"{prefix}_text", "")
            words = candidate_words(candidate_text)
            token_word_indices = align_tokens_to_word_indices(tokens[:scored_len], words)
            candidate_total = pd.to_numeric(pd.Series([row.get(f"{prefix}_total_logprob")]), errors="coerce").iloc[0]
            candidate_mean = pd.to_numeric(pd.Series([row.get(f"{prefix}_mean_logprob")]), errors="coerce").iloc[0]

            for token_zero_index in range(scored_len):
                token_logprob = float(token_logprobs[token_zero_index])
                token_position = token_zero_index + 1
                token_id = token_ids[token_zero_index] if token_zero_index < len(token_ids) else np.nan
                word_index = token_word_indices[token_zero_index]
                rows.append(
                    {
                        **base,
                        "candidate_slot": prefix,
                        "candidate_label": row.get(f"{prefix}_label"),
                        "candidate_voice": target_voice,
                        "candidate_text": candidate_text,
                        "candidate_total_logprob": candidate_total,
                        "candidate_mean_logprob": candidate_mean,
                        "token_index": token_zero_index,
                        "token_position": token_position,
                        "token_count": scored_len,
                        "token_id": token_id,
                        "token": tokens[token_zero_index],
                        "word_index": word_index,
                        "word": words[word_index] if 0 <= word_index < len(words) else "",
                        "token_logprob": token_logprob,
                        "token_probability": float(np.exp(token_logprob)),
                        "token_surprisal": float(-token_logprob),
                        "roi": roi_label_for_word(target_voice, word_index),
                    }
                )

    return pd.DataFrame(rows)


def build_token_roi_summary(token_level: pd.DataFrame) -> pd.DataFrame:
    if token_level.empty:
        return pd.DataFrame()

    group_cols = [
        "experiment",
        "task_type",
        "dataset",
        "source_model_run",
        "model_condition",
        "model_size",
        "model_family",
        "model_instruct",
        "model_label",
        "lexicality_condition",
        "prime_condition",
        "candidate_voice",
        "roi",
        "item_id",
        "item_index",
        "item_order",
    ]
    group_cols = [column for column in group_cols if column in token_level.columns]
    return (
        token_level.groupby(group_cols, dropna=False)
        .agg(
            roi_logprob_sum=("token_logprob", "sum"),
            roi_logprob_mean=("token_logprob", "mean"),
            roi_surprisal_sum=("token_surprisal", "sum"),
            roi_surprisal_mean=("token_surprisal", "mean"),
            roi_token_count=("token_logprob", "size"),
        )
        .reset_index()
    )


def build_exp2_generation() -> pd.DataFrame:
    frames = read_csvs(exp2_reviewed_paths(), experiment="experiment_2", task_type="free_generation")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["generation_voice"] = out.get("generation_class_lax_final", out.get("generation_class_lax", "other"))
    out["generation_voice_strict"] = out.get(
        "generation_class_strict_final",
        out.get("generation_class_strict", "other"),
    )
    out["is_active_lax"] = out["generation_voice"].eq("active").astype(int)
    out["is_passive_lax"] = out["generation_voice"].eq("passive").astype(int)
    out["is_other_lax"] = out["generation_voice"].eq("other").astype(int)
    out["is_active_strict"] = out["generation_voice_strict"].eq("active").astype(int)
    out["is_passive_strict"] = out["generation_voice_strict"].eq("passive").astype(int)
    out["is_other_strict"] = out["generation_voice_strict"].eq("other").astype(int)
    return out


def build_exp4_comprehension() -> pd.DataFrame:
    frames = read_csvs(exp4_comprehension_paths(), experiment="experiment_4", task_type="role_recovery")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["is_correct"] = out["is_correct"].astype(int)
    out["is_foil"] = out["is_foil"].astype(int)
    return out


def build_exp4_pe() -> pd.DataFrame:
    frames = read_csvs(exp4_pe_paths(), experiment="experiment_4", task_type="target_processing_pe")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["pe_passive_minus_active_sum"] = out.get("pe_logprob_sum_imbalance_passive_minus_active")
    out["pe_passive_minus_active_mean"] = out.get("pe_logprob_mean_imbalance_passive_minus_active")
    return out


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {path.relative_to(REPO_ROOT)} rows={len(frame)} cols={len(frame.columns)}")


def write_manifest(outputs: Dict[str, pd.DataFrame]) -> None:
    manifest = {
        name: {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
        }
        for name, frame in outputs.items()
    }
    path = OUTPUT_DIR / "analysis_ready_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path.relative_to(REPO_ROOT)}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    item_level = build_item_level_master()
    baseline_shifts = build_baseline_shifts(item_level)
    token_level = build_token_level_probabilities(item_level)
    token_roi_summary = build_token_roi_summary(token_level)
    exp2_generation = build_exp2_generation()
    exp4_comprehension = build_exp4_comprehension()
    exp4_pe = build_exp4_pe()

    outputs = {
        "item_level_master.csv": item_level,
        "baseline_centered_logodds_shifts.csv": baseline_shifts,
        "token_level_probabilities.csv": token_level,
        "token_roi_summary.csv": token_roi_summary,
        "exp2_generation_item_level.csv": exp2_generation,
        "exp4_comprehension_item_level.csv": exp4_comprehension,
        "exp4_sinclair_pe_item_level.csv": exp4_pe,
    }
    for filename, frame in outputs.items():
        write_csv(frame, OUTPUT_DIR / filename)
    write_manifest(outputs)

    if not item_level.empty:
        print("\nTeacher-forced rows by experiment/model/lexicality:")
        print(
            item_level.groupby(["experiment", "source_model_run", "lexicality_condition"], dropna=False)
            .size()
            .to_string()
        )
    if not exp2_generation.empty:
        print("\nExp2 selected reviewed paths:")
        for path in exp2_reviewed_paths():
            print(f"- {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
