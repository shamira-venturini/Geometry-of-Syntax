from __future__ import annotations

import argparse
import json
import logging
import platform
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from src.analysis import run_analysis
from src.data import DatasetBundle, ExperimentItem, load_dataset_from_experiment_config
from src.models import CausalLMWrapper, ModelConfig
from src.plots import save_all_default_plots
from src.prompts import (
    question_templates_for_item,
    render_experiment_1_prompt,
    render_experiment_2_prompt,
)
from src.scoring import score_candidates_batched


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairwiseRequest:
    metadata: Dict[str, object]
    prefix: str
    candidate_a: str
    candidate_b: str
    candidate_a_label: str
    candidate_b_label: str


def _safe_token_value(values: Sequence[float], index: int) -> float:
    if not values:
        return float("nan")
    if index < 0:
        index = len(values) + index
    if index < 0 or index >= len(values):
        return float("nan")
    return float(values[index])


def _shared_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    limit = min(len(a), len(b))
    count = 0
    while count < limit and int(a[count]) == int(b[count]):
        count += 1
    return count


def _location_metrics(
    *,
    token_ids_a: Sequence[int],
    token_ids_b: Sequence[int],
    token_logprobs_a: Sequence[float],
    token_logprobs_b: Sequence[float],
) -> Dict[str, object]:
    shared_prefix = _shared_prefix_len(token_ids_a, token_ids_b)
    divergence_index = shared_prefix if shared_prefix < min(len(token_ids_a), len(token_ids_b)) else -1
    aligned_length = min(len(token_logprobs_a), len(token_logprobs_b))
    aligned_diffs = [
        float(token_logprobs_a[idx] - token_logprobs_b[idx])
        for idx in range(aligned_length)
    ]

    if aligned_diffs:
        aligned_mean = float(np.mean(aligned_diffs))
        aligned_first = float(aligned_diffs[0])
        aligned_last = float(aligned_diffs[-1])
    else:
        aligned_mean = float("nan")
        aligned_first = float("nan")
        aligned_last = float("nan")

    if divergence_index >= 0:
        divergence_a = _safe_token_value(token_logprobs_a, divergence_index)
        divergence_b = _safe_token_value(token_logprobs_b, divergence_index)
        divergence_diff = divergence_a - divergence_b
    else:
        divergence_a = float("nan")
        divergence_b = float("nan")
        divergence_diff = float("nan")

    return {
        "shared_prefix_token_count": int(shared_prefix),
        "divergence_token_index": int(divergence_index),
        "candidate_a_first_token_logprob": _safe_token_value(token_logprobs_a, 0),
        "candidate_b_first_token_logprob": _safe_token_value(token_logprobs_b, 0),
        "candidate_a_second_token_logprob": _safe_token_value(token_logprobs_a, 1),
        "candidate_b_second_token_logprob": _safe_token_value(token_logprobs_b, 1),
        "candidate_a_last_token_logprob": _safe_token_value(token_logprobs_a, -1),
        "candidate_b_last_token_logprob": _safe_token_value(token_logprobs_b, -1),
        "candidate_a_divergence_token_logprob": divergence_a,
        "candidate_b_divergence_token_logprob": divergence_b,
        "preference_first_token": _safe_token_value(token_logprobs_a, 0) - _safe_token_value(token_logprobs_b, 0),
        "preference_second_token": _safe_token_value(token_logprobs_a, 1) - _safe_token_value(token_logprobs_b, 1),
        "preference_last_token": _safe_token_value(token_logprobs_a, -1) - _safe_token_value(token_logprobs_b, -1),
        "preference_divergence_token": divergence_diff,
        "preference_aligned_mean": aligned_mean,
        "preference_aligned_first": aligned_first,
        "preference_aligned_last": aligned_last,
        "preference_tokenwise_aligned_diffs": _token_debug_json(aligned_diffs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 3 controlled disambiguation with deterministic structural scoring."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional override for output directory.")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_candidate_join(prefix: str, candidate: str) -> str:
    if not candidate:
        return candidate
    if candidate[0].isspace():
        return candidate
    if not prefix:
        return candidate
    if prefix.endswith((" ", "\n", "\t", '"', "'")):
        return candidate
    return " " + candidate


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return payload


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    handlers: List[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def _software_versions() -> Dict[str, str]:
    import matplotlib
    import scipy
    import transformers

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _resolve_no_prime_text(
    *,
    experiment_cfg: Dict[str, object],
    model_cfg: ModelConfig,
    model_wrapper: CausalLMWrapper,
) -> str:
    mode = str(
        experiment_cfg.get(
            "no_prime_mode",
            experiment_cfg.get("no_prime_eos_mode", "empty"),
        )
    ).strip().lower()
    if mode == "tokenizer_eos":
        eos_text = model_wrapper.eos_token_text or ""
    elif mode == "empty":
        eos_text = ""
    elif mode == "custom_text":
        eos_text = str(experiment_cfg.get("no_prime_text", experiment_cfg.get("no_prime_eos_text", "")))
    else:
        raise ValueError(
            f"Unsupported experiment.no_prime_mode='{mode}'. "
            "Use tokenizer_eos, empty, or custom_text."
        )

    overrides = experiment_cfg.get("no_prime_overrides", experiment_cfg.get("no_prime_eos_overrides", {}))
    if isinstance(overrides, dict):
        if model_cfg.model_condition in overrides:
            eos_text = str(overrides[model_cfg.model_condition])
        elif model_cfg.name in overrides:
            eos_text = str(overrides[model_cfg.name])
    return eos_text


def _build_model_configs(config: Dict[str, object]) -> List[ModelConfig]:
    model_entries = config.get("models", [])
    if not isinstance(model_entries, list) or not model_entries:
        raise ValueError("Config must define at least one model entry under 'models'.")

    model_configs: List[ModelConfig] = []
    for entry in model_entries:
        if not isinstance(entry, dict):
            raise ValueError("Each model entry must be a mapping.")
        model_configs.append(
            ModelConfig(
                name=str(entry["name"]),
                model_condition=str(entry.get("model_condition", entry["name"])),
                device=str(entry.get("device", "cuda")),
                torch_dtype=str(entry.get("torch_dtype", "auto")),
                batch_size=int(entry.get("batch_size", 4)),
                max_length=int(entry.get("max_length", 1024)),
                local_files_only=bool(entry.get("local_files_only", False)),
                use_chat_template=bool(entry.get("use_chat_template", False)),
                prompt_style=str(entry.get("prompt_style", "plain")),
            )
        )
    return model_configs


def _base_metadata_for_item(item: ExperimentItem, model_cfg: ModelConfig) -> Dict[str, object]:
    return {
        "experiment_id": "experiment_3",
        "item_id": item.item_id,
        "item_model_condition": item.model_condition,
        "model_name": model_cfg.name,
        "model_condition": model_cfg.model_condition,
        "prime_condition": item.prime_condition,
        "prime_text": item.prime_text,
        "lexicality_condition": item.lexicality_condition,
        "notes": item.notes,
    }


def build_experiment_1_requests(
    items: Sequence[ExperimentItem],
    model_cfg: ModelConfig,
    eos_token_text: str,
) -> List[PairwiseRequest]:
    requests: List[PairwiseRequest] = []
    for item in items:
        prefix = render_experiment_1_prompt(item=item, eos_token_text=eos_token_text)
        active_candidate = _safe_candidate_join(prefix, item.active_completion)
        passive_candidate = _safe_candidate_join(prefix, item.passive_completion)

        metadata = _base_metadata_for_item(item=item, model_cfg=model_cfg)
        metadata.update(
            {
                "task": "experiment_3_production_disambiguation",
                "task_short": "experiment_3_production",
                "task_family": "production_like_disambiguation",
                "pairing_key": item.item_id,
                "question_template_used": "",
                "target_voice": "",
                "target_sentence_used": "",
            }
        )
        requests.append(
            PairwiseRequest(
                metadata=metadata,
                prefix=prefix,
                candidate_a=active_candidate,
                candidate_b=passive_candidate,
                candidate_a_label="active_completion",
                candidate_b_label="passive_completion",
            )
        )
    return requests


def build_experiment_2_requests(
    items: Sequence[ExperimentItem],
    model_cfg: ModelConfig,
    eos_token_text: str,
    additional_question_templates: Iterable[str],
) -> List[PairwiseRequest]:
    requests: List[PairwiseRequest] = []
    for item in items:
        templates = question_templates_for_item(item=item, additional_templates=additional_question_templates)
        for question_template in templates:
            for target_voice in ("active", "passive"):
                if target_voice == "active":
                    target_sentence = item.target_sentence_active
                    correct = item.correct_answer_for_active
                    incorrect = item.incorrect_answer_for_active
                else:
                    target_sentence = item.target_sentence_passive
                    correct = item.correct_answer_for_passive
                    incorrect = item.incorrect_answer_for_passive

                prefix = render_experiment_2_prompt(
                    item=item,
                    target_sentence=target_sentence,
                    question_template=question_template,
                    eos_token_text=eos_token_text,
                )

                correct_candidate = _safe_candidate_join(prefix, correct)
                incorrect_candidate = _safe_candidate_join(prefix, incorrect)

                pairing_key = f"{item.item_id}|{target_voice}|{question_template}"
                metadata = _base_metadata_for_item(item=item, model_cfg=model_cfg)
                metadata.update(
                    {
                        "task": "experiment_3_comprehension_disambiguation",
                        "task_short": "experiment_3_comprehension",
                        "task_family": "comprehension_role_commitment",
                        "pairing_key": pairing_key,
                        "question_template_used": question_template,
                        "target_voice": target_voice,
                        "target_sentence_used": target_sentence,
                    }
                )
                requests.append(
                    PairwiseRequest(
                        metadata=metadata,
                        prefix=prefix,
                        candidate_a=correct_candidate,
                        candidate_b=incorrect_candidate,
                        candidate_a_label="correct_answer",
                        candidate_b_label="incorrect_answer",
                    )
                )
    return requests


def _token_debug_json(values: Sequence[object]) -> str:
    return json.dumps(list(values), ensure_ascii=True)


def score_requests(
    requests: Sequence[PairwiseRequest],
    model_wrapper: CausalLMWrapper,
    batch_size: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for start in range(0, len(requests), batch_size):
        chunk = requests[start : start + batch_size]
        prefixes: List[str] = []
        candidates: List[str] = []
        for request in chunk:
            prefixes.extend([request.prefix, request.prefix])
            candidates.extend([request.candidate_a, request.candidate_b])

        scores = score_candidates_batched(
            model_wrapper=model_wrapper,
            prefixes=prefixes,
            candidates=candidates,
        )

        for request_index, request in enumerate(chunk):
            score_a = scores[2 * request_index]
            score_b = scores[2 * request_index + 1]

            row = dict(request.metadata)
            row.update(
                {
                    "prompt_format_used": score_a.prompt_format,
                    "prompt_text": request.prefix,
                    "prompt_token_count": score_a.prefix_token_count,
                    "candidate_a_label": request.candidate_a_label,
                    "candidate_a_text": request.candidate_a,
                    "candidate_a_total_logprob": score_a.total_logprob,
                    "candidate_a_mean_logprob": score_a.mean_logprob,
                    "candidate_a_token_count": score_a.num_candidate_tokens,
                    "candidate_a_token_ids": _token_debug_json(score_a.candidate_token_ids),
                    "candidate_a_tokens": _token_debug_json(score_a.candidate_tokens),
                    "candidate_a_token_logprobs": _token_debug_json(score_a.candidate_token_logprobs),
                    "candidate_b_label": request.candidate_b_label,
                    "candidate_b_text": request.candidate_b,
                    "candidate_b_total_logprob": score_b.total_logprob,
                    "candidate_b_mean_logprob": score_b.mean_logprob,
                    "candidate_b_token_count": score_b.num_candidate_tokens,
                    "candidate_b_token_ids": _token_debug_json(score_b.candidate_token_ids),
                    "candidate_b_tokens": _token_debug_json(score_b.candidate_tokens),
                    "candidate_b_token_logprobs": _token_debug_json(score_b.candidate_token_logprobs),
                    "preference_total": score_a.total_logprob - score_b.total_logprob,
                    "preference_mean": score_a.mean_logprob - score_b.mean_logprob,
                }
            )
            row.update(
                _location_metrics(
                    token_ids_a=score_a.candidate_token_ids,
                    token_ids_b=score_b.candidate_token_ids,
                    token_logprobs_a=score_a.candidate_token_logprobs,
                    token_logprobs_b=score_b.candidate_token_logprobs,
                )
            )

            if request.candidate_a_label == "active_completion" and request.candidate_b_label == "passive_completion":
                row["active_minus_passive_logprob_total"] = row["preference_total"]
                row["active_minus_passive_logprob_mean"] = row["preference_mean"]
            else:
                row["answer_preference_logprob_total"] = row["preference_total"]
                row["answer_preference_logprob_mean"] = row["preference_mean"]

            rows.append(row)

    return rows


def run_for_model(
    dataset: DatasetBundle,
    model_cfg: ModelConfig,
    experiment_cfg: Dict[str, object],
    prompting_cfg: Dict[str, object],
) -> List[Dict[str, object]]:
    force_bos = bool(experiment_cfg.get("force_bos", False))
    append_eos_to_candidate = bool(experiment_cfg.get("append_eos_to_candidate", False))
    run_experiment_1 = bool(
        experiment_cfg.get(
            "run_experiment_3_production",
            experiment_cfg.get("run_experiment_1", True),
        )
    )
    run_experiment_2 = bool(
        experiment_cfg.get(
            "run_experiment_3_comprehension",
            experiment_cfg.get("run_experiment_2", True),
        )
    )

    wrapper = CausalLMWrapper(
        config=model_cfg,
        force_bos=force_bos,
        append_eos_to_candidate=append_eos_to_candidate,
    )

    no_prime_text = _resolve_no_prime_text(
        experiment_cfg=experiment_cfg,
        model_cfg=model_cfg,
        model_wrapper=wrapper,
    )
    LOGGER.info(
        "Using no-prime prefix text for model=%s model_condition=%s repr=%r",
        model_cfg.name,
        model_cfg.model_condition,
        no_prime_text,
    )
    all_rows: List[Dict[str, object]] = []

    if run_experiment_1:
        requests_1 = build_experiment_1_requests(
            items=dataset.items,
            model_cfg=model_cfg,
            eos_token_text=no_prime_text,
        )
        LOGGER.info(
            "Scoring Experiment 3 production-like requests for %s: %d",
            model_cfg.name,
            len(requests_1),
        )
        all_rows.extend(
            score_requests(
                requests=requests_1,
                model_wrapper=wrapper,
                batch_size=max(1, int(model_cfg.batch_size)),
            )
        )

    if run_experiment_2:
        additional_templates = prompting_cfg.get("additional_question_templates", [])
        if not isinstance(additional_templates, list):
            raise ValueError("prompting.additional_question_templates must be a list")

        requests_2 = build_experiment_2_requests(
            items=dataset.items,
            model_cfg=model_cfg,
            eos_token_text=no_prime_text,
            additional_question_templates=[str(template) for template in additional_templates],
        )
        LOGGER.info(
            "Scoring Experiment 3 comprehension-like requests for %s: %d",
            model_cfg.name,
            len(requests_2),
        )
        all_rows.extend(
            score_requests(
                requests=requests_2,
                model_wrapper=wrapper,
                batch_size=max(1, int(model_cfg.batch_size)),
            )
        )

    return all_rows


def write_outputs(
    output_dir: Path,
    item_level: pd.DataFrame,
    analysis_outputs: Dict[str, pd.DataFrame],
    create_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    item_level.to_csv(output_dir / "item_level_results.csv", index=False)

    for key, frame in analysis_outputs.items():
        frame.to_csv(output_dir / f"{key}.csv", index=False)

    # Required summary file names.
    analysis_outputs["summary_by_model_condition"].to_csv(
        output_dir / "summary_by_model_condition.csv", index=False
    )
    analysis_outputs["summary_by_prime_condition"].to_csv(
        output_dir / "summary_by_prime_condition.csv", index=False
    )
    analysis_outputs["summary_by_lexicality"].to_csv(
        output_dir / "summary_by_lexicality.csv", index=False
    )
    analysis_outputs["summary_by_task"].to_csv(output_dir / "summary_by_task.csv", index=False)

    if create_plots:
        save_all_default_plots(item_level=item_level, output_dir=output_dir)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path)

    experiment_cfg = config.get("experiment", {})
    prompting_cfg = config.get("prompting", {})

    if not isinstance(experiment_cfg, dict):
        raise ValueError("Config key 'experiment' must be a mapping.")
    if not isinstance(prompting_cfg, dict):
        raise ValueError("Config key 'prompting' must be a mapping.")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else Path(str(experiment_cfg.get("output_dir", "behavioral_results/experiment-3"))).resolve()
    )
    _setup_logging(output_dir)

    seed = int(experiment_cfg.get("seed", 13))
    set_global_seed(seed)

    dataset = load_dataset_from_experiment_config(experiment_cfg)
    model_configs = _build_model_configs(config)

    LOGGER.info("Starting Experiment 3 controlled disambiguation run")
    LOGGER.info("Config path: %s", config_path)
    LOGGER.info("Output dir: %s", output_dir)
    LOGGER.info("Dataset rows: %d", len(dataset.items))
    LOGGER.info("Models: %d", len(model_configs))

    all_rows: List[Dict[str, object]] = []
    for model_cfg in model_configs:
        LOGGER.info("Running model: %s (%s)", model_cfg.name, model_cfg.model_condition)
        model_rows = run_for_model(
            dataset=dataset,
            model_cfg=model_cfg,
            experiment_cfg=experiment_cfg,
            prompting_cfg=prompting_cfg,
        )
        all_rows.extend(model_rows)

    if not all_rows:
        raise RuntimeError(
            "No results produced. Check run_experiment_3_production/run_experiment_3_comprehension flags."
        )

    item_level = pd.DataFrame(all_rows)
    n_bootstrap = int(experiment_cfg.get("bootstrap_resamples", 5000))
    bootstrap_ci = float(experiment_cfg.get("bootstrap_ci", 95.0))
    create_plots = bool(experiment_cfg.get("create_plots", True))

    analysis_outputs = run_analysis(
        item_level=item_level,
        n_bootstrap=n_bootstrap,
        bootstrap_ci=bootstrap_ci,
        seed=seed,
    )

    write_outputs(
        output_dir=output_dir,
        item_level=item_level,
        analysis_outputs=analysis_outputs,
        create_plots=create_plots,
    )

    metadata = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config_path": str(config_path),
        "dataset_path": str(dataset.path),
        "seed": seed,
        "software_versions": _software_versions(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    LOGGER.info("Run complete. Results written to %s", output_dir)


if __name__ == "__main__":
    main()
