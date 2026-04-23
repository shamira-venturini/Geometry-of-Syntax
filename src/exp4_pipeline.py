from __future__ import annotations

import json
import logging
import platform
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from .data import DatasetBundle, ExperimentItem, load_dataset_from_experiment_config
from .exp4_analysis import run_exp4_analysis
from .exp4_generation import greedy_generate_answers
from .exp4_prompts import build_exp4_prompt
from .exp4_scoring import evaluate_generated_answer
from .models import CausalLMWrapper, ModelConfig


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Exp4Request:
    item: ExperimentItem
    target_voice: str
    prompt_text: str
    target_sentence: str
    question_text: str
    correct_answer: str
    foil_answer: str
    verb_lemma: str
    verb_nominalized: str
    nominalization_fallback_used: bool


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return payload


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_exp4.log"
    handlers: List[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _software_versions() -> Dict[str, str]:
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
    }


def _build_model_configs(config: Mapping[str, object]) -> List[ModelConfig]:
    model_entries = config.get("models", [])
    if not isinstance(model_entries, list) or not model_entries:
        raise ValueError("Config must define at least one model entry under 'models'.")

    model_configs: List[ModelConfig] = []
    for entry in model_entries:
        if not isinstance(entry, Mapping):
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


def _filter_model_configs(
    model_configs: Sequence[ModelConfig],
    model_filters: Sequence[str] | None,
) -> List[ModelConfig]:
    if not model_filters:
        return list(model_configs)

    requested = [token.strip() for token in model_filters if str(token).strip()]
    if not requested:
        return list(model_configs)

    filtered = [
        cfg
        for cfg in model_configs
        if cfg.name in requested or cfg.model_condition in requested
    ]
    if filtered:
        return filtered

    available = ", ".join(f"{cfg.name} ({cfg.model_condition})" for cfg in model_configs)
    raise ValueError(
        "No models matched --model-filter values "
        f"{requested}. Available: {available}"
    )


def _item_index_from_item_id(item_id: str) -> int | None:
    suffix = str(item_id).rsplit("_", maxsplit=1)
    if len(suffix) != 2:
        return None
    try:
        return int(suffix[1])
    except Exception:
        return None


def _build_requests(
    *,
    items: Sequence[ExperimentItem],
    question_template: str,
    nominalization_overrides: Mapping[str, str] | None,
    answer_prefix: str,
) -> List[Exp4Request]:
    requests: List[Exp4Request] = []

    for item in items:
        for target_voice in ("active", "passive"):
            prompt = build_exp4_prompt(
                item=item,
                target_voice=target_voice,
                question_template=question_template,
                nominalization_overrides=nominalization_overrides,
                answer_prefix=answer_prefix,
                logger=LOGGER,
            )

            if target_voice == "active":
                correct_answer = item.correct_answer_for_active
                foil_answer = item.incorrect_answer_for_active
            else:
                correct_answer = item.correct_answer_for_passive
                foil_answer = item.incorrect_answer_for_passive

            requests.append(
                Exp4Request(
                    item=item,
                    target_voice=target_voice,
                    prompt_text=prompt.prompt_text,
                    target_sentence=prompt.target_sentence,
                    question_text=prompt.question_text,
                    correct_answer=correct_answer,
                    foil_answer=foil_answer,
                    verb_lemma=prompt.verb_lemma,
                    verb_nominalized=prompt.verb_nominalized,
                    nominalization_fallback_used=prompt.nominalization_fallback_used,
                )
            )

    return requests


def _score_requests_for_model(
    *,
    dataset: DatasetBundle,
    model_cfg: ModelConfig,
    question_template: str,
    nominalization_overrides: Mapping[str, str] | None,
    answer_prefix: str,
    max_new_tokens: int,
) -> List[Dict[str, object]]:
    wrapper = CausalLMWrapper(config=model_cfg)
    requests = _build_requests(
        items=dataset.items,
        question_template=question_template,
        nominalization_overrides=nominalization_overrides,
        answer_prefix=answer_prefix,
    )

    LOGGER.info("Exp4 generating answers for model=%s n_requests=%d", model_cfg.name, len(requests))
    completions = greedy_generate_answers(
        model_wrapper=wrapper,
        prompts=[request.prompt_text for request in requests],
        batch_size=max(1, int(model_cfg.batch_size)),
        max_new_tokens=max(1, int(max_new_tokens)),
    )

    rows: List[Dict[str, object]] = []
    for request, generated_raw in zip(requests, completions):
        evaluation = evaluate_generated_answer(
            generated_answer_raw=generated_raw,
            correct_answer=request.correct_answer,
            foil_answer=request.foil_answer,
        )
        rows.append(
            {
                "experiment_id": "experiment_4",
                "item_id": request.item.item_id,
                "item_index": _item_index_from_item_id(request.item.item_id),
                "prime_condition": request.item.prime_condition,
                "target_voice": request.target_voice,
                "lexicality_condition": request.item.lexicality_condition,
                "model_name": model_cfg.name,
                "model_condition": model_cfg.model_condition,
                "prompt_text": request.prompt_text,
                "target_sentence": request.target_sentence,
                "question_text": request.question_text,
                "generated_answer_raw": generated_raw,
                "generated_answer_normalized": evaluation.generated_answer_normalized,
                "correct_answer": request.correct_answer,
                "foil_answer": request.foil_answer,
                "matched_label": evaluation.matched_label,
                "is_correct": bool(evaluation.is_correct),
                "is_foil": bool(evaluation.matched_label == "foil"),
                "verb_lemma": request.verb_lemma,
                "verb_nominalized": request.verb_nominalized,
                "nominalization_fallback_used": bool(request.nominalization_fallback_used),
            }
        )

    return rows


def run_experiment_4(
    *,
    config_path: Path,
    output_dir_override: Path | None = None,
    model_filters: Sequence[str] | None = None,
) -> None:
    config = _load_yaml(config_path)
    experiment_cfg = config.get("experiment", {})
    if not isinstance(experiment_cfg, Mapping):
        raise ValueError("Config key 'experiment' must be a mapping.")

    output_dir = (
        output_dir_override.expanduser().resolve()
        if output_dir_override is not None
        else Path(str(experiment_cfg.get("output_dir", "behavioral_results/experiment-4"))).resolve()
    )
    _setup_logging(output_dir)

    seed = int(experiment_cfg.get("seed", 13))
    _set_seed(seed)

    question_template = str(experiment_cfg.get("question_template", "Who did the {nominalized_verb}?"))
    answer_prefix = str(experiment_cfg.get("answer_prefix", "The"))
    max_new_tokens = int(experiment_cfg.get("max_new_tokens", 10))
    ceiling_threshold = float(experiment_cfg.get("ceiling_threshold", 0.95))

    nominalization_overrides_cfg = experiment_cfg.get("nominalization_overrides", {})
    nominalization_overrides = (
        nominalization_overrides_cfg
        if isinstance(nominalization_overrides_cfg, Mapping)
        else {}
    )

    dataset = load_dataset_from_experiment_config(experiment_cfg)
    model_configs = _filter_model_configs(
        _build_model_configs(config),
        model_filters,
    )

    LOGGER.info("Starting Experiment 4 Ferreira-style comprehension probe")
    LOGGER.info("Config path: %s", config_path)
    LOGGER.info("Output dir: %s", output_dir)
    LOGGER.info("Dataset rows (items x prime_conditions): %d", len(dataset.items))
    LOGGER.info("Models: %d", len(model_configs))

    all_rows: List[Dict[str, object]] = []
    for model_cfg in model_configs:
        LOGGER.info("Running model: %s (%s)", model_cfg.name, model_cfg.model_condition)
        model_rows = _score_requests_for_model(
            dataset=dataset,
            model_cfg=model_cfg,
            question_template=question_template,
            nominalization_overrides=nominalization_overrides,
            answer_prefix=answer_prefix,
            max_new_tokens=max_new_tokens,
        )
        all_rows.extend(model_rows)

    if not all_rows:
        raise RuntimeError("No Experiment 4 rows produced.")

    item_level = pd.DataFrame(all_rows)
    analysis_outputs = run_exp4_analysis(item_level, ceiling_threshold=ceiling_threshold)

    item_level.to_csv(output_dir / "item_level_results_exp4.csv", index=False)
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

    metadata = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config_path": str(config_path),
        "dataset_path": str(dataset.path),
        "seed": seed,
        "model_filter": list(model_filters or []),
        "question_template": question_template,
        "answer_prefix": answer_prefix,
        "max_new_tokens": max_new_tokens,
        "ceiling_threshold": ceiling_threshold,
        "software_versions": _software_versions(),
    }
    (output_dir / "run_metadata_exp4.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "resolved_config_exp4.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    fallback_count = int(item_level["nominalization_fallback_used"].sum())
    LOGGER.info("Exp4 nominalization fallbacks used: %d", fallback_count)
    LOGGER.info("Experiment 4 complete. Results written to %s", output_dir)
