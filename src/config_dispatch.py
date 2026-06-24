from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXPERIMENTS = ("exp1a", "exp1b", "exp2", "exp3", "exp4")
LEGACY_EXPERIMENTS = ("exp1a", "exp1b", "exp2")


@dataclass(frozen=True)
class CommandPlan:
    label: str
    command: List[str]
    output_dir: Path | None = None


def load_config(path: Path) -> Dict[str, object]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {resolved}")
    return payload


def validate_common_config(
    config: Mapping[str, object],
    experiment_id: str,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    if experiment_id not in SUPPORTED_EXPERIMENTS:
        raise ValueError(f"Unsupported experiment '{experiment_id}'.")

    schema_version = int(config.get("schema_version", 1))
    if schema_version != 1:
        raise ValueError(f"Unsupported config schema_version {schema_version}; expected 1.")

    experiment_raw = config.get("experiment")
    if not isinstance(experiment_raw, Mapping):
        raise ValueError("Config key 'experiment' must be a mapping.")
    experiment = dict(experiment_raw)

    declared_id = str(experiment.get("id", experiment_id)).strip().lower()
    if declared_id != experiment_id:
        raise ValueError(
            f"Config declares experiment.id='{declared_id}', but --experiment is '{experiment_id}'."
        )

    models_raw = config.get("models")
    if not isinstance(models_raw, list) or not models_raw:
        raise ValueError("Config must define at least one model entry under 'models'.")

    models: List[Dict[str, object]] = []
    for index, entry in enumerate(models_raw):
        if not isinstance(entry, Mapping):
            raise ValueError(f"models[{index}] must be a mapping.")
        model = dict(entry)
        name = str(model.get("name", "")).strip()
        if not name:
            raise ValueError(f"models[{index}].name must be non-empty.")
        model["name"] = name
        model["model_condition"] = str(model.get("model_condition", name)).strip()
        models.append(model)

    return experiment, models


def select_model_entries(
    models: Sequence[Dict[str, object]],
    model_filters: Sequence[str] | None,
) -> List[Dict[str, object]]:
    if not model_filters:
        return list(models)
    requested = {str(token).strip() for token in model_filters if str(token).strip()}
    if not requested:
        return list(models)
    selected = [
        model
        for model in models
        if str(model["name"]) in requested or str(model["model_condition"]) in requested
    ]
    if selected:
        return selected
    available = ", ".join(
        f"{model['name']} ({model['model_condition']})" for model in models
    )
    raise ValueError(f"No models matched {sorted(requested)}. Available: {available}")


def _resolved_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _base_output_dir(
    experiment: Mapping[str, object],
    default: str,
    output_dir_override: Path | None,
) -> Path:
    if output_dir_override is not None:
        return output_dir_override.expanduser().resolve()
    return _resolved_path(experiment.get("output_dir", default))


def _model_output_dir(base: Path, model: Mapping[str, object], model_count: int) -> Path:
    if model_count == 1:
        return base
    return base / str(model["model_condition"])


def _append_switch(command: List[str], enabled: object, flag: str) -> None:
    if bool(enabled):
        command.append(flag)


def _common_model_args(
    command: List[str],
    model: Mapping[str, object],
    *,
    default_batch_size: int,
) -> None:
    command.extend(["--model-name", str(model["name"])])
    device = model.get("device")
    if device not in (None, ""):
        command.extend(["--device", str(device)])
    command.extend(["--torch-dtype", str(model.get("torch_dtype", "auto"))])
    command.extend(["--batch-size", str(int(model.get("batch_size", default_batch_size)))])
    _append_switch(command, model.get("local_files_only", False), "--local-files-only")


def _build_exp1a_plan(
    experiment: Mapping[str, object],
    models: Sequence[Dict[str, object]],
    output_dir_override: Path | None,
) -> tuple[Path, List[CommandPlan]]:
    base = _base_output_dir(
        experiment,
        "behavioral_results/experiment-1/experiment-1a/transitive_token_profiles",
        output_dir_override,
    )
    plans: List[CommandPlan] = []
    for model in models:
        output_dir = _model_output_dir(base, model, len(models))
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/experiments/10_run_colab_experiment_1a.py"),
        ]
        _common_model_args(command, model, default_batch_size=32)
        command.extend(
            [
                "--max-items",
                str(int(experiment.get("max_items", 15000))),
                "--preset",
                str(experiment.get("preset", "paper_main")),
                "--seed",
                str(int(experiment.get("seed", 13))),
                "--output-root",
                str(output_dir),
            ]
        )
        _append_switch(command, experiment.get("skip_report", False), "--skip-report")
        _append_switch(command, experiment.get("skip_stats", False), "--skip-stats")
        plans.append(CommandPlan(f"exp1a:{model['model_condition']}", command, output_dir))
    return base, plans


def _conditions_to_which(conditions: object) -> str:
    if conditions is None:
        return "both"
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("experiment.conditions must be a non-empty list.")
    requested = {str(value) for value in conditions}
    valid = {"core", "jabberwocky"}
    invalid = requested - valid
    if invalid:
        raise ValueError(f"Invalid Experiment 1b conditions: {sorted(invalid)}")
    return "both" if requested == valid else next(iter(requested))


def _build_exp1b_plan(
    experiment: Mapping[str, object],
    models: Sequence[Dict[str, object]],
    output_dir_override: Path | None,
) -> tuple[Path, List[CommandPlan]]:
    base = _base_output_dir(
        experiment,
        "behavioral_results/experiment-1/experiment-1b/default_run",
        output_dir_override,
    )
    which = _conditions_to_which(experiment.get("conditions", ["core", "jabberwocky"]))
    prime_conditions = experiment.get(
        "prime_conditions", ["active", "passive", "no_prime", "filler"]
    )
    if not isinstance(prime_conditions, list) or not prime_conditions:
        raise ValueError("experiment.prime_conditions must be a non-empty list.")

    plans: List[CommandPlan] = []
    for model in models:
        output_dir = _model_output_dir(base, model, len(models))
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/experiments/16_run_processing_experiment_1b.py"),
        ]
        _common_model_args(command, model, default_batch_size=128)
        command.extend(
            [
                "--which",
                which,
                "--core-prime-mode",
                str(experiment.get("core_prime_mode", "lexically_controlled")),
                "--seed",
                str(int(experiment.get("seed", 13))),
                "--output-root",
                str(output_dir),
                "--prime-conditions",
                *[str(value) for value in prime_conditions],
            ]
        )
        max_items = experiment.get("max_items")
        if max_items is not None:
            command.extend(["--max-items", str(int(max_items))])
        plans.append(CommandPlan(f"exp1b:{model['model_condition']}", command, output_dir))
    return base, plans


def _exp2_prompt_paths(prompt_dir: Path, mode: str) -> Dict[str, Path]:
    return {
        "core": prompt_dir / f"experiment_2_core_demo_prompts_{mode}.csv",
        "jabberwocky": prompt_dir / f"experiment_2_jabberwocky_demo_prompts_{mode}.csv",
        "core_targets_jabberwocky_primes": (
            prompt_dir / f"experiment_2_core_targets_jabberwocky_primes_demo_prompts_{mode}.csv"
        ),
    }


def _build_exp2_plan(
    experiment: Mapping[str, object],
    models: Sequence[Dict[str, object]],
    output_dir_override: Path | None,
) -> tuple[Path, List[CommandPlan]]:
    base = _base_output_dir(
        experiment,
        "behavioral_results/experiment-2/default_run",
        output_dir_override,
    )
    prompt_dir = (
        base / "_prompts"
        if output_dir_override is not None
        else _resolved_path(
            experiment.get(
                "prompt_output_dir", "behavioral_results/generated_materials/experiment-2/prompts"
            )
        )
    )
    conditions_raw = experiment.get(
        "conditions", ["core", "jabberwocky", "core_targets_jabberwocky_primes"]
    )
    if not isinstance(conditions_raw, list) or not conditions_raw:
        raise ValueError("experiment.conditions must be a non-empty list.")
    conditions = [str(value) for value in conditions_raw]
    valid = {"core", "jabberwocky", "core_targets_jabberwocky_primes"}
    invalid = set(conditions) - valid
    if invalid:
        raise ValueError(f"Invalid Experiment 2 conditions: {sorted(invalid)}")

    mode = str(experiment.get("core_prime_mode", "lexically_controlled"))
    max_items = int(experiment.get("max_items", 2048))
    prompt_columns = experiment.get(
        "prompt_columns", ["prompt_active", "prompt_passive", "prompt_no_prime", "prompt_filler"]
    )
    if not isinstance(prompt_columns, list) or not prompt_columns:
        raise ValueError("experiment.prompt_columns must be a non-empty list.")

    export_command = [
        sys.executable,
        str(REPO_ROOT / "scripts/materials/28_export_demo_prompt_csvs.py"),
        "--core-prime-mode",
        mode,
        "--max-items",
        str(max_items),
        "--seed",
        str(int(experiment.get("seed", 13))),
        "--event-style",
        str(experiment.get("event_style", "involving_event")),
        "--role-style",
        str(experiment.get("role_style", "did_to")),
        "--quote-style",
        str(experiment.get("quote_style", "mary_answered")),
        "--role-order",
        str(experiment.get("role_order", "counterbalanced")),
        "--target-verb-cue",
        str(experiment.get("target_verb_cue", "auto_real_targets")),
        "--output-dir",
        str(prompt_dir),
    ]
    plans = [CommandPlan("exp2:export-prompts", export_command, prompt_dir)]
    prompt_paths = _exp2_prompt_paths(prompt_dir, mode)

    for model in models:
        model_root = _model_output_dir(base, model, len(models))
        for condition in conditions:
            output_dir = model_root / condition
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts/experiments/29_demo_prompt_generation_audit.py"),
            ]
            _common_model_args(command, model, default_batch_size=2)
            command.extend(
                [
                    "--prompt-csv",
                    str(prompt_paths[condition]),
                    "--output-dir",
                    str(output_dir),
                    "--max-items",
                    str(max_items),
                    "--max-new-tokens",
                    str(int(experiment.get("max_new_tokens", 24))),
                    "--seed",
                    str(int(experiment.get("seed", 13))),
                    "--prompt-columns",
                    *[str(value) for value in prompt_columns],
                ]
            )
            plans.append(
                CommandPlan(
                    f"exp2:{model['model_condition']}:{condition}",
                    command,
                    output_dir,
                )
            )
    return base, plans


def build_legacy_plan(
    *,
    experiment_id: str,
    config: Mapping[str, object],
    output_dir_override: Path | None = None,
    model_filters: Sequence[str] | None = None,
) -> tuple[Path, List[CommandPlan]]:
    experiment, models = validate_common_config(config, experiment_id)
    selected_models = select_model_entries(models, model_filters)
    if experiment_id == "exp1a":
        return _build_exp1a_plan(experiment, selected_models, output_dir_override)
    if experiment_id == "exp1b":
        return _build_exp1b_plan(experiment, selected_models, output_dir_override)
    if experiment_id == "exp2":
        return _build_exp2_plan(experiment, selected_models, output_dir_override)
    raise ValueError(f"No legacy dispatcher for '{experiment_id}'.")


def print_plan(experiment_id: str, plans: Sequence[CommandPlan]) -> None:
    print(f"Validated configuration for {experiment_id}. Planned commands:")
    for plan in plans:
        print(f"[{plan.label}]")
        print(shlex.join(plan.command))


def run_legacy_experiment(
    *,
    experiment_id: str,
    config_path: Path,
    output_dir_override: Path | None = None,
    model_filters: Sequence[str] | None = None,
    dry_run: bool = False,
) -> None:
    config = load_config(config_path)
    base_output, plans = build_legacy_plan(
        experiment_id=experiment_id,
        config=config,
        output_dir_override=output_dir_override,
        model_filters=model_filters,
    )
    if dry_run:
        print_plan(experiment_id, plans)
        return

    base_output.mkdir(parents=True, exist_ok=True)
    (base_output / "resolved_config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    for plan in plans:
        print(f"\n[{plan.label}]")
        print(shlex.join(plan.command))
        subprocess.run(plan.command, cwd=REPO_ROOT, check=True)

    manifest = {
        "experiment": experiment_id,
        "config_path": str(config_path.expanduser().resolve()),
        "commands": [
            {
                "label": plan.label,
                "command": plan.command,
                "output_dir": str(plan.output_dir) if plan.output_dir is not None else None,
            }
            for plan in plans
        ],
    }
    (base_output / "config_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
