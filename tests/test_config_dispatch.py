from __future__ import annotations

import unittest
from pathlib import Path

from src.config_dispatch import (
    build_legacy_plan,
    load_config,
    validate_common_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigDispatchTests(unittest.TestCase):
    def test_every_default_config_uses_shared_schema(self) -> None:
        configs = {
            "exp1a": "experiment1a_default.yaml",
            "exp1b": "experiment1b_default.yaml",
            "exp2": "experiment2_default.yaml",
            "exp3": "experiment3_default.yaml",
            "exp4": "experiment4_default.yaml",
        }
        for experiment_id, filename in configs.items():
            with self.subTest(experiment=experiment_id):
                config = load_config(REPO_ROOT / "configs" / filename)
                experiment, models = validate_common_config(config, experiment_id)
                self.assertEqual(experiment["id"], experiment_id)
                self.assertGreaterEqual(len(models), 1)

    def test_experiment_identity_mismatch_is_rejected(self) -> None:
        config = load_config(REPO_ROOT / "configs/experiment1a_default.yaml")
        with self.assertRaisesRegex(ValueError, "experiment.id"):
            validate_common_config(config, "exp2")

    def test_exp1a_plan_preserves_runner_interface(self) -> None:
        config = load_config(REPO_ROOT / "configs/experiment1a_default.yaml")
        _, plans = build_legacy_plan(experiment_id="exp1a", config=config)
        self.assertEqual(len(plans), 1)
        command = plans[0].command
        self.assertTrue(
            command[1].endswith("scripts/experiments/10_run_colab_experiment_1a.py")
        )
        self.assertIn("--preset", command)
        self.assertIn("paper_main", command)
        self.assertIn("--output-root", command)

    def test_exp1b_plan_selects_both_controlled_conditions(self) -> None:
        config = load_config(REPO_ROOT / "configs/experiment1b_default.yaml")
        _, plans = build_legacy_plan(experiment_id="exp1b", config=config)
        command = plans[0].command
        which_index = command.index("--which")
        self.assertEqual(command[which_index + 1], "both")
        self.assertIn("--prime-conditions", command)

    def test_exp2_plan_uses_bundled_prompts(self) -> None:
        config = load_config(REPO_ROOT / "configs/experiment2_default.yaml")
        _, plans = build_legacy_plan(experiment_id="exp2", config=config)
        self.assertEqual(
            [plan.label.rsplit(":", 1)[-1] for plan in plans],
            ["core", "jabberwocky", "core_targets_jabberwocky_primes"],
        )
        for plan in plans:
            self.assertIn("--prompt-csv", plan.command)
            self.assertIn("--prompt-columns", plan.command)


if __name__ == "__main__":
    unittest.main()
