PYTHON ?= .venv/bin/python

.PHONY: python-version test experiment-1a experiment-1b experiment-2 experiment-3 experiment-4 config-check transitive-priming transitive-report transitive-stats jabberwocky-lexicon-audit jabberwocky-semantic-audit regenerate-jabberwocky-transitive export-experiment-2-prompts export-experiment-4-materials generated-materials

python-version:
	$(PYTHON) --version

test:
	$(PYTHON) -m unittest discover -s tests -v

experiment-1a:
	$(PYTHON) run_experiment.py --experiment exp1a --config configs/experiment1a_default.yaml

experiment-1b:
	$(PYTHON) run_experiment.py --experiment exp1b --config configs/experiment1b_default.yaml

experiment-2:
	$(PYTHON) run_experiment.py --experiment exp2 --config configs/experiment2_default.yaml

experiment-3:
	$(PYTHON) run_experiment.py --experiment exp3 --config configs/experiment3_default.yaml

experiment-4:
	$(PYTHON) run_experiment.py --experiment exp4 --config configs/experiment4_default.yaml

config-check:
	$(PYTHON) run_experiment.py --experiment exp1a --config configs/experiment1a_default.yaml --dry-run
	$(PYTHON) run_experiment.py --experiment exp1b --config configs/experiment1b_default.yaml --dry-run
	$(PYTHON) run_experiment.py --experiment exp2 --config configs/experiment2_default.yaml --dry-run
	$(PYTHON) run_experiment.py --experiment exp3 --config configs/experiment3_default.yaml --dry-run
	$(PYTHON) run_experiment.py --experiment exp4 --config configs/experiment4_default.yaml --dry-run

transitive-priming:
	$(PYTHON) scripts/experiments/2_transitive_token_priming.py --preset paper_main

transitive-report:
	$(PYTHON) scripts/analysis/exp1a_exploratory_summarize_prediction_error.py

transitive-stats:
	$(PYTHON) scripts/analysis/exp1a_exploratory_prediction_error_stats.py

jabberwocky-lexicon-audit:
	$(PYTHON) scripts/audits/1_audit_jabberwocky_lexicon.py

jabberwocky-semantic-audit:
	$(PYTHON) scripts/audits/2_audit_jabberwocky_semantics.py

regenerate-jabberwocky-transitive:
	$(PYTHON) scripts/materials/42_build_gpt2_monosyllabic_jabberwocky_strict_4cell.py

export-experiment-2-prompts:
	$(PYTHON) scripts/materials/28_export_demo_prompt_csvs.py

export-experiment-4-materials:
	$(PYTHON) scripts/materials/43_build_exp4_complex_np_role_recovery.py
	$(PYTHON) scripts/materials/44_export_exp4_complex_np_prompts.py

generated-materials: export-experiment-2-prompts export-experiment-4-materials
