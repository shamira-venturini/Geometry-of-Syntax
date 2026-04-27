ifeq ($(wildcard .venv312-mps/bin/python),.venv312-mps/bin/python)
PYTHON ?= .venv312-mps/bin/python
else
PYTHON ?= .venv/bin/python
endif

.PHONY: python-version transitive-priming transitive-report transitive-stats colab-experiment-1a processing-experiment-1b processing-experiment-1b-suite processing-experiment-1b-report core-completion-choice-pilot counterbalanced-completion-choice counterbalanced-generation-choice counterbalanced-production-suite emnlp-story-figures jabberwocky-lexicon-audit jabberwocky-semantic-audit jabberwocky-tokenizer-filter regenerate-jabberwocky-transitive regenerate-jabberwocky-transitive-bpe export-experiment-2-prompts

python-version:
	$(PYTHON) --version

transitive-priming:
	$(PYTHON) scripts/2_transitive_token_priming.py --preset paper_main

transitive-report:
	$(PYTHON) scripts/3_summarize_transitive_priming.py

transitive-stats:
	$(PYTHON) scripts/5_analyze_transitive_statistics.py

colab-experiment-1a:
	$(PYTHON) scripts/10_run_colab_experiment_1a.py

processing-experiment-1b:
	$(PYTHON) scripts/15_counterbalanced_processing_experiment_1b.py

processing-experiment-1b-suite:
	$(PYTHON) scripts/16_run_processing_experiment_1b.py

processing-experiment-1b-report:
	$(PYTHON) scripts/18_report_processing_experiment_1b.py

core-completion-choice-pilot:
	@echo "Retired: completion-choice pilot path. Use 'make colab-experiment-1a' for Experiment 1a."

counterbalanced-completion-choice:
	@echo "Retired: Experiment 2a completion-choice."

counterbalanced-generation-choice:
	@echo "Retired: Experiment 2b generation-choice."

counterbalanced-production-suite:
	@echo "Retired: 2a/2b production suite wrapper."

emnlp-story-figures:
	mkdir -p .cache/matplotlib
	MPLCONFIGDIR=$(CURDIR)/.cache/matplotlib $(PYTHON) scripts/17_make_emnlp_story_figures.py

jabberwocky-lexicon-audit:
	$(PYTHON) scripts/1_audit_jabberwocky_lexicon.py

jabberwocky-semantic-audit:
	$(PYTHON) scripts/2_audit_jabberwocky_semantics.py

jabberwocky-tokenizer-filter:
	@echo "Retired: old BPE-filtered Jabberwocky vocabulary path. Use 'make regenerate-jabberwocky-transitive'."

regenerate-jabberwocky-transitive:
	$(PYTHON) scripts/42_build_gpt2_monosyllabic_jabberwocky_strict_4cell.py

regenerate-jabberwocky-transitive-bpe:
	@echo "Retired: old BPE-filtered Jabberwocky corpus path. Use 'make regenerate-jabberwocky-transitive'."

export-experiment-2-prompts:
	$(PYTHON) scripts/28_export_demo_prompt_csvs.py
