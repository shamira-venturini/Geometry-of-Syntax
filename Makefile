ifeq ($(wildcard .venv312-mps/bin/python),.venv312-mps/bin/python)
PYTHON ?= .venv312-mps/bin/python
else
PYTHON ?= .venv/bin/python
endif

.PHONY: python-version transitive-priming transitive-report transitive-stats processing-experiment-1b processing-experiment-1b-suite processing-experiment-1b-report core-completion-choice-pilot counterbalanced-completion-choice counterbalanced-generation-choice counterbalanced-production-suite emnlp-story-figures jabberwocky-lexicon-audit jabberwocky-semantic-audit jabberwocky-tokenizer-filter regenerate-jabberwocky-transitive regenerate-jabberwocky-transitive-bpe

python-version:
	$(PYTHON) --version

transitive-priming:
	$(PYTHON) scripts/2_transitive_token_priming.py --preset paper_main

transitive-report:
	$(PYTHON) scripts/3_summarize_transitive_priming.py

transitive-stats:
	$(PYTHON) scripts/5_analyze_transitive_statistics.py

processing-experiment-1b:
	$(PYTHON) scripts/15_counterbalanced_processing_experiment_1b.py

processing-experiment-1b-suite:
	$(PYTHON) scripts/16_run_processing_experiment_1b.py

processing-experiment-1b-report:
	$(PYTHON) scripts/18_report_processing_experiment_1b.py

core-completion-choice-pilot:
	$(PYTHON) scripts/7_core_completion_choice_pilot.py

counterbalanced-completion-choice:
	$(PYTHON) scripts/12_counterbalanced_completion_choice_experiment.py

counterbalanced-generation-choice:
	$(PYTHON) scripts/13_counterbalanced_generation_experiment.py

counterbalanced-production-suite:
	$(PYTHON) scripts/14_run_counterbalanced_production_experiments.py

emnlp-story-figures:
	mkdir -p .cache/matplotlib
	MPLCONFIGDIR=$(CURDIR)/.cache/matplotlib $(PYTHON) scripts/17_make_emnlp_story_figures.py

jabberwocky-lexicon-audit:
	$(PYTHON) scripts/1_audit_jabberwocky_lexicon.py

jabberwocky-semantic-audit:
	$(PYTHON) scripts/2_audit_jabberwocky_semantics.py

jabberwocky-tokenizer-filter:
	$(PYTHON) scripts/4_filter_jabberwocky_tokenizer_lengths.py

regenerate-jabberwocky-transitive:
	$(PYTHON) scripts/0_generate_jabberwocky_transitive.py

regenerate-jabberwocky-transitive-bpe:
	$(PYTHON) scripts/0_generate_jabberwocky_transitive.py \
		--vocab-path corpora/transitive/jabberwocky_transitive_strict_bpe_filtered_vocabulary.json \
		--output-path corpora/transitive/jabberwocky_transitive_bpe_filtered.csv
