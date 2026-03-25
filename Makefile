PYTHON := .venv/bin/python

.PHONY: python-version transitive-priming transitive-report transitive-stats core-completion-choice-pilot jabberwocky-lexicon-audit jabberwocky-semantic-audit jabberwocky-tokenizer-filter regenerate-jabberwocky-transitive regenerate-jabberwocky-transitive-bpe

python-version:
	$(PYTHON) --version

transitive-priming:
	$(PYTHON) scripts/2_transitive_token_priming.py --preset paper_main

transitive-report:
	$(PYTHON) scripts/3_summarize_transitive_priming.py

transitive-stats:
	$(PYTHON) scripts/5_analyze_transitive_statistics.py

core-completion-choice-pilot:
	$(PYTHON) scripts/7_core_completion_choice_pilot.py

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
