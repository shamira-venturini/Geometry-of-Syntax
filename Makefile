PYTHON := .venv/bin/python

.PHONY: python-version transitive-priming transitive-report transitive-stats core-generation-pilot jabberwocky-lexicon-audit jabberwocky-semantic-audit jabberwocky-tokenizer-filter regenerate-jabberwocky-transitive regenerate-jabberwocky-transitive-bpe

python-version:
	$(PYTHON) --version

transitive-priming:
	$(PYTHON) scripts/Phase-1/2_transitive_token_priming.py --preset paper_main

transitive-report:
	$(PYTHON) scripts/Phase-1/3_summarize_transitive_priming.py

transitive-stats:
	$(PYTHON) scripts/Phase-1/5_analyze_transitive_statistics.py

core-generation-pilot:
	$(PYTHON) scripts/Phase-1/6_core_generation_priming_pilot.py

jabberwocky-lexicon-audit:
	$(PYTHON) scripts/Phase-1/1_audit_jabberwocky_lexicon.py

jabberwocky-semantic-audit:
	$(PYTHON) scripts/Phase-1/2_audit_jabberwocky_semantics.py

jabberwocky-tokenizer-filter:
	$(PYTHON) scripts/Phase-1/4_filter_jabberwocky_tokenizer_lengths.py

regenerate-jabberwocky-transitive:
	$(PYTHON) scripts/corpora_generators/0_generate_jabberwocky_transitive.py

regenerate-jabberwocky-transitive-bpe:
	$(PYTHON) scripts/corpora_generators/0_generate_jabberwocky_transitive.py \
		--vocab-path corpora/transitive/jabberwocky_transitive_strict_bpe_filtered_vocabulary.json \
		--output-path corpora/transitive/jabberwocky_transitive_bpe_filtered.csv
