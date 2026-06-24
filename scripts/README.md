# Script organization

The command-line tools are grouped by function. The numbered filenames are
retained for provenance and to keep earlier run records interpretable.

- `experiments/`: model execution, generation, and scoring passes;
- `materials/`: corpus, vocabulary-resource, and prompt construction;
- `audits/`: material validation, generation annotation, and review tools;
- `analysis/`: statistical preparation, summaries, and R Markdown analyses.

Shared Python functionality belongs in `src/`. The experiment configurations
should normally be run through the repository-level `run_experiment.py`; direct
script commands remain available for lower-level or specialized workflows.

## Current audit workflow

- `audits/1_audit_jabberwocky_lexicon.py` checks exact, stem, and near-form
  overlap with English and the bundled controlled vocabulary.
- `audits/2_audit_jabberwocky_semantics.py` compares the 40 canonical nonce
  nouns with real-noun representations and reports a leave-one-out real-noun
  baseline for calibration.
- `audits/31_annotate_generation_structure_labels.py` assigns automatic
  Experiment 2 structure and role-frame labels.
- `audits/32_summarize_reviewed_generation_audit.py` overlays manual labels and
  produces reviewed summaries.
- `audits/47_exp2_master_review.py` builds and applies a deduplicated review
  file across Experiment 2 model runs.

Generated vocabulary-validation outputs are written under
`corpora/transitive/validation/` and remain local. Compact canonical corpus
audits remain tracked under `corpora/transitive/metadata/`.
