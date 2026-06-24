# Script organization

The command-line tools are grouped by function. The numbered filenames are
retained for provenance and to keep earlier run records interpretable.

- `experiments/`: model execution, generation, and scoring passes;
- `materials/`: corpus, vocabulary-resource, and prompt construction;
- `audits/`: material validation, generation annotation, and review tools;
- `analysis/`: statistical preparation, summaries, and R Markdown analyses;
- `reporting/`: figures, reports, and presentation builders.

Shared Python functionality belongs in `src/`. The experiment configurations
should normally be run through the repository-level `run_experiment.py`; direct
script commands remain available for lower-level or specialized workflows.
