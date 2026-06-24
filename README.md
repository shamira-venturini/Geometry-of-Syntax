# Geometry of Syntax

Research code for studying structural representations in language models through
syntactic priming. The project compares real-word and Jabberwocky materials to
investigate how active and passive structure is represented across processing,
generation, continuation-scoring, and role-recovery tasks.

## Research questions

- Can structural priming serve as a behavioral probe of latent syntactic
  representations in language models?
- Which priming effects remain when lexical-semantic information is reduced
  using Jabberwocky materials?
- How do processing-style and production-style measurements compare?
- How strongly do baseline structural preferences influence apparent priming
  effects?

## Experiment overview

- **Experiment 1a:** Processing-based structural priming replication and
  Jabberwocky extension.
- **Experiment 1b:** Controlled active/passive target competition with explicit
  baseline conditions.
- **Experiment 2:** Full-sentence generation and structural annotation.
- **Experiment 3:** Teacher-forced active/passive continuation scoring.
- **Experiment 4:** Sentence-to-event role recovery.

## Repository structure

```text
configs/        Experiment configuration files
corpora/        Controlled materials, metadata, and lexical resources
colab_*.ipynb   Reproducible Colab workflow
scripts/        Grouped experiment, material, audit, analysis, and reporting tools
src/            Shared experiment and scoring modules
tests/          Configuration and dispatch regression tests
```

Generated materials and run outputs are written under `behavioral_results/` and
are not source-controlled. Retired designs remain in a dated local archive and
are not part of the reproducibility package.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the experiments

Every experiment uses the same entry point and YAML configuration pattern:

```bash
python run_experiment.py --experiment exp1a --config configs/experiment1a_default.yaml
python run_experiment.py --experiment exp1b --config configs/experiment1b_default.yaml
python run_experiment.py --experiment exp2 --config configs/experiment2_default.yaml
python run_experiment.py --experiment exp3 --config configs/experiment3_default.yaml
python run_experiment.py --experiment exp4 --config configs/experiment4_default.yaml
```

Validate a config and inspect its execution plan without loading a model:

```bash
python run_experiment.py --experiment exp2 --config configs/experiment2_default.yaml --dry-run
```

Regenerate the expanded Experiment 2 and complex-NP Experiment 4 materials:

```bash
make generated-materials
```

All configs use the shared top-level keys `schema_version`, `experiment`, and
`models`. Task-specific settings remain inside `experiment`. Lower-level
experiment components are documented in `scripts/README.md`.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the canonical materials,
generated-artifact policy, provenance records, and complete run interface.

## Reproducibility

The experiment configurations record model and task settings, while run outputs
include resolved configurations and software metadata where supported. Model
weights are not stored in this repository.
