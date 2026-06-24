# Geometry of Syntax

Research code for studying structural representations in language models through
syntactic priming. The project compares real-word and Jabberwocky materials to
investigate how active and passive structure is represented across processing,
generation, continuation-scoring, and role-recovery tasks.

> **Project status:** Active research. The repository is currently being
> reorganized for reproducibility and public release. Results and documentation
> may change while analyses are finalized.

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
corpora/        Controlled experimental materials
docs/           Methods and design documentation
notebooks/      Colab and exploratory notebooks
scripts/        Corpus construction, experiment, and analysis scripts
src/            Shared experiment and scoring modules
```

Large raw outputs, model artifacts, temporary files, and working archives are
being moved out of the public repository. A curated set of summary results and
figures will be added when the analysis structure is finalized.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the current pipelines

Experiment 3:

```bash
python run_experiment.py --experiment exp3 --config configs/experiment3_default.yaml
```

Experiment 4:

```bash
python run_experiment.py --experiment exp4 --config configs/exp4.yaml
```

Additional experiment-specific runners are available in `scripts/`. See
`docs/experiment_scoring_methods.md` for the current scoring definitions and
`docs/corpus_control_counterbalancing.md` for corpus controls.

## Reproducibility note

The experiment configurations record model and task settings, while run outputs
include resolved configurations and software metadata where supported. Model
weights are not stored in this repository.

## Citation and license

Citation information and a project license will be added before the first
archival release.
