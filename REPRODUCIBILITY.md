# Reproducibility Guide

This repository contains the code, controlled materials, configuration files,
and machine-readable audit metadata needed to reproduce the structural-priming
experiments. Model weights and generated run outputs are intentionally excluded.

## Environment

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model weights are loaded from their configured Hugging Face identifiers. Access
to gated models must be arranged separately by the user.

## Unified experiment interface

All experiments use the same command structure:

```bash
python run_experiment.py --experiment EXPERIMENT --config CONFIG
```

Default configurations:

```text
exp1a  configs/experiment1a_default.yaml
exp1b  configs/experiment1b_default.yaml
exp2   configs/experiment2_default.yaml
exp3   configs/experiment3_default.yaml
exp4   configs/experiment4_default.yaml
```

Validate a configuration without loading a model or writing outputs:

```bash
python run_experiment.py \
  --experiment exp2 \
  --config configs/experiment2_default.yaml \
  --dry-run
```

Each completed config-driven run records its resolved configuration and run
metadata in the selected output directory.

## Canonical transitive materials

Each four-column corpus stores matched sentence quartets:

- `pa`: active prime
- `pp`: passive prime
- `ta`: active target
- `tp`: passive target

The canonical corpora are:

- `materials/corpora/CORE_transitive_strict_4cell_counterbalanced.csv`:
  controlled real-word materials;
- `materials/corpora/jabberwocky_transitive_monosyllabic_strict_4cell-counterbalanced.csv`:
  Jabberwocky materials built on the same row structure;
- `materials/corpora/experiment_2_CORE_transitive_core-targets_jabberwocky-primes_2048.csv`:
  Jabberwocky primes paired with real-word targets.

The corresponding files under `materials/metadata/` record balance,
lexical-overlap, tokenization, and construction audits. Paths in tracked
metadata are relative to the repository root.

## Controlled vocabulary resources

`materials/vocabulary_lists/` contains the noun, verb, adjective, and
frozen Jabberwocky lexicon required by the corpus builders. Historical nonce
candidate pools are archived locally rather than published as active inputs.
The same folder contains the USF association edge table, which records directed
cue-target pairs used to exclude semantic associations during material
construction and validation.

Model-specific vocabulary audits are reproducible local outputs under
`materials/validation/`. They are distinct from the tracked corpus
metadata, which records prime-target lexical overlap and semantic-association
checks for the canonical materials.

## Generated working artifacts

The following are reproducible outputs rather than source materials and are not
tracked:

- behavioral results, logs, checkpoints, and ZIP archives;
- rendered R Markdown figures and reports.

The exact expanded prompt/material files used by Experiments 2 and 4 are bundled
as tracked materials:

```text
materials/prompts/experiment-2/prompts/
materials/prompts/experiment-4/complex_np/
```

They are kept outside `materials/corpora/` so canonical base corpora remain
separate from expanded prompt tables.

## Local design archive

Retired corpora and intermediate designs are retained locally under `_archive/`
and excluded from the public repository. They are not inputs to the default
experiment configurations. The dated archive manifest records why each artifact
was retired and which canonical file superseded it.

## Determinism and provenance

- Config files record seeds, model identifiers, model conditions, corpus paths,
  batch sizes, prompt settings, and output locations.
- Corpus summary files record construction constraints and audit counts.
- Experiment 2 generation is greedy; Experiments 1 and 3 use deterministic
  scoring rather than sampled generation.
- Hardware and numerical precision can affect model execution and are recorded
  in run metadata where supported.
