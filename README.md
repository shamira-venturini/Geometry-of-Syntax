# Experiment 3 Controlled Disambiguation Pipeline

This repository includes **Experiment 3**, an additional (non-substitutive) experiment that runs deterministic, teacher-forced disambiguation scoring for two matched subtasks:

1. **Experiment 3 production-like task** (active vs passive completion scoring)
2. **Experiment 3 comprehension-like task** (role-commitment answer scoring)

The pipeline is configured to run on the **same corpora family used elsewhere in this project**, specifically:

- `corpora/transitive/CORE_transitive_constrained_counterbalanced_lexically_controlled.csv`
- `corpora/transitive/CORE_transitive_constrained_counterbalanced.csv`
- `corpora/transitive/jabberwocky_transitive_bpe_filtered_2080.csv`

No toy dataset is used.

## Project Structure

- `src/data.py` - corpus-driven dataset construction + optional table loader
- `src/prompts.py` - prompt rendering for both tasks
- `src/models.py` - model wrapper with optional chat-template mode
- `src/scoring.py` - full-candidate batched logprob scoring
- `src/analysis.py` - baseline/priming contrasts, bootstrap CIs, paired tests
- `src/plots.py` - PNG plots by prime/task/lexicality
- `run_experiment.py` - CLI entry point
- `configs/` - ready-to-run YAML configs

## Installation

```bash
pip install -r requirements.txt
```

## Core Design

The pipeline keeps **baseline preference** and **priming effect** separate.

Prime conditions supported:

- `active`
- `passive`
- `filler`
- `no_prime`

Lexicality conditions supported:

- `real` (lexically controlled + lexical overlap corpora)
- `nonce` (Jabberwocky corpus)

Filler behavior:

- `filler` uses domain-matched **intransitive filler pools** by default (`real` vs `nonce`).
- Legacy offset fillers can still be requested with `experiment.filler_mode: offset_target`.

For each corpus row (`pa`, `pp`, `ta`, `tp`):

- Experiment 3 production-like scores active vs passive full completions.
- Experiment 3 comprehension-like scores correct vs incorrect role-answer candidates against active/passive target sentences.

All scores are teacher-forced and deterministic.

## Scoring Outputs

For each candidate, the pipeline stores:

- total logprob
- mean logprob per token
- candidate token count
- token IDs/tokens/token logprobs (debug)

Primary preferences:

- production-like: `active_minus_passive_logprob_total` and `_mean`
- comprehension-like: `answer_preference_logprob_total` and `_mean`

Additional location/token-wise preference metrics are exported, including first/second/last-token, divergence-token, aligned-position means, and token-position summaries.

## Running

### Combined (GPT-2-large + Llama-3.2-3B-Instruct plain)

```bash
python run_experiment.py --config configs/experiment3_default.yaml
```

### GPT-2 only

```bash
python run_experiment.py --config configs/experiment3_gpt2.yaml
```

### Llama plain

```bash
python run_experiment.py --config configs/experiment3_llama_plain.yaml
```

### Llama chat-template

```bash
python run_experiment.py --config configs/experiment3_llama_chat.yaml
```

### Colab profile

```bash
python run_experiment.py --config configs/experiment3_colab.yaml
```

### Override output directory

```bash
python run_experiment.py --config configs/experiment3_default.yaml --output-dir behavioral_results/experiment-3/custom_run
```

## Required Outputs

Each run writes:

- `item_level_results.csv`
- `summary_by_model_condition.csv`
- `summary_by_prime_condition.csv`
- `summary_by_lexicality.csv`
- `summary_by_task.csv`
- `priming_effects_relative_to_baseline.csv`
- `paired_condition_tests.csv`
- `bootstrap_by_prime_condition.csv`
- `summary_by_measure_prime_condition.csv`
- `priming_effects_all_measures.csv`
- `paired_condition_tests_all_measures.csv`
- `token_position_summary.csv`
- `token_position_preference_summary.csv`
- optional plots:
  - `preference_by_prime_condition.png`
  - `preference_by_task.png`
  - `preference_by_lexicality.png`
  - `interaction_prime_by_task.png`
- reproducibility artifacts:
  - `run.log`
  - `run_metadata.json`
  - `resolved_config.yaml`

## Caveats

1. Total logprob can prefer shorter candidates; always inspect mean-per-token variants.
2. Instruction/chat formatting can change scores materially; compare `plain` vs `chat_template` explicitly.
3. Legacy labels (`active_prime`, `passive_prime`, `filler_prime`, `no_prime_eos`, `no_prime_empty`, `no_demo`) are normalized to canonical labels at load/analysis time.
4. Tokenization boundary artifacts are logged in item-level debug fields.
