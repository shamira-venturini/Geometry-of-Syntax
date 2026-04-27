# Experiment 3 Demo-Prompt Continuation Pipeline

This repository includes **Experiment 3**, a deterministic, teacher-forced continuation-scoring pipeline that uses an Experiment 2-style demo prompt and scores **active vs passive full-sentence continuations**.

The pipeline is configured to run on the **same corpora family used elsewhere in this project**, specifically:

- `corpora/transitive/CORE_transitive_constrained_counterbalanced.csv`
- `corpora/transitive/CORE_transitive_strict_4cell_counterbalanced.csv`
- `corpora/transitive/jabberwocky_transitive_matched_strict_4cell.csv`

No toy dataset is used.

## Project Structure

- `src/data.py` - corpus-driven dataset construction + optional table loader
- `src/prompts.py` - demo-prime + target continuation prompt rendering
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

For each corpus row (`pa`, `pp`, `ta`, `tp`), Experiment 3 builds a demo prompt in the Experiment 2 discourse frame:

- prime as solved example (`active`, `passive`, `filler`, or omitted for `no_prime`)
- target event scaffold ending at `Mary answered, "`
- continuation candidates: active target sentence vs passive target sentence

All scores are teacher-forced and deterministic.

## Scoring Outputs

For each candidate, the pipeline stores:

- total logprob
- mean logprob per token
- candidate token count
- token IDs/tokens/token logprobs (debug)

Primary preferences:

- continuation preference: `active_minus_passive_logprob_total` and `_mean`
- 1B-compatible deltas: `passive_minus_active_logprob_sum` and `passive_minus_active_logprob`

1B-compatible item-level fields are also exported:

- `active_choice_logprob`, `passive_choice_logprob`
- `active_choice_logprob_sum`, `passive_choice_logprob_sum`
- `active_target_token_count`, `passive_target_token_count`
- `chosen_structure`, `passive_choice_indicator`

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

## Experiment 4 (Additional)

Experiment 4 is a Ferreira-inspired free-answer comprehension probe aligned to Experiment 1B materials.  
Use:

```bash
python run_experiment.py --experiment exp4 --config configs/exp4.yaml
```

Toy run:

```bash
python run_experiment.py --experiment exp4 --config configs/exp4_toy.yaml
```

See [docs/experiment4.md](docs/experiment4.md) for prompt format, outputs, and diagnostics.
