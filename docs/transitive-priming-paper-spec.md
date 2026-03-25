# Transitive Priming Paper Spec

## Goal

Produce one simple behavioral paper on structural priming in language models.
The paper should be clean, narrow, and defensible.

## Scope

- alternation: transitive active/passive only
- conditions: `CORE`, `ANOMALOUS`, `jabberwocky`
- primary model: `gpt2-large`
- optional replication: 1-2 additional base models if easy

Do not include:

- PCA
- steering
- the current dative materials
- clinical framing
- semantic-similarity conditions in the first pass

## Chain of experiments

### Experiment 1: fixed-target structural priming

Question:

Do language models show structural priming in transitive active/passive targets
when lexical and semantic support are progressively depleted?

Measures:

- sentence-level priming effect (`sentence_pe`)
- token-level priming effect by target position (`token_pe`)
- structure-region priming effect (`structure_region_pe`)
- literal post-divergence priming effect (`post_divergence_pe`)

Primary result:

- `structure_region_pe`

Supporting result:

- token-level traces by position
- `post_divergence_pe` as a secondary descriptive measure

Conditions:

- `CORE`
- `ANOMALOUS`
- `jabberwocky`

Interpretation:

- `CORE` provides the ordinary lexical baseline
- `ANOMALOUS` keeps real words while degrading semantic coherence
- `jabberwocky` removes ordinary lexical semantics from the content words while
  preserving English morphosyntax

Current Jabberwocky corpus status:

- active file: `corpora/transitive/jabberwocky_transitive.csv`
- active vocabulary: `corpora/transitive/jabberwocky_transitive_strict_vocabulary.json`
- archived legacy corpus retained separately
- current strict-lexicon audit reports zero exact English overlaps, zero stem
  overlaps, and zero neighbors within edit distance 2 under the current checks

Target claim:

- structural priming weakens under semantic depletion, but remains detectable in
  fully nonce materials
- the residual effect is concentrated in the structure-diagnostic region of the
  target

Current status:

- this is the only active experiment for the first pass
- all implementation should serve this experiment first

### Experiment 2: priming dynamics

Question:

Does the transitive priming effect behave like a priming phenomenon rather than
like a static structure preference?

Recommended option:

- recency manipulation

Alternative:

- cumulativity manipulation

Measures:

- change in `post_divergence_pe` across distances or prime counts

Reason:

One dynamics experiment makes the paper feel complete without making it large.

Data source:

- use `PrimeLM/corpora/RECENCY_5_transitive_15000sampled_10-1.csv`
- optional follow-up: `PrimeLM/corpora/CUMULATIVE_5_transitive_15000sampled_10-1.csv`

Current status:

- postponed
- only revisit after Experiment 1 is clean and interpretable

### Experiment 3: generation-based validation

Question:

Does the same bias appear in free continuation behavior, not only in rescored
fixed targets?

Design:

- prime sentence
- ambiguous transitive prompt stem
- compare active/passive continuation bias against an unprimed baseline

Measure:

- structure choice shift

This experiment is optional.
If implementation becomes costly, the paper can survive without it.

Current status:

- postponed

## Minimal figure set

1. Mean sentence-level and post-divergence priming by condition and structure
2. Token-level priming traces by condition

First-pass figure set:

1. is sufficient for the first analysis pass
2. is required before deciding whether the paper is viable

## Minimal table set

1. Dataset counts and condition overview
2. Main effect sizes for `sentence_pe`, `structure_region_pe`, and
   `post_divergence_pe`
3. Optional replication results across models

## Required implementation in this repo

The first-pass implementation should produce:

- item-level transitive priming scores
- token-level transitive priming scores
- region summaries
- markdown or CSV outputs that can be used for the paper figures

The new scripts in `scripts/` are intended to support exactly that.

Current presets in the scoring script:

- `paper_main`: local `CORE` + `ANOMALOUS` + `jabberwocky`
- `primelm_core`: original Prime-LM transitive core conditions
- `primelm_recency`: original Prime-LM recency corpus
- `primelm_cumulative`: original Prime-LM cumulative corpus
- `primelm_semsim`: original Prime-LM semantic-similarity corpora

Active preset for the first pass:

- `paper_main`

Interpreter rule:

- use `.venv/bin/python` explicitly
- or use `make transitive-priming` and `make transitive-report`
- do not rely on the shell default `python`, which may point to a different
  interpreter than the project venv

## Submission strategy

Target a venue where a compact, careful LM behavioral study fits.

Good fit:

- workshop paper
- short paper
- focused computational linguistics venue

Bad fit:

- broad theory paper
- mechanistic interpretability paper
- clinically framed modeling paper

## Decision rule

Proceed with the paper only if:

- the token-level curves are interpretable
- `jabberwocky` remains above zero in the structure region
- the effect direction is consistent across active and passive targets

If those fail, stop the paper rather than broadening the claim.

## Timeline updates

### 2026-03-23

- Completed full `paper_main` run (`CORE`, `ANOMALOUS`, strict BPE-filtered `jabberwocky`) with checkpointed condition outputs and merged root outputs.
- Added reproducible statistics pipeline: `scripts/5_analyze_transitive_statistics.py` and `make transitive-stats`.
- Implemented item-paired inference on `sentence_pe_mean` with sign-flip permutation (`n=10,000`) and bootstrap CI (`n=10,000`), plus secondary metrics.
- Added confound-aware delta regression and per-condition LMM robustness with covariates for `target_length` and `critical_word_token_count`.
- Fixed local preset paths in scoring script to `corpora/transitive/*` to match current repository structure.

## Updated strategy after 2026-03-23

- Primary claim to carry forward:
  passive-over-active priming asymmetry remains robust under strict Jabberwocky lexical degradation.
- Primary reporting metric:
  `sentence_pe_mean` (token-normalized) and `critical_word_pe_mean`.
- Secondary reporting metric:
  raw `sentence_pe` only as descriptive, explicitly flagged as token-length sensitive.
- Required robustness blocks in Results:
  paired permutation/CI results, delta regression with confounds, per-condition LMM coefficients.
- Mandatory limitation statement:
  current evidence supports morphosyntactic-cue-sensitive priming, not fully isolated abstract syntax.
- Current best interpretation:
  the strongest Jabberwocky effects are concentrated around passive morphosyntactic cues (auxiliary/function-word zones), suggesting a morphology-heavy component.

## Next experiment priority (still within this paper path)

- Add a morphology-isolation control analysis:
  recompute priming with cue-token residualization/exclusion windows (`was`, `is`, `by`, determiners) and report whether residual structure-sensitive priming remains.
- If residual priming remains above zero, include this as the final robustness figure for Experiment 1.
- Only after this step, decide whether to run Experiment 2 (recency dynamics) for paper completion.
