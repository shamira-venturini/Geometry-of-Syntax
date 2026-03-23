# Jabberwocky Priming Salvage Plan

## Purpose

This document reframes the current repository into a narrower, publishable project.
The goal is not to defend a broad "geometry of syntax" claim. The goal is to turn
the existing materials into a careful LLM structural priming study with one clear
extension beyond Prime-LM.

## Recommended paper claim

Recommended claim:

> Large language models retain measurable structural priming under severe lexical
> and semantic degradation, but the effect emerges late in the target sentence and
> is selectively strengthened by minimal semantic scaffolding.

This is a safer claim than:

- "syntax is isolated"
- "there is a syntax PC1"
- "Jabberwocky proves purely abstract syntax"

## What to keep

- The prime-target design inherited from Prime-LM.
- The transitive active/passive alternation.
- The Jabberwocky transitive corpus.
- The idea of comparing `CORE`, `ANOMALOUS`, and `jabberwocky`.
- The hybrid Jabberwocky idea as a semantic scaffolding manipulation.

## What to drop

- The current PCA/PC1/steering narrative as the main result.
- Any strong claim that one low-dimensional direction is a syntax mechanism.
- The current dative materials in their present form.

Reasons:

- The current null tests do not isolate a syntax-specific PC1 cleanly.
- The dative set mixes `to`-datives and `for`-benefactives, so the alternation is
  conceptually impure.
- The hard-coded token index for dative critical positions does not generalize to
  the hybrid materials.

## Main paper structure

### Experiment 1: Transitive structural priming under lexical degradation

Question:

Does structural priming survive when lexical content is replaced with Jabberwocky
items?

Design:

- Models: start with `gpt2-large`, then extend to 2-4 current base models.
- Conditions: `CORE`, `ANOMALOUS`, `jabberwocky`.
- Structure: active/passive only.
- Metric: sentence-level PE and token-level PE across the full target.

Required changes:

- Recompute token-level PE for every target token, not one preselected token.
- Report divergence-point PE as the primary metric, not full-sentence PE.
- Compare effect size before and after target divergence.

Core prediction:

- Priming remains above zero in Jabberwocky.
- Much of the early target contribution is structure-irrelevant carryover.
- The informative priming signal emerges from the divergence region onward.

### Experiment 2: Semantic scaffolding in Jabberwocky

Question:

Does minimal semantic support increase structural commitment in otherwise
content-poor materials?

Design:

- Stay with one alternation only.
- Best option: transitive scaffolding first, because the current transitive
  alternation is cleaner than the dative one.
- Replace a subset of nouns with semantically constrained human-denoting items or
  pronouns only if that manipulation is theoretically justified.

If you insist on dative scaffolding:

- Regenerate the dative corpus from scratch.
- Use only one construction family:
  - either `give/send ... to ...`
  - or benefactive `make/build ... for ...`
- Do not mix `to` and `for` within the same "PO" bucket.

Core prediction:

- Minimal semantic scaffolding increases late token-level priming more than early
  token-level priming.

## Minimal viable revision of the current pipeline

### 1. Replace single-token analysis with full token trajectories

For each target token `i`, compute:

`w_PE(X, i) = log P(T_X_i | P_X, T_X_<i) - log P(T_X_i | P_Y, T_X_<i)`

Then plot the average token-wise curves by condition.

Why:

- The current one-token summary throws away the most informative part of the
  signal.
- Prior work already shows that sentence-level PE can be dominated by shared
  prefix effects.

### 2. Use divergence-point PE as the main dependent variable

Define a target-divergence index and sum token-level PEs only from that point
forward.

For transitives:

- active/passive divergence starts at the auxiliary or participial region, not at
  the beginning of the target string.

For datives:

- divergence must be computed from the actual tokenization of each structure, not
  from a fixed word index.

### 3. Make the item design balanced across conditions

Current problem:

- `CORE` and `ANOMALOUS` targets are repeated about 10 times each.
- Jabberwocky targets are almost unique.

Required fix:

- either regenerate all conditions with matched target repetition
- or downsample to a target-matched subset before analysis

Without this, condition comparisons remain confounded by target inventory.

### 4. Evaluate on multiple models

Minimum recommendation:

- `gpt2-large`
- one modern small base model
- one modern medium base model

Avoid instruction-tuned models in the first pass.

Reason:

- the cleaner contribution is about autoregressive syntactic sensitivity, not chat
  alignment behavior

## Suggested title directions

- Structural Priming Survives Jabberwocky in Language Models
- Late Structural Commitment in LLMs Under Lexical Degradation
- Semantic Scaffolding Restores Structural Priming in Jabberwocky Language Model Input

## Publishability threshold

This becomes worth submitting if all of the following hold:

- token-wise curves show a clear late-emerging priming effect
- the effect replicates across more than one model
- the target-matching confound is removed
- the scaffolding manipulation produces a selective, interpretable change

If those do not hold, this should stay a side analysis rather than a paper.

## Concrete next steps in this repo

1. Rebuild the scoring code so it outputs token-level PE vectors for each item.
2. Recompute transitive results with divergence-point summaries.
3. Focus the first pass strictly on `CORE`, `ANOMALOUS`, and `jabberwocky`.
4. Audit target repetition and create matched subsets across conditions.
5. Decide whether to repair or abandon the dative branch later.
6. Only if the behavioral results are clean, revisit internal analyses.

## Recommendation

The best use of this repository is a restrained behavioral paper, not a mechanistic
or clinical modeling paper. Treat it as a syntax-under-degradation project.
