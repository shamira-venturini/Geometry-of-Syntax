# Agrammatic-Like Editing Study

## Status

This document is intended to be moved into a separate repository.
It is written as a project specification, not as a note for the current
`Geometry-of-Syntax` repository.

## Working title

Inducing an Agrammatic-Like Morphosyntactic Syndrome in Autoregressive Language Models

## One-sentence project goal

Causally perturb a language model so that morphosyntactic competence degrades in a
targeted way while lexical-semantic competence remains comparatively intact.

## Scope

This is not a disease model of primary progressive aphasia.
This is a computational lesion study inspired by the behavioral profile of
agrammatic PPA.

Recommended phrasing:

- "agrammatic-like"
- "syntactic syndrome"
- "targeted morphosyntactic impairment"

Avoid phrasing such as:

- "simulating PPA"
- "modeling neurodegeneration"
- "capturing the disease"

## Scientific motivation

The motivating clinical pattern is selective impairment in:

- grammatical morphology
- functional categories
- verb argument structure
- noncanonical and complex syntax

with weaker impact on lexical-semantic knowledge.

The project asks whether transformer computations contain partly separable
mechanisms for morphosyntax that can be causally disrupted.

## Main research questions

1. Can we localize transformer components that make substantial causal
   contributions to morphosyntactic performance?
2. Can targeted interventions on those components induce a selective syntactic
   impairment profile?
3. Can we dissociate syntax-sensitive degradation from broad semantic or fluency
   collapse?
4. Do different syntactic domains rely on overlapping or separable mechanisms?

## Core project design

The project should be organized into four stages:

1. build a syntax-focused evaluation battery
2. localize syntax-relevant mechanisms
3. edit or suppress the relevant mechanisms
4. quantify syndrome specificity

## Stage 1: Evaluation battery

### Target capabilities to measure

- subject-verb agreement
- tense and inflectional morphology
- auxiliary selection and support
- determiner and function-word use
- passive comprehension/production preference
- filler-gap or relative clause syntax
- verb subcategorization and argument structure
- dative alternation only if the materials are cleanly controlled

### Required benchmark categories

#### Morphosyntax battery

Use minimal pairs whenever possible.

Candidate sources:

- BLiMP-style minimal pairs
- SyntaxGym suites
- custom targeted prompts for auxiliary omission, inflection, and passives

### Control batteries

You need explicit non-syntactic controls.

At minimum:

- lexical semantics / word association
- factual completion
- short-range fluency / local perplexity
- broad next-token loss on a held-out corpus slice

Reason:

Without controls, any successful "lesion" may just be a general capability drop.

## Stage 2: Localization

### Required methods

Use causal methods, not only correlational probes.

Recommended sequence:

1. baseline behavior on the full battery
2. activation patching between good and bad syntactic contexts
3. head- and MLP-level ablation screens
4. path patching or mediation analysis on top candidate components

### Localization targets

Search for:

- attention heads involved in long-distance agreement
- heads or MLPs implicated in auxiliary and function-word predictions
- components active around subject, verb, and clause-boundary positions

### Model choice

Start with small or medium causal LMs that are practical for interpretability.

Recommended first-pass criteria:

- open weights
- mature tooling
- manageable layer count

Do not start with the largest available model.

## Stage 3: Interventions

### Intervention families

Use more than one intervention type.

Recommended options:

- zero ablation of selected heads
- mean ablation at selected positions
- learned linear dampening of selected residual directions
- low-rank edits to selected MLP modules
- run-time activation steering with negative syntax-associated directions

### Intervention principle

Prefer small, targeted edits over global degradation.

Success means:

- syntax drops clearly
- semantics drops less
- general fluency remains usable

Failure modes:

- global perplexity explosion
- degeneration on all tasks
- only one benchmark drops while others stay unchanged for trivial reasons

## Stage 4: Syndrome evaluation

### Primary outcome

Create a composite morphosyntax impairment score.

Recommended components:

- agreement accuracy delta
- auxiliary/function-word accuracy delta
- inflection accuracy delta
- passive/noncanonical sentence preference delta
- argument-structure accuracy delta

### Secondary outcome

Create a preservation score for non-syntactic capacities.

Recommended components:

- lexical semantic task accuracy
- factual QA or completion stability
- held-out perplexity change
- lexical choice stability in simple clauses

### Success criterion

A successful intervention produces:

- large negative change on the morphosyntax composite
- substantially smaller change on semantic and general-language controls

This should be treated as a dissociation problem.

## Repository structure for the new project

Recommended layout:

```text
agrammatic-editing/
  README.md
  pyproject.toml
  configs/
    models/
    eval/
    interventions/
  data/
    raw/
    processed/
    custom_suites/
  src/
    eval/
    localization/
    interventions/
    analysis/
    utils/
  notebooks/
  reports/
    figures/
    tables/
  scripts/
    run_eval.py
    run_localization.py
    run_intervention.py
    summarize_results.py
  tests/
```

## Experimental roadmap

### Phase A: build the battery

Deliverables:

- reproducible syntax benchmark runner
- reproducible control benchmark runner
- baseline report for 1-2 models

Exit criterion:

- stable baseline results across reruns

### Phase B: locate syntax-relevant components

Deliverables:

- ranked list of candidate heads/MLPs by syntactic contribution
- per-task localization maps
- evidence that syntax-related effects are not identical across all tasks

Exit criterion:

- at least one compact set of components with strong causal effect on syntax tasks

### Phase C: lesion and edit

Deliverables:

- at least two intervention methods
- full pre/post evaluation on syntax and control tasks
- syndrome specificity analysis

Exit criterion:

- one intervention yields a clear dissociation

### Phase D: writeup

Deliverables:

- main paper figures
- ablation tables
- error analysis on edited outputs

## Main paper figures

Recommended figure set:

1. Baseline syntax vs control performance by task
2. Localization heatmap across layers/components
3. Pre/post intervention syntax impairment profile
4. Pre/post intervention control-task preservation profile
5. Example completions showing selective auxiliary/inflection/passive errors

## Risks

### Risk 1: no clean dissociation

Interpretation:

- syntax may not be neatly localized at this model scale
- your intervention may be too crude

Mitigation:

- test multiple intervention granularities
- target narrower phenomena first, such as agreement and auxiliaries

### Risk 2: the battery is too broad

Interpretation:

- "syntax" may fragment into partially independent computations

Mitigation:

- report subsyndromes instead of forcing one unitary score

### Risk 3: edits only affect benchmark artifacts

Mitigation:

- include hand-built sanity suites
- inspect generations qualitatively
- test on out-of-distribution templates

## What not to carry over from the current repo

- structural priming as the main causal assay
- hard-coded critical-token logic
- broad PCA claims without strong causal validation
- mixed construction families inside one syntactic category

## Immediate implementation plan

1. Pick one model family and one interpretability stack.
2. Implement the syntax and control battery first.
3. Run localization on agreement and auxiliaries before tackling passives.
4. Define the morphosyntax composite and preservation score before any edits.
5. Only then start intervention experiments.

## First-pass project recommendation

If you want the fastest route to a serious result, start with:

- one medium open-weight autoregressive model
- agreement, auxiliaries, inflection, and passive syntax
- head/MLP ablation plus activation patching

Do not start with datives, steering, or full clinical analogy.

## Decision rule

Continue the project only if you can show a selective impairment profile.
If every intervention causes broad collapse, the project is still informative, but
the main claim must become about limits of syntactic dissociation in current LMs.

## Timeline updates

### 2026-03-23

- The transitive priming pipeline in the source repo was stabilized and fully rerun with strict BPE-filtered Jabberwocky materials.
- New statistics confirmed a robust passive-over-active asymmetry in `CORE` and `jabberwocky`, which remain the active conditions for the paper path.
- Diagnostic interpretation shifted:
  strongest Jabberwocky effects appear concentrated in morphosyntactic cue regions (auxiliary/function-word zones), not nonce lexical stems.
- Consequence for this project:
  priming remains useful as an assay, but as a morphology-sensitive probe rather than a standalone pure-syntax claim.

## Priming as a means to an end (updated plan)

Use priming as one assay inside the editing project, not as the project center.

### Bridge Stage P0: build morphology-sensitive priming assay

- Keep fixed-target priming with active/passive contrasts.
- Add cue-controlled variants:
  cue-inclusive score, cue-excluded score, and cue-residualized score.
- Define morphology-cue priming index:
  passive-minus-active effect concentrated on auxiliary/function-word cue windows.
- Define residual-structure priming index:
  passive-minus-active effect after cue removal/residualization.

Exit criterion:

- assay distinguishes morphology-cue-heavy effects from residual structure effects with stable estimates.

### Bridge Stage P1: localization using priming gradients

- Use P0 indices as targets in localization screens (heads/MLPs by contribution).
- Rank components by differential impact on:
  morphology-cue priming index vs lexical-semantic controls.
- Carry top-ranked components into intervention stage.

Exit criterion:

- compact component set that shifts morphology-cue priming with limited semantic collateral.

### Bridge Stage P2: intervention objective tied to syndrome profile

- Intervention goal:
  reduce morphology-cue priming and morphosyntax battery accuracy in parallel.
- Preservation goal:
  keep lexical semantics, factual completion, and broad fluency within pre-registered tolerance.
- Treat priming as one quantitative endpoint in a multi-endpoint syndrome objective.

Exit criterion:

- at least one intervention yields selective morphosyntactic degradation with preserved controls.

## Immediate cross-repo transfer checklist

- Move the stabilized priming scripts and stats outputs into a dedicated `assays/priming/` module in the new repo.
- Freeze one assay configuration as version `priming_assay_v1` before any editing experiments.
- Register priming-derived metrics alongside non-priming syntax/control metrics in the main evaluation table.

## Transition plan from priming study to editing study

### 2026-03-23 action plan

Before fully switching projects, complete one short closure pass on the priming study:

1. Freeze `priming_assay_v1`:
   fixed stimuli set, fixed scoring script, fixed stats script, fixed output schema.
2. Add one final descriptive analysis:
   priming contribution by token class (`was/by/ed/function/content`) to document where the effect is concentrated.
3. Lock three carried-forward endpoints for editing:
   - passive-minus-active `sentence_pe_mean`
   - morphology-cue-zone priming index
   - lexical-semantic/control preservation index

Then start the agrammatic-like project using mechanistic interpretability methods.

### Mechanistic interpretability pipeline (recommended)

1. Baseline battery (pre-edit):
   run priming assay + agreement/auxiliary/inflection tasks + semantic/fluency controls.
2. Localization:
   activation patching, head/MLP ablation screening, and path patching around passive-cue and agreement positions.
3. Intervention:
   apply small targeted edits on top-ranked components (avoid global degradation edits first).
4. Post-edit evaluation:
   test for selective morphosyntactic impairment with minimal collateral damage on control tasks.

### Rationale for this transition

- Current priming results already identify a strong and stable morphosyntactic target signal.
- This signal can function as a causal objective in editing, not only as a behavioral reporting result.
- The priming assay should be treated as one component of a broader dissociation battery, not the sole endpoint.
