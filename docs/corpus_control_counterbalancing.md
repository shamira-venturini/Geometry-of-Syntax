# Corpus Control and Counterbalancing Ledger

This document records the controls currently built into the strict transitive corpora and prompt materials. It is intended as the methods-facing checklist for explaining why CORE, Jabberwocky, and mixed-prime conditions are comparable.

## Active Materials

The current controlled backbone is:

- CORE corpus: `corpora/transitive/CORE_transitive_strict_4cell_counterbalanced.csv`
- Jabberwocky corpus: `corpora/transitive/jabberwocky_transitive_gpt2_monosyllabic_strict_4cell.csv`
- Mixed Experiment 2 corpus: `corpora/transitive/CORE_transitive_core_targets_jabberwocky_primes_2048.csv`
- Experiment 2 CORE prompts: `corpora/transitive/experiment_2_core_demo_prompts_lexically_controlled.csv`
- Experiment 2 Jabberwocky prompts: `corpora/transitive/experiment_2_jabberwocky_demo_prompts_lexically_controlled.csv`
- Experiment 2 Jabberwocky-prime/CORE-target prompts: `corpora/transitive/experiment_2_core_targets_jabberwocky_primes_demo_prompts_lexically_controlled.csv`

Older independently generated Jabberwocky materials are retired for strict-design analyses. They can be used as provenance/design history only.

## Shared Backbone Across Experiments

Experiments 1b, 2, and 3 use the same fixed item backbone wherever the task permits it.

- The same row inventory of `pa`, `pp`, `ta`, and `tp` items is reused.
- The target sentences after `no_prime` are the same item-level `ta` and `tp` targets across experiments.
- Differences across experiments are intended to come from task format, prompting, and readout, not from changing prime/target materials.
- Experiment 2 additionally includes a mixed condition where Jabberwocky `pa`/`pp` primes are paired with CORE `ta`/`tp` targets.

## Row Counts and Cell Balance

Each active 2048-row corpus is balanced across target and prime determiner/tense cells.

| Corpus | Rows | Target `def_past` | Target `def_present` | Target `indef_past` | Target `indef_present` |
| --- | ---: | ---: | ---: | ---: | ---: |
| CORE | 2048 | 512 | 512 | 512 | 512 |
| Jabberwocky | 2048 | 512 | 512 | 512 | 512 |
| Mixed Exp2 | 2048 | 512 | 512 | 512 | 512 |

The same 512/512/512/512 balance also holds for the prime cells in each corpus.

## Prime-Target Disjointness

Within each row, prime and target are deliberately disjoint along the dimensions most likely to create lexical or surface-form confounds.

Verified for CORE, Jabberwocky, and mixed Experiment 2 corpora:

- `0` rows with shared prime-target nouns.
- `0` rows with shared active verbs or active verb fragments.
- `0` rows with the same passive auxiliary in prime and target.
- `0` rows with the same determiner family in prime and target.

This means the prime and target always differ in:

- noun identities,
- active verb or active fragment,
- passive auxiliary: `is` vs `was`,
- determiner family: definite `the` vs indefinite `a/an`.

## Tense and Determiner Counterbalancing

The corpora use a four-cell target design:

- definite / past,
- definite / present,
- indefinite / past,
- indefinite / present.

Prime cells are balanced in parallel, but prime and target are mismatched within each row:

- If the target passive auxiliary is `is`, the prime passive auxiliary is `was`.
- If the target passive auxiliary is `was`, the prime passive auxiliary is `is`.
- If the target is definite, the prime is indefinite.
- If the target is indefinite, the prime is definite.

The tense/determiner alternation mainly protects prime-target disjointness and reduces surface overlap. It is not intended as a theoretical manipulation by itself.

## Active/Passive Target Balance

Every item has both an active and a passive target realization:

- `ta`: target active sentence,
- `tp`: target passive sentence.

The no-prime baseline therefore always compares the same active/passive target pair for a given item. The number of active and passive target candidates is matched by construction within every condition, including CORE, Jabberwocky, and mixed Experiment 2.

## Reversible-Verb and Thematic-Role Controls

The CORE corpus is designed around reversible transitive events, following the Sinclair-style item logic.

Controls encoded in the strict CORE design include:

- reversible human-role nouns for agent and patient positions,
- verbs compatible with role-reversible active/passive alternations,
- exact lemma-level role balance across the generated item set,
- no systematic verb preference for one thematic role to be the subject, insofar as this is inherited from the controlled item source and enforced by the generator’s balancing logic.

The goal is that active/passive preference is not trivially explained by one noun or verb being a more plausible subject independent of structure.

## Jabberwocky Mirroring

The Jabberwocky corpus is modeled directly on the CORE backbone rather than generated ex novo.

For each CORE row, Jabberwocky preserves:

- the same syntactic template skeleton,
- the same row position,
- the same target and prime determiner-family assignment,
- the same target and prime tense/auxiliary assignment,
- the same active/passive structural alternation,
- the same four-cell distribution.

The lexical material is replaced with meaning-poor forms:

- nonce nouns are one-token GPT-2-large monosyllabic forms,
- active present verbs are the standalone fragment `s`,
- active past verbs are the standalone fragment `ed`,
- passive participles are the standalone fragment `ed`.

Jabberwocky determiners preserve determiner family, not exact allomorph spelling. Indefinites use phonologically appropriate `a/an`; because the selected nonce nouns are consonant-initial, current indefinite Jabberwocky NPs use `a`.

## Jabberwocky Tokenization Controls

The current Jabberwocky design is model-relative rather than human-normed nonce-word design.

For GPT-2-large:

- all 40 active nonce nouns are one token,
- sentence token counts are close to CORE,
- residual differences are small and should still be carried into analyses.

Current GPT-2-large sentence-tokenization audit:

| Column | CORE mean | Jabberwocky mean | Jabberwocky minus CORE mean | Exact rows | Positive diff rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pa` | 6.140625 | 6.0 | -0.140625 | 1760 | 0 |
| `pp` | 8.03125 | 8.0 | -0.03125 | 1984 | 0 |
| `ta` | 6.140625 | 6.0 | -0.140625 | 1760 | 0 |
| `tp` | 8.03125 | 8.0 | -0.03125 | 1984 | 0 |

The goal is not perfect token-count matching at any cost. The goal is a meaning-poor condition that mirrors CORE structurally while keeping tokenization artifacts small enough that they cannot explain the result.

## Filler Controls

Filler materials are domain-matched:

- CORE uses real intransitive filler sentences.
- Jabberwocky uses nonce intransitive filler sentences.

When fillers are selected from pools, they are screened to avoid overlap with target nouns where possible. Filler conditions are included as a baseline-like comparison but should not replace the no-prime baseline.

## Experiment 2 Prompt Controls

Experiment 2 prompt CSVs are exported deterministically from the same strict corpora.

The default prompt settings are:

- event style: `involving_event`,
- role style: `did_to`,
- quote style: `mary_answered`,
- role order: `counterbalanced`.

The three Experiment 2 prompt corpora each contain 2048 rows:

- CORE to CORE,
- Jabberwocky to Jabberwocky,
- Jabberwocky-prime to CORE-target.

Lexical overlap audits for the exported prompt corpora show:

- `0` same active verb rows,
- `0` shared noun rows,
- `0` shared-both-noun rows.

## Role-Order Counterbalancing

Experiments 2 and 3 counterbalance event role-description order:

- `agent_first`: agent line before patient line,
- `patient_first`: patient line before agent line.

This addresses the concern that the no-prime/demo scaffold could bias passive continuations merely because the patient line is closer to the answer slot.

Role order is exactly balanced in each Experiment 2 prompt corpus:

| Prompt corpus | `agent_first` | `patient_first` |
| --- | ---: | ---: |
| CORE | 1024 | 1024 |
| Jabberwocky | 1024 | 1024 |
| Mixed Exp2 | 1024 | 1024 |

Within each target determiner/tense cell, role order is also balanced:

- 256 `agent_first`,
- 256 `patient_first`.

The same `role_order` is used for the demonstration block and target block within an item row. `role_order` is preserved in Experiment 2 result metadata and Experiment 3 item-level metadata.

## Mixed Experiment 2 Controls

The mixed Experiment 2 condition tests Jabberwocky primes with CORE targets.

Controls verified in the mixed corpus:

- 2048 rows,
- same CORE target distribution as the CORE condition,
- balanced Jabberwocky prime cells,
- `0` same auxiliary rows,
- `0` same determiner-family rows,
- `0` shared noun rows,
- `0` same active-verb rows,
- active tense matches the corresponding passive auxiliary for both prime and target.

This condition preserves the CORE target/no-prime baseline while changing the prime domain.

## Metadata Preserved for Analysis

Item-level outputs should retain:

- `item_id`,
- `item_index`,
- `item_order`,
- `model_condition`,
- `lexicality_condition`,
- `prime_condition`,
- `role_order`,
- active/passive candidate token counts,
- `passive_minus_active_token_count` where available,
- CORE/Jabberwocky token-count residuals where available.

Mixed models and sensitivity analyses should include:

- prime condition,
- lexicality or corpus condition,
- `role_order`,
- centered `item_order`,
- token-count covariates.

Where repeated target items appear across prime conditions, item-level random intercepts should be used if supported by the data. The strongest interpretation should be reserved for priming effects that survive token-count, role-order, and item-order controls.

## Known Tradeoffs

The design intentionally prioritizes structural comparability and model-facing tokenization control over human-style nonce naturalness.

Important caveats:

- Jabberwocky nouns are tokenizer-aligned fragments, not normed human nonwords.
- Jabberwocky verbs use standalone inflectional fragments because the models tokenize nonce verb stems and suffixes separately anyway.
- CORE and Jabberwocky token counts are close but not perfectly identical.
- Llama-specific tokenization should be audited separately from GPT-2-large before Llama conclusions are finalized.
- Any regenerated corpus must be re-audited before use.

## Regeneration and Audit Scripts

Relevant scripts:

- Build strict Jabberwocky: `scripts/42_build_gpt2_monosyllabic_jabberwocky_strict_4cell.py`
- Build mixed CORE-target/Jabberwocky-prime corpus: `scripts/34_build_mixed_core_targets_jabberwocky_primes.py`
- Export Experiment 2 prompt CSVs: `scripts/28_export_demo_prompt_csvs.py`
- Run Experiment 2 completion-choice variant: `scripts/24_demo_prompt_completion_experiment.py`
- Run Experiment 2 generation audit: `scripts/29_demo_prompt_generation_audit.py`
- Run Experiment 3: `run_experiment.py`

When materials are regenerated, re-check:

- row count,
- target and prime cell counts,
- prime-target noun overlap,
- prime-target active verb or fragment overlap,
- passive auxiliary mismatch,
- determiner-family mismatch,
- role-order balance,
- token-count summaries,
- prompt CSV metadata preservation.
