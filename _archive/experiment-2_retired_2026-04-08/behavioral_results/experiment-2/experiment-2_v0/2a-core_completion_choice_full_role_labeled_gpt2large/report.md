# CORE Completion-Choice Priming Pilot

## Summary

```csv
prompt_template,prime_structure,n_items,n_active_choice,n_passive_choice,active_choice_rate,passive_choice_rate,mean_passive_minus_active_logprob,sd_passive_minus_active_logprob
role_labeled,active,15000,9322,5678,0.6214666666666666,0.37853333333333333,-0.5642272084792455,1.8750742394626034
role_labeled,passive,15000,8447,6553,0.5631333333333334,0.4368666666666667,-0.2975034530719121,1.8765200816002767
```

## Comparison

```csv
prompt_template,passive_choice_rate_after_active_prime,passive_choice_rate_after_passive_prime,passive_choice_rate_shift,active_choice_rate_after_active_prime,active_choice_rate_after_passive_prime,active_choice_rate_shift,mean_logprob_shift
role_labeled,0.37853333333333333,0.4368666666666667,0.05833333333333335,0.6214666666666666,0.5631333333333334,0.05833333333333324,0.26672375540733334
```

## Paired Significance Tests

```csv
prompt_template,metric,n_items,mean_diff,sd_diff,effect_size_dz,t_stat,t_p_two_sided,perm_p_two_sided,bootstrap_ci95_low,bootstrap_ci95_high,mcnemar_b,mcnemar_c,mcnemar_p_exact
role_labeled,logprob_delta,15000,0.2667237554073334,0.6200359247364511,0.4301746798311815,52.68542329257584,0.0,0.0,0.25688387024879455,0.2765490019573768,,,
role_labeled,passive_choice_delta,15000,0.058333333333333334,0.31602301644872416,0.18458571147395564,22.60704034598948,2.6531525499264602e-111,0.0,0.053266666666666664,0.06333333333333334,1212.0,337.0,8.130757279793204e-116
```

Interpretation:
- `passive_choice_rate_shift` is the key priming statistic in this pilot.
- Positive values mean passive primes increase passive first-noun choices relative to active primes.
- `mean_logprob_shift` compares patient-vs-agent noun preference after passive versus active primes.
- `passive_choice_delta` is the paired item-level passive-choice difference (passive-prime minus active-prime).
- `logprob_delta` is the paired item-level shift in patient-vs-agent noun preference.