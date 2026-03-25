# CORE Completion-Choice Priming Pilot

## Summary

```csv
prompt_template,prime_structure,n_items,n_active_choice,n_passive_choice,active_choice_rate,passive_choice_rate,mean_passive_minus_active_logprob,sd_passive_minus_active_logprob
role_labeled,active,1,1,0,1.0,0.0,-0.5522918701171875,0.0
role_labeled,passive,1,1,0,1.0,0.0,-1.2448480129241943,0.0
```

## Comparison

```csv
prompt_template,passive_choice_rate_after_active_prime,passive_choice_rate_after_passive_prime,passive_choice_rate_shift,active_choice_rate_after_active_prime,active_choice_rate_after_passive_prime,active_choice_rate_shift,mean_logprob_shift
role_labeled,0.0,0.0,0.0,1.0,1.0,0.0,-0.6925561428070068
```

## Paired Significance Tests

```csv
prompt_template,metric,n_items,mean_diff,sd_diff,effect_size_dz,t_stat,t_p_two_sided,perm_p_two_sided,bootstrap_ci95_low,bootstrap_ci95_high,mcnemar_b,mcnemar_c,mcnemar_p_exact
role_labeled,logprob_delta,1,-0.6925561428070068,0.0,0.0,,,1.0,-0.6925561428070068,-0.6925561428070068,,,
role_labeled,passive_choice_delta,1,0.0,0.0,0.0,,,1.0,0.0,0.0,0.0,0.0,1.0
```

Interpretation:
- `passive_choice_rate_shift` is the key priming statistic in this pilot.
- Positive values mean passive primes increase passive first-noun choices relative to active primes.
- `mean_logprob_shift` compares patient-vs-agent noun preference after passive versus active primes.
- `passive_choice_delta` is the paired item-level passive-choice difference (passive-prime minus active-prime).
- `logprob_delta` is the paired item-level shift in patient-vs-agent noun preference.