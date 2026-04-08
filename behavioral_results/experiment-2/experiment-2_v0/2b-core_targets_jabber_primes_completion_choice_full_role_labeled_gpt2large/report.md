# CORE Completion-Choice Priming Pilot

## Summary

```csv
prompt_template,prime_structure,n_items,n_active_choice,n_passive_choice,active_choice_rate,passive_choice_rate,mean_passive_minus_active_logprob,sd_passive_minus_active_logprob
role_labeled,active,15000,8487,6513,0.5658,0.4342,-0.3224179532210032,1.8574665514244442
role_labeled,passive,15000,8099,6901,0.5399333333333334,0.4600666666666667,-0.20430176926453908,1.8051916741019005
```

## Comparison

```csv
prompt_template,passive_choice_rate_after_active_prime,passive_choice_rate_after_passive_prime,passive_choice_rate_shift,active_choice_rate_after_active_prime,active_choice_rate_after_passive_prime,active_choice_rate_shift,mean_logprob_shift
role_labeled,0.4342,0.4600666666666667,0.025866666666666704,0.5658,0.5399333333333334,0.025866666666666593,0.11811618395646412
```

## Paired Significance Tests

```csv
prompt_template,metric,n_items,mean_diff,sd_diff,effect_size_dz,t_stat,t_p_two_sided,perm_p_two_sided,bootstrap_ci95_low,bootstrap_ci95_high,mcnemar_b,mcnemar_c,mcnemar_p_exact
role_labeled,logprob_delta,15000,0.11811618395646413,0.5910520056140125,0.19984059411787208,24.4753742741712,9.348264091279578e-130,0.0,0.10864489965538184,0.12753906899909176,,,
role_labeled,passive_choice_delta,15000,0.025866666666666666,0.3160235860649521,0.08185043081357322,10.02458953601159,1.410029742255338e-23,0.0,0.020866666666666665,0.030866666666666667,948.0,560.0,1.229649062438366e-23
```

Interpretation:
- `passive_choice_rate_shift` is the key priming statistic in this pilot.
- Positive values mean passive primes increase passive first-noun choices relative to active primes.
- `mean_logprob_shift` compares patient-vs-agent noun preference after passive versus active primes.
- `passive_choice_delta` is the paired item-level passive-choice difference (passive-prime minus active-prime).
- `logprob_delta` is the paired item-level shift in patient-vs-agent noun preference.