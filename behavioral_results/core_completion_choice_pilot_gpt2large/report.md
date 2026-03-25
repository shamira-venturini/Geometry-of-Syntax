# CORE Completion-Choice Priming Pilot

## Summary

```csv
prompt_template,prime_structure,n_items,n_active_choice,n_passive_choice,active_choice_rate,passive_choice_rate,mean_passive_minus_active_logprob,sd_passive_minus_active_logprob
role_labeled,active,200,125,75,0.625,0.375,-0.35919766664505004,1.8739797697172118
role_labeled,passive,200,113,87,0.565,0.435,-0.08147949516773224,1.9115886095583194
word_list,active,200,181,19,0.905,0.095,-1.697718586921692,1.4126418931541112
word_list,passive,200,174,26,0.87,0.13,-1.445166580080986,1.4688269240874208
```

## Comparison

```csv
prompt_template,passive_choice_rate_after_active_prime,passive_choice_rate_after_passive_prime,passive_choice_rate_shift,active_choice_rate_after_active_prime,active_choice_rate_after_passive_prime,active_choice_rate_shift,mean_logprob_shift
role_labeled,0.375,0.435,0.06,0.625,0.565,0.06000000000000005,0.2777181714773178
word_list,0.095,0.13,0.035,0.905,0.87,0.03500000000000003,0.2525520068407059
```

## Paired Significance Tests

```csv
prompt_template,metric,n_items,mean_diff,sd_diff,effect_size_dz,t_stat,t_p_two_sided,perm_p_two_sided,bootstrap_ci95_low,bootstrap_ci95_high,mcnemar_b,mcnemar_c,mcnemar_p_exact
role_labeled,logprob_delta,200,0.2777181714773178,0.5859568085738076,0.4739567275500583,6.702760320792624,2.0595423256049514e-10,0.0,0.19519382704794405,0.3587504575848579,,,
role_labeled,passive_choice_delta,200,0.06,0.2770991151158552,0.21652902058136844,3.062182775535343,0.002501030926758089,0.0046,0.025,0.1,14.0,2.0,0.004180908203125
word_list,logprob_delta,200,0.2525520068407059,0.5351539396582669,0.47192403554382495,6.674013714759199,2.41766088668163e-10,0.0,0.17832270120084284,0.326014085739851,,,
word_list,passive_choice_delta,200,0.035,0.2531708462420874,0.138246565588094,1.955101680061842,0.05197139874474365,0.0971,0.0,0.07,10.0,3.0,0.09228515625
```

Interpretation:
- `passive_choice_rate_shift` is the key priming statistic in this pilot.
- Positive values mean passive primes increase passive first-noun choices relative to active primes.
- `mean_logprob_shift` compares patient-vs-agent noun preference after passive versus active primes.
- `passive_choice_delta` is the paired item-level passive-choice difference (passive-prime minus active-prime).
- `logprob_delta` is the paired item-level shift in patient-vs-agent noun preference.