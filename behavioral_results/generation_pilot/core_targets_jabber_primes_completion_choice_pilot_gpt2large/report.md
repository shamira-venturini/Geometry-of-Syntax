# CORE Completion-Choice Priming Pilot

## Summary

```csv
prompt_template,prime_structure,n_items,n_active_choice,n_passive_choice,active_choice_rate,passive_choice_rate,mean_passive_minus_active_logprob,sd_passive_minus_active_logprob
role_labeled,active,200,108,92,0.54,0.46,-0.27509740352630613,1.9177005325517864
role_labeled,passive,200,101,99,0.505,0.495,-0.10322565674781799,1.8218218741080157
word_list,active,200,170,30,0.85,0.15,-1.6215987890958785,1.600726257224799
word_list,passive,200,164,36,0.82,0.18,-1.4194614332914353,1.4786497424501914
```

## Comparison

```csv
prompt_template,passive_choice_rate_after_active_prime,passive_choice_rate_after_passive_prime,passive_choice_rate_shift,active_choice_rate_after_active_prime,active_choice_rate_after_passive_prime,active_choice_rate_shift,mean_logprob_shift
role_labeled,0.46,0.495,0.034999999999999976,0.54,0.505,0.03500000000000003,0.17187174677848815
word_list,0.15,0.18,0.03,0.85,0.82,0.030000000000000027,0.20213735580444325
```

## Paired Significance Tests

```csv
prompt_template,metric,n_items,mean_diff,sd_diff,effect_size_dz,t_stat,t_p_two_sided,perm_p_two_sided,bootstrap_ci95_low,bootstrap_ci95_high,mcnemar_b,mcnemar_c,mcnemar_p_exact
role_labeled,logprob_delta,200,0.17187174677848815,0.6380813016982575,0.26935712787861105,3.809285033677959,0.00018564648262101903,0.0001,0.0861781081855297,0.26111170575022696,,,
role_labeled,passive_choice_delta,200,0.035,0.3381519387321943,0.10350376854624188,1.4637643323482106,0.14483615305716602,0.2217,-0.01,0.08,15.0,8.0,0.21003961563110352
word_list,logprob_delta,200,0.20213735580444336,0.5607239475948063,0.36049353103519144,5.09814840737734,7.958832627109231e-07,0.0,0.12412687057256698,0.2813332606554031,,,
word_list,passive_choice_delta,200,0.03,0.26352843524163483,0.11383970755373082,1.6099365835907302,0.10899681557446599,0.1834,-0.005,0.065,10.0,4.0,0.1795654296875
```

Interpretation:
- `passive_choice_rate_shift` is the key priming statistic in this pilot.
- Positive values mean passive primes increase passive first-noun choices relative to active primes.
- `mean_logprob_shift` compares patient-vs-agent noun preference after passive versus active primes.
- `passive_choice_delta` is the paired item-level passive-choice difference (passive-prime minus active-prime).
- `logprob_delta` is the paired item-level shift in patient-vs-agent noun preference.