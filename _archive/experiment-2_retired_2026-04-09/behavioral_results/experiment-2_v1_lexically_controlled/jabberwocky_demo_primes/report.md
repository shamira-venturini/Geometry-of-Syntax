# Demonstration-based completion-choice production experiment

## Summary

```csv
prompt_template,prime_condition,n_items,n_active_choice,n_passive_choice,active_choice_rate,passive_choice_rate,mean_passive_minus_active_logprob,sd_passive_minus_active_logprob
demo__involving_event__did_to__mary_answered,active,2080,2061,19,0.9908653846153846,0.009134615384615385,-3.4790800112944384,1.8359519341366062
demo__involving_event__did_to__mary_answered,filler,2080,1869,211,0.8985576923076923,0.10144230769230769,-1.201776827298678,1.006862757670496
demo__involving_event__did_to__mary_answered,no_demo,2080,1807,273,0.86875,0.13125,-1.1923480107234075,1.1077985474173822
demo__involving_event__did_to__mary_answered,passive,2080,228,1852,0.10961538461538461,0.8903846153846153,1.8727766926472003,1.5860407623076924
```

## Pairwise Condition Comparisons

```csv
prompt_template,condition_a,condition_b,passive_choice_rate_a,passive_choice_rate_b,passive_choice_rate_diff_b_minus_a,active_choice_rate_a,active_choice_rate_b,active_choice_rate_diff_b_minus_a,mean_logprob_a,mean_logprob_b,mean_logprob_diff_b_minus_a
demo__involving_event__did_to__mary_answered,active,filler,0.009134615384615385,0.10144230769230769,0.09230769230769231,0.9908653846153846,0.8985576923076923,-0.09230769230769231,-3.4790800112944384,-1.201776827298678,2.2773031839957607
demo__involving_event__did_to__mary_answered,active,no_demo,0.009134615384615385,0.13125,0.12211538461538463,0.9908653846153846,0.86875,-0.12211538461538463,-3.4790800112944384,-1.1923480107234075,2.286732000571031
demo__involving_event__did_to__mary_answered,active,passive,0.009134615384615385,0.8903846153846153,0.88125,0.9908653846153846,0.10961538461538461,-0.8812500000000001,-3.4790800112944384,1.8727766926472003,5.351856703941639
demo__involving_event__did_to__mary_answered,no_demo,filler,0.13125,0.10144230769230769,-0.029807692307692313,0.86875,0.8985576923076923,0.029807692307692313,-1.1923480107234075,-1.201776827298678,-0.009428816575270504
demo__involving_event__did_to__mary_answered,passive,filler,0.8903846153846153,0.10144230769230769,-0.7889423076923077,0.10961538461538461,0.8985576923076923,0.7889423076923077,1.8727766926472003,-1.201776827298678,-3.074553519945878
demo__involving_event__did_to__mary_answered,passive,no_demo,0.8903846153846153,0.13125,-0.7591346153846154,0.10961538461538461,0.86875,0.7591346153846155,1.8727766926472003,-1.1923480107234075,-3.065124703370608
```

## Paired Significance Tests

```csv
prompt_template,metric,condition_a,condition_b,n_items,mean_diff_b_minus_a,sd_diff,effect_size_dz,t_stat,t_p_two_sided,perm_p_two_sided,bootstrap_ci95_low,bootstrap_ci95_high,mcnemar_b,mcnemar_c,mcnemar_p_exact
demo__involving_event__did_to__mary_answered,logprob_delta,active,filler,2080,2.2773031839957603,1.8803626150852943,1.2110978838475048,55.23456178209981,0.0,0.0,2.1969034259594404,2.3588735107275154,,,
demo__involving_event__did_to__mary_answered,passive_choice_delta,active,filler,2080,0.09230769230769231,0.3041139465484812,0.3035299543323536,13.843095788448528,9.394963095801353e-42,0.0,0.07932692307692307,0.10576923076923077,201.0,9.0,2.339656211288732e-48
demo__involving_event__did_to__mary_answered,logprob_delta,active,no_demo,2080,2.286732000571031,1.941169375439863,1.1780177605845779,53.72587603795423,0.0,0.0,2.2027723255524267,2.371046768839543,,,
demo__involving_event__did_to__mary_answered,passive_choice_delta,active,no_demo,2080,0.12211538461538461,0.33762225783971794,0.3616923404183779,16.495708719665046,1.5186697710269644e-57,0.0,0.1076923076923077,0.13653846153846153,261.0,7.0,7.88694561356351e-68
demo__involving_event__did_to__mary_answered,logprob_delta,active,passive,2080,5.351856703941639,2.9511951711430973,1.8134540054389845,82.70622766196513,0.0,0.0,5.226259319553008,5.476913417279721,,,
demo__involving_event__did_to__mary_answered,passive_choice_delta,active,passive,2080,0.88125,0.3235718984225507,2.723505979030295,124.2109834960364,0.0,0.0,0.8673076923076923,0.8951923076923077,1833.0,0.0,0.0
demo__involving_event__did_to__mary_answered,logprob_delta,no_demo,filler,2080,-0.009428816575270433,0.8792366211420136,-0.010723866987050234,-0.4890835840266645,0.6248341143191327,0.622,-0.04824876051682692,0.02863214492797851,,,
demo__involving_event__did_to__mary_answered,passive_choice_delta,no_demo,filler,2080,-0.02980769230769231,0.3298168943204453,-0.09037648713874429,-4.121801985695382,3.906788496791538e-05,0.0,-0.04423076923076923,-0.015865384615384615,83.0,145.0,4.843544745277168e-05
demo__involving_event__did_to__mary_answered,logprob_delta,passive,filler,2080,-3.074553519945878,1.7601786902541878,-1.7467280662862026,-79.66305662041864,0.0,0.0,-3.150051554143429,-2.9968183735013008,,,
demo__involving_event__did_to__mary_answered,passive_choice_delta,passive,filler,2080,-0.7889423076923077,0.4116775901460247,-1.9164081955796155,-87.4016611623384,0.0,0.0,-0.8067307692307693,-0.7711538461538462,3.0,1644.0,0.0
demo__involving_event__did_to__mary_answered,logprob_delta,passive,no_demo,2080,-3.065124703370608,1.832286853799069,-1.6728410712631427,-76.29329118203005,0.0,0.0,-3.1439383803422634,-2.9876171166392473,,,
demo__involving_event__did_to__mary_answered,passive_choice_delta,passive,no_demo,2080,-0.7591346153846154,0.43771590140776284,-1.7343089728819987,-79.09665881635928,0.0,0.0,-0.7778846153846154,-0.7399038461538462,9.0,1588.0,0.0
```

Interpretation:
- `passive_choice_rate` is the share of items where the passive option outranked the active option.
- `mean_passive_minus_active_logprob` is the mean passive-vs-active structural preference score.
- In `comparison.csv` and `stats.csv`, differences are always `condition_b - condition_a`.
- `passive_choice_delta` tests paired shifts in passive choice rates across prime conditions.
- `logprob_delta` tests paired shifts in passive-vs-active preference across prime conditions.