# Experiment 1b: Controlled Processing Structural Priming

## Summary

```csv
prompt_template,prime_condition,n_items,n_active_choice,n_passive_choice,active_choice_rate,passive_choice_rate,mean_passive_minus_active_logprob,sd_passive_minus_active_logprob,mean_passive_minus_active_logprob_sum,sd_passive_minus_active_logprob_sum
processing_teacher_forced,active,2048,6,2042,0.0029296875,0.9970703125,0.8757546146710713,0.38777804861355086,-6.523458957672119,2.7105314705636117
processing_teacher_forced,filler,2048,86,1962,0.0419921875,0.9580078125,0.6573576529820759,0.373912972213512,-7.32577109336853,2.5577761033128836
processing_teacher_forced,no_prime,2048,65,1983,0.03173828125,0.96826171875,1.6009175845554897,0.7258145738792204,-7.150423526763916,4.4444108230210535
processing_teacher_forced,passive,2048,0,2048,0.0,1.0,1.5207400818665824,0.3489480248199812,-0.8012182712554932,2.5308817664435534
```

## Pairwise Condition Comparisons

```csv
prompt_template,condition_a,condition_b,passive_choice_rate_a,passive_choice_rate_b,passive_choice_rate_diff_b_minus_a,active_choice_rate_a,active_choice_rate_b,active_choice_rate_diff_b_minus_a,mean_logprob_a,mean_logprob_b,mean_logprob_diff_b_minus_a,mean_logprob_sum_a,mean_logprob_sum_b,mean_logprob_sum_diff_b_minus_a
processing_teacher_forced,active,filler,0.9970703125,0.9580078125,-0.0390625,0.0029296875,0.0419921875,0.0390625,0.8757546146710713,0.6573576529820759,-0.21839696168899536,-6.523458957672119,-7.32577109336853,-0.8023121356964111
processing_teacher_forced,active,no_prime,0.9970703125,0.96826171875,-0.02880859375,0.0029296875,0.03173828125,0.02880859375,0.8757546146710713,1.6009175845554897,0.7251629698844184,-6.523458957672119,-7.150423526763916,-0.6269645690917969
processing_teacher_forced,active,passive,0.9970703125,1.0,0.0029296875,0.0029296875,0.0,-0.0029296875,0.8757546146710713,1.5207400818665824,0.6449854671955111,-6.523458957672119,-0.8012182712554932,5.722240686416626
processing_teacher_forced,no_prime,filler,0.96826171875,0.9580078125,-0.01025390625,0.03173828125,0.0419921875,0.01025390625,1.6009175845554897,0.6573576529820759,-0.9435599315734138,-7.150423526763916,-7.32577109336853,-0.17534756660461426
processing_teacher_forced,passive,filler,1.0,0.9580078125,-0.0419921875,0.0,0.0419921875,0.0419921875,1.5207400818665824,0.6573576529820759,-0.8633824288845064,-0.8012182712554932,-7.32577109336853,-6.524552822113037
processing_teacher_forced,passive,no_prime,1.0,0.96826171875,-0.03173828125,0.0,0.03173828125,0.03173828125,1.5207400818665824,1.6009175845554897,0.08017750268890733,-0.8012182712554932,-7.150423526763916,-6.349205255508423
```

## Paired Significance Tests

```csv
prompt_template,metric,condition_a,condition_b,n_items,mean_diff_b_minus_a,sd_diff,effect_size_dz,t_stat,t_p_two_sided,perm_p_two_sided,bootstrap_ci95_low,bootstrap_ci95_high,mcnemar_b,mcnemar_c,mcnemar_p_exact
processing_teacher_forced,logprob_delta,active,filler,2048,-0.21839696168899536,0.40957938266219684,-0.5332225471640002,-24.130897854798604,1.975244684323852e-113,0.0,-0.23605016519625985,-0.2006367827455203,,,
processing_teacher_forced,logprob_sum_delta,active,filler,2048,-0.8023121356964111,2.2988321880350395,-0.3490085704699477,-15.794324919777477,4.0911167240215303e-53,0.0,-0.902135843038559,-0.7045058667659759,,,
processing_teacher_forced,passive_choice_delta,active,filler,2048,-0.0390625,0.20836793394482225,-0.18746886462071516,-8.483872347817634,4.112049175914848e-17,0.0,-0.04833984375,-0.0302734375,6.0,86.0,3.090494990539711e-19
processing_teacher_forced,logprob_delta,active,no_prime,2048,0.7251629698844184,0.6400398049128545,1.1329966735165073,51.273576377940685,0.0,0.0,0.6977337451917786,0.7529632180503437,,,
processing_teacher_forced,logprob_sum_delta,active,no_prime,2048,-0.6269645690917969,4.029156137886239,-0.15560691808302884,-7.0419652464671545,2.577695909519799e-12,0.0,-0.7985016822814941,-0.4552193880081177,,,
processing_teacher_forced,passive_choice_delta,active,no_prime,2048,-0.02880859375,0.1839960983963668,-0.1565717642987198,-7.08562920178986,1.896468830791538e-12,0.0,-0.037109375,-0.02099609375,6.0,65.0,1.3321308760808536e-13
processing_teacher_forced,logprob_delta,active,passive,2048,0.6449854671955109,0.3503412087286619,1.8410208423270296,83.31509260257359,0.0,0.0,0.6300649692614874,0.6605011949936549,,,
processing_teacher_forced,logprob_sum_delta,active,passive,2048,5.722240686416626,2.1913622512041084,2.611270995141206,118.17263540352583,0.0,0.0,5.629119020700455,5.817617851495743,,,
processing_teacher_forced,passive_choice_delta,active,passive,2048,0.0029296875,0.0540604425477739,0.054192813856656794,2.452486794855828,0.014270198824074569,0.0293,0.0009765625,0.00537109375,6.0,0.0,0.03125
processing_teacher_forced,logprob_delta,no_prime,filler,2048,-0.9435599315734138,0.7089295553275088,-1.3309643031281313,-60.23256859258428,0.0,0.0,-0.9743855658315478,-0.9131524609384084,,,
processing_teacher_forced,logprob_sum_delta,no_prime,filler,2048,-0.17534756660461426,4.121914754497386,-0.042540318528735716,-1.9251550531523043,0.05434837410657703,0.0526,-0.35354522466659544,0.0025672733783721684,,,
processing_teacher_forced,passive_choice_delta,no_prime,filler,2048,-0.01025390625,0.2622519699346635,-0.039099444143564,-1.7694388540504797,0.07696952875871217,0.0949,-0.021484375,0.00146484375,60.0,81.0,0.09176922936810444
processing_teacher_forced,logprob_delta,passive,filler,2048,-0.8633824288845062,0.3355105197377855,-2.5733393681940973,-116.45604592283854,0.0,0.0,-0.8783285585542521,-0.8489620896677176,,,
processing_teacher_forced,logprob_sum_delta,passive,filler,2048,-6.524552822113037,2.293490467445246,-2.844813577699687,-128.74156620819275,0.0,0.0,-6.621974992752075,-6.424813216924667,,,
processing_teacher_forced,passive_choice_delta,passive,filler,2048,-0.0419921875,0.2006202788230458,-0.20931177917980362,-9.472369820176663,7.265397335909824e-21,0.0,-0.05078125,-0.03369140625,0.0,86.0,2.5849394142282115e-26
processing_teacher_forced,logprob_delta,passive,no_prime,2048,0.08017750268890747,0.6748058227510866,0.1188156651672555,5.376983203261226,8.438729736957481e-08,0.0,0.05041942091924807,0.10938666303952538,,,
processing_teacher_forced,logprob_sum_delta,passive,no_prime,2048,-6.349205255508423,4.082794605721645,-1.5551125806354844,-70.37636168165523,0.0,0.0,-6.528196620941162,-6.174493253231049,,,
processing_teacher_forced,passive_choice_delta,passive,no_prime,2048,-0.03173828125,0.17534530343472152,-0.1810044559409354,-8.191326606132295,4.496622243010133e-16,0.0,-0.03955078125,-0.0244140625,0.0,65.0,5.421010862427522e-20
```

## Sinclair-Style PE (Same Minus Other)

```csv
prompt_template,n_items,pe_active_target_logprob_same_minus_other,pe_active_target_logprob_ci95_low,pe_active_target_logprob_ci95_high,pe_active_target_logprob_p,pe_active_target_logprob_perm_p,pe_passive_target_logprob_same_minus_other,pe_passive_target_logprob_ci95_low,pe_passive_target_logprob_ci95_high,pe_passive_target_logprob_p,pe_passive_target_logprob_perm_p,pe_logprob_imbalance_passive_minus_active,pe_logprob_imbalance_ci95_low,pe_logprob_imbalance_ci95_high,pe_logprob_imbalance_p,pe_logprob_imbalance_perm_p,pe_active_target_logprob_sum_same_minus_other,pe_active_target_logprob_sum_ci95_low,pe_active_target_logprob_sum_ci95_high,pe_active_target_logprob_sum_p,pe_active_target_logprob_sum_perm_p,pe_passive_target_logprob_sum_same_minus_other,pe_passive_target_logprob_sum_ci95_low,pe_passive_target_logprob_sum_ci95_high,pe_passive_target_logprob_sum_p,pe_passive_target_logprob_sum_perm_p,pe_logprob_sum_imbalance_passive_minus_active,pe_logprob_sum_imbalance_ci95_low,pe_logprob_sum_imbalance_ci95_high,pe_logprob_sum_imbalance_p,pe_logprob_sum_imbalance_perm_p
processing_teacher_forced,2048,-0.28117847442626953,-0.30099244515101115,-0.26038681268692015,1.4805958280503337e-140,0.0,0.9261639416217804,0.9130416303873062,0.9395064786076546,0.0,0.0,1.20734241604805,1.1773352704942226,1.2384774630268414,0.0,0.0,-1.6870708465576172,-1.8082186698913574,-1.5639520645141602,1.4805958280501648e-140,0.0,7.409311532974243,7.305345213413238,7.518480849266052,0.0,0.0,9.09638237953186,8.88929088115692,9.306351357698441,0.0,0.0
```

Interpretation:
- `passive_choice_rate` is the share of items where the passive option outranked the active option.
- `mean_passive_minus_active_logprob` is the mean passive-vs-active structural preference score.
- In `comparison.csv` and `stats.csv`, differences are always `condition_b - condition_a`.
- `passive_choice_delta` tests paired shifts in passive choice rates across prime conditions.
- `logprob_delta` tests paired shifts in passive-vs-active preference across prime conditions.
- `sinclair_pe_summary.csv` reports paper-style PE: same-prime minus other-prime for each target form.
- `pe_active_target_logprob_same_minus_other = logP(active|active prime) - logP(active|passive prime)`.
- `pe_passive_target_logprob_same_minus_other = logP(passive|passive prime) - logP(passive|active prime)`.
- `mean_passive_minus_active_logprob_sum` is the summed passive-vs-active preference score.
- `logprob_sum_delta` tests paired shifts in summed passive-vs-active preference across prime conditions.