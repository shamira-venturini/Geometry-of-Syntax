# Experiment 1b: Controlled Processing Structural Priming

## Summary

```csv
prompt_template,prime_condition,n_items,n_active_choice,n_passive_choice,active_choice_rate,passive_choice_rate,mean_passive_minus_active_logprob,sd_passive_minus_active_logprob,mean_passive_minus_active_logprob_sum,sd_passive_minus_active_logprob_sum
processing_teacher_forced,active,2048,51,1997,0.02490234375,0.97509765625,1.0219482382138572,0.5416877037086637,-6.312009334564209,3.2592922980757946
processing_teacher_forced,filler,2048,121,1927,0.05908203125,0.94091796875,0.7236578539013863,0.4266791859074403,-7.35636967420578,2.8288507948030555
processing_teacher_forced,no_prime,2048,0,2048,0.0,1.0,1.8416834831237794,0.5781683627128481,-5.471324443817139,3.4454119551044964
processing_teacher_forced,passive,2048,0,2048,0.0,1.0,1.6781761894623437,0.32462019240828566,-0.1883608102798462,2.4922000766904735
```

## Pairwise Condition Comparisons

```csv
prompt_template,condition_a,condition_b,passive_choice_rate_a,passive_choice_rate_b,passive_choice_rate_diff_b_minus_a,active_choice_rate_a,active_choice_rate_b,active_choice_rate_diff_b_minus_a,mean_logprob_a,mean_logprob_b,mean_logprob_diff_b_minus_a,mean_logprob_sum_a,mean_logprob_sum_b,mean_logprob_sum_diff_b_minus_a
processing_teacher_forced,active,filler,0.97509765625,0.94091796875,-0.0341796875,0.02490234375,0.05908203125,0.0341796875,1.0219482382138572,0.7236578539013863,-0.2982903843124709,-6.312009334564209,-7.35636967420578,-1.044360339641571
processing_teacher_forced,active,no_prime,0.97509765625,1.0,0.02490234375,0.02490234375,0.0,-0.02490234375,1.0219482382138572,1.8416834831237794,0.8197352449099222,-6.312009334564209,-5.471324443817139,0.8406848907470703
processing_teacher_forced,active,passive,0.97509765625,1.0,0.02490234375,0.02490234375,0.0,-0.02490234375,1.0219482382138572,1.6781761894623437,0.6562279512484865,-6.312009334564209,-0.1883608102798462,6.123648524284363
processing_teacher_forced,no_prime,filler,1.0,0.94091796875,-0.05908203125,0.0,0.05908203125,0.05908203125,1.8416834831237794,0.7236578539013863,-1.1180256292223931,-5.471324443817139,-7.35636967420578,-1.8850452303886414
processing_teacher_forced,passive,filler,1.0,0.94091796875,-0.05908203125,0.0,0.05908203125,0.05908203125,1.6781761894623437,0.7236578539013863,-0.9545183355609574,-0.1883608102798462,-7.35636967420578,-7.168008863925934
processing_teacher_forced,passive,no_prime,1.0,1.0,0.0,0.0,0.0,0.0,1.6781761894623437,1.8416834831237794,0.16350729366143568,-0.1883608102798462,-5.471324443817139,-5.2829636335372925
```

## Paired Significance Tests

```csv
prompt_template,metric,condition_a,condition_b,n_items,mean_diff_b_minus_a,sd_diff,effect_size_dz,t_stat,t_p_two_sided,perm_p_two_sided,bootstrap_ci95_low,bootstrap_ci95_high,mcnemar_b,mcnemar_c,mcnemar_p_exact
processing_teacher_forced,logprob_delta,active,filler,2048,-0.2982903843124708,0.6010901146308736,-0.4962490266466109,-22.457667321538707,4.989094553815308e-100,0.0,-0.32437366998444,-0.27243971476952245,,,
processing_teacher_forced,logprob_sum_delta,active,filler,2048,-1.044360339641571,3.1305813445695616,-0.3335994899008654,-15.096989536593606,6.584060612501769e-49,0.0,-1.1846459373831748,-0.9082377299666404,,,
processing_teacher_forced,passive_choice_delta,active,filler,2048,-0.0341796875,0.26853146185568416,-0.12728373526067147,-5.76020430960474,9.672981334319289e-09,0.0,-0.0458984375,-0.0224609375,40.0,110.0,9.56282907771787e-09
processing_teacher_forced,logprob_delta,active,no_prime,2048,0.8197352449099222,0.6686084459075166,1.2260318425940282,55.48386751032801,0.0,0.0,0.7913540743788084,0.8486861949307579,,,
processing_teacher_forced,logprob_sum_delta,active,no_prime,2048,0.8406848907470703,4.048373000122562,0.2076599391216222,9.39761607255582,1.438886041275904e-20,0.0,0.666600227355957,1.0147755861282348,,,
processing_teacher_forced,passive_choice_delta,active,no_prime,2048,0.02490234375,0.1558655810921668,0.1597680743593718,7.230277682984215,6.776646962481733e-13,0.0,0.0185546875,0.03173828125,51.0,0.0,8.881784197001252e-16
processing_teacher_forced,logprob_delta,active,passive,2048,0.6562279512484868,0.5505899510135146,1.1918632914395113,53.9375753999486,0.0,0.0,0.6321046061813831,0.6802718248218297,,,
processing_teacher_forced,logprob_sum_delta,active,passive,2048,6.123648524284363,3.2861694413705984,1.8634609789720114,84.33061726128841,0.0,0.0,5.983110344409942,6.266761124134064,,,
processing_teacher_forced,passive_choice_delta,active,passive,2048,0.02490234375,0.1558655810921668,0.1597680743593718,7.230277682984215,6.776646962481733e-13,0.0,0.0185546875,0.03173828125,51.0,0.0,8.881784197001252e-16
processing_teacher_forced,logprob_delta,no_prime,filler,2048,-1.1180256292223931,0.6614276622362051,-1.6903218493198287,-76.49523469067753,0.0,0.0,-1.1468793942938957,-1.0890286972834953,,,
processing_teacher_forced,logprob_sum_delta,no_prime,filler,2048,-1.8850452303886414,3.7853777876731014,-0.49798073960469663,-22.536035703985497,1.210369323304443e-100,0.0,-2.0488524720072747,-1.7206490516662598,,,
processing_teacher_forced,passive_choice_delta,no_prime,filler,2048,-0.05908203125,0.23583575281369604,-0.2505219439593336,-11.337328986219585,6.130794920971272e-29,0.0,-0.0693359375,-0.04931640625,0.0,121.0,7.52316384526264e-37
processing_teacher_forced,logprob_delta,passive,filler,2048,-0.9545183355609576,0.39927435319533167,-2.390632726400013,-108.18768717849176,0.0,0.0,-0.9716176261504491,-0.936895806901157,,,
processing_teacher_forced,logprob_sum_delta,passive,filler,2048,-7.168008863925934,2.8069977672922435,-2.553621149061518,-115.5637011892981,0.0,0.0,-7.288725194334984,-7.0445645794272425,,,
processing_teacher_forced,passive_choice_delta,passive,filler,2048,-0.05908203125,0.23583575281369604,-0.2505219439593336,-11.337328986219585,6.130794920971272e-29,0.0,-0.06982421875,-0.04931640625,0.0,121.0,7.52316384526264e-37
processing_teacher_forced,logprob_delta,passive,no_prime,2048,0.16350729366143552,0.6484005979267938,0.25217017717786855,11.411919506910978,2.743552788108745e-29,0.0,0.13569498490364784,0.1913104797119187,,,
processing_teacher_forced,logprob_sum_delta,passive,no_prime,2048,-5.2829636335372925,4.002023339742824,-1.3200731692576244,-59.73969213724693,0.0,0.0,-5.457817921042443,-5.1118745565414425,,,
processing_teacher_forced,passive_choice_delta,passive,no_prime,2048,0.0,0.0,0.0,,,1.0,0.0,0.0,0.0,0.0,1.0
```

## Sinclair-Style PE (Same Minus Other)

```csv
prompt_template,n_items,pe_active_target_logprob_same_minus_other,pe_active_target_logprob_ci95_low,pe_active_target_logprob_ci95_high,pe_active_target_logprob_p,pe_active_target_logprob_perm_p,pe_passive_target_logprob_same_minus_other,pe_passive_target_logprob_ci95_low,pe_passive_target_logprob_ci95_high,pe_passive_target_logprob_p,pe_passive_target_logprob_perm_p,pe_logprob_imbalance_passive_minus_active,pe_logprob_imbalance_ci95_low,pe_logprob_imbalance_ci95_high,pe_logprob_imbalance_p,pe_logprob_imbalance_perm_p,pe_active_target_logprob_sum_same_minus_other,pe_active_target_logprob_sum_ci95_low,pe_active_target_logprob_sum_ci95_high,pe_active_target_logprob_sum_p,pe_active_target_logprob_sum_perm_p,pe_passive_target_logprob_sum_same_minus_other,pe_passive_target_logprob_sum_ci95_low,pe_passive_target_logprob_sum_ci95_high,pe_passive_target_logprob_sum_p,pe_passive_target_logprob_sum_perm_p,pe_logprob_sum_imbalance_passive_minus_active,pe_logprob_sum_imbalance_ci95_low,pe_logprob_sum_imbalance_ci95_high,pe_logprob_sum_imbalance_p,pe_logprob_sum_imbalance_perm_p
processing_teacher_forced,2048,-0.4369124571482341,-0.47055440346399946,-0.40325663089752195,2.1164911915208785e-125,0.0,1.0931404083967209,1.0729148793965577,1.1130271892994643,0.0,0.0,1.530052865544955,1.480262857923905,1.581073210015893,0.0,0.0,-2.6214747428894043,-2.8197373270988466,-2.413338851928711,2.1164911915208783e-125,0.0,8.745123267173767,8.583771646022797,8.906582543253899,0.0,0.0,11.366598010063171,11.0319654494524,11.699964073300361,0.0,0.0
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