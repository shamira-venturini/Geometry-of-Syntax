# Experiment 1b: Controlled Processing Structural Priming

## Summary

```csv
prompt_template,prime_condition,n_items,n_active_choice,n_passive_choice,active_choice_rate,passive_choice_rate,mean_passive_minus_active_logprob,sd_passive_minus_active_logprob,mean_passive_minus_active_logprob_sum,sd_passive_minus_active_logprob_sum
processing_teacher_forced,active,2048,41,2007,0.02001953125,0.97998046875,1.0119195094479927,0.429845358732641,-3.724152570590377,2.4499021888582493
processing_teacher_forced,filler,2048,32,2016,0.015625,0.984375,1.088631874908294,0.45514752221954774,-3.561844287440181,2.5037527626669607
processing_teacher_forced,no_prime,2048,16,2032,0.0078125,0.9921875,1.7966254545619624,0.6923921622640238,-2.9909764174371958,3.217753919074416
processing_teacher_forced,passive,2048,6,2042,0.0029296875,0.9970703125,1.259803238549521,0.42966379519942594,-1.896702571772039,2.4796468012290203
```

## Pairwise Condition Comparisons

```csv
prompt_template,condition_a,condition_b,passive_choice_rate_a,passive_choice_rate_b,passive_choice_rate_diff_b_minus_a,active_choice_rate_a,active_choice_rate_b,active_choice_rate_diff_b_minus_a,mean_logprob_a,mean_logprob_b,mean_logprob_diff_b_minus_a,mean_logprob_sum_a,mean_logprob_sum_b,mean_logprob_sum_diff_b_minus_a
processing_teacher_forced,active,filler,0.97998046875,0.984375,0.00439453125,0.02001953125,0.015625,-0.00439453125,1.0119195094479927,1.088631874908294,0.07671236546030125,-3.724152570590377,-3.561844287440181,0.16230828315019608
processing_teacher_forced,active,no_prime,0.97998046875,0.9921875,0.01220703125,0.02001953125,0.0078125,-0.01220703125,1.0119195094479927,1.7966254545619624,0.7847059451139697,-3.724152570590377,-2.9909764174371958,0.7331761531531811
processing_teacher_forced,active,passive,0.97998046875,0.9970703125,0.01708984375,0.02001953125,0.0029296875,-0.01708984375,1.0119195094479927,1.259803238549521,0.24788372910152834,-3.724152570590377,-1.896702571772039,1.827449998818338
processing_teacher_forced,no_prime,filler,0.9921875,0.984375,-0.0078125,0.0078125,0.015625,0.0078125,1.7966254545619624,1.088631874908294,-0.7079935796536685,-2.9909764174371958,-3.561844287440181,-0.570867870002985
processing_teacher_forced,passive,filler,0.9970703125,0.984375,-0.0126953125,0.0029296875,0.015625,0.0126953125,1.259803238549521,1.088631874908294,-0.1711713636412271,-1.896702571772039,-3.561844287440181,-1.6651417156681418
processing_teacher_forced,passive,no_prime,0.9970703125,0.9921875,-0.0048828125,0.0029296875,0.0078125,0.0048828125,1.259803238549521,1.7966254545619624,0.5368222160124414,-1.896702571772039,-2.9909764174371958,-1.0942738456651568
```

## Paired Significance Tests

```csv
prompt_template,metric,condition_a,condition_b,n_items,mean_diff_b_minus_a,sd_diff,effect_size_dz,t_stat,t_p_two_sided,perm_p_two_sided,bootstrap_ci95_low,bootstrap_ci95_high,mcnemar_b,mcnemar_c,mcnemar_p_exact
processing_teacher_forced,logprob_delta,active,filler,2048,0.07671236546030122,0.33927829066324905,0.22610455066352048,10.232323906004007,5.316470335897011e-24,0.0,0.0621029432055851,0.09118708352366138,,,
processing_teacher_forced,logprob_sum_delta,active,filler,2048,0.16230828315019608,2.402026005182793,0.06757140963502786,3.0579329259047814,0.0022575457338453874,0.0017,0.057615206111222506,0.26633560797199607,,,
processing_teacher_forced,passive_choice_delta,active,filler,2048,0.00439453125,0.1546549324410915,0.028415073354830695,1.2859194276552937,0.19861662183810913,0.2554,-0.001953125,0.01123046875,29.0,20.0,0.2528697301676033
processing_teacher_forced,logprob_delta,active,no_prime,2048,0.7847059451139695,0.5271630604422649,1.4885450138627663,67.36385749784226,0.0,0.0,0.7625131816348791,0.807559570649598,,,
processing_teacher_forced,logprob_sum_delta,active,no_prime,2048,0.7331761531531811,3.035771110992464,0.24151232960165062,10.929600384095213,4.582713658831461e-27,0.0,0.6016588839236647,0.8642624897882342,,,
processing_teacher_forced,passive_choice_delta,active,no_prime,2048,0.01220703125,0.14099725805860186,0.08657637331448292,3.9179994023173714,9.221728549217175e-05,0.0002,0.00634765625,0.0185546875,33.0,8.0,0.00011222142620681552
processing_teacher_forced,logprob_delta,active,passive,2048,0.24788372910152823,0.2026592075367279,1.2231555235732594,55.35370017052375,0.0,0.0,0.2391424465431492,0.25693158216257067,,,
processing_teacher_forced,logprob_sum_delta,active,passive,2048,1.827449998818338,1.4056985779448996,1.3000297698885273,58.832631426084134,0.0,0.0,1.7667893113801256,1.887497093854472,,,
processing_teacher_forced,passive_choice_delta,active,passive,2048,0.01708984375,0.1333530149980196,0.12815491086012415,5.799629216739284,7.681149713852576e-09,0.0,0.01171875,0.02294921875,36.0,1.0,5.529727786779404e-10
processing_teacher_forced,logprob_delta,no_prime,filler,2048,-0.7079935796536683,0.5475665431621747,-1.2929818092336942,-58.51367713663974,0.0,0.0,-0.7316870913508955,-0.6841257429262624,,,
processing_teacher_forced,logprob_sum_delta,no_prime,filler,2048,-0.570867870002985,3.231162820926686,-0.17667567425130934,-7.995428309363606,2.137891273904297e-15,0.0,-0.7103427063673735,-0.43334850133396685,,,
processing_teacher_forced,passive_choice_delta,no_prime,filler,2048,-0.0078125,0.1323844677018992,-0.05901372068505829,-2.6706561330850267,0.007630235293191679,0.0099,-0.013671875,-0.00244140625,10.0,26.0,0.011330984183587134
processing_teacher_forced,logprob_delta,passive,filler,2048,-0.17117136364122704,0.34312675005435606,-0.49885753184242004,-22.575714791152595,5.901908770081998e-101,0.0,-0.18571479007239555,-0.1564167737672549,,,
processing_teacher_forced,logprob_sum_delta,passive,filler,2048,-1.6651417156681418,2.471227466592368,-0.6738115928940542,-30.493231780959686,1.0947688430878598e-168,0.0,-1.7746265423018486,-1.5564253833843396,,,
processing_teacher_forced,passive_choice_delta,passive,filler,2048,-0.0126953125,0.1282514054389615,-0.09898770665746866,-4.479672232422454,7.887836143358173e-06,0.0001,-0.01806640625,-0.00732421875,4.0,30.0,6.16488978266716e-06
processing_teacher_forced,logprob_delta,passive,no_prime,2048,0.5368222160124412,0.5306309663997912,1.0116677126000695,45.78285439276752,9.8599175e-316,0.0,0.5141845643493377,0.5601998426964403,,,
processing_teacher_forced,logprob_sum_delta,passive,no_prime,2048,-1.0942738456651568,3.0949840151695103,-0.3535636501842237,-16.000464296085305,2.188084937108267e-54,0.0,-1.2276822001906111,-0.9613753109239042,,,
processing_teacher_forced,passive_choice_delta,passive,no_prime,2048,-0.0048828125,0.09872457746628462,-0.04945893540711812,-2.2382559114650022,0.02531150461595415,0.0424,-0.00927734375,-0.0009765625,5.0,15.0,0.04138946533203125
```

## Sinclair-Style PE (Same Minus Other)

```csv
prompt_template,n_items,pe_active_target_logprob_same_minus_other,pe_active_target_logprob_ci95_low,pe_active_target_logprob_ci95_high,pe_active_target_logprob_p,pe_active_target_logprob_perm_p,pe_passive_target_logprob_same_minus_other,pe_passive_target_logprob_ci95_low,pe_passive_target_logprob_ci95_high,pe_passive_target_logprob_p,pe_passive_target_logprob_perm_p,pe_logprob_imbalance_passive_minus_active,pe_logprob_imbalance_ci95_low,pe_logprob_imbalance_ci95_high,pe_logprob_imbalance_p,pe_logprob_imbalance_perm_p,pe_active_target_logprob_sum_same_minus_other,pe_active_target_logprob_sum_ci95_low,pe_active_target_logprob_sum_ci95_high,pe_active_target_logprob_sum_p,pe_active_target_logprob_sum_perm_p,pe_passive_target_logprob_sum_same_minus_other,pe_passive_target_logprob_sum_ci95_low,pe_passive_target_logprob_sum_ci95_high,pe_passive_target_logprob_sum_p,pe_passive_target_logprob_sum_perm_p,pe_logprob_sum_imbalance_passive_minus_active,pe_logprob_sum_imbalance_ci95_low,pe_logprob_sum_imbalance_ci95_high,pe_logprob_sum_imbalance_p,pe_logprob_sum_imbalance_perm_p
processing_teacher_forced,2048,0.08883801758998916,0.07844488633175692,0.09962846201711467,1.0708551203885194e-56,0.0,0.15904571151153907,0.15026311751506807,0.16793657767741632,1.9183455386052755e-213,0.0,0.07020769392154992,0.052445342842533826,0.0876669527933238,5.495009242408239e-15,0.0,0.546206422150135,0.4818274941295385,0.6116705801337957,6.046748585540118e-57,0.0,1.2812435766682029,1.2109457014827059,1.352968282182701,2.5672826691641923e-213,0.0,0.7350371545180678,0.6122856445843354,0.8584331571357324,4.625539061033191e-31,0.0
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