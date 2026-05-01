# Experiment 1b: Controlled Processing Structural Priming

## Summary

```csv
prompt_template,prime_condition,n_items,n_active_choice,n_passive_choice,active_choice_rate,passive_choice_rate,mean_passive_minus_active_logprob,sd_passive_minus_active_logprob,mean_passive_minus_active_logprob_sum,sd_passive_minus_active_logprob_sum
processing_teacher_forced,active,2048,15,2033,0.00732421875,0.99267578125,1.2571356232618056,0.4501636695686227,-2.9372135046869516,2.194271259269354
processing_teacher_forced,filler,2048,8,2040,0.00390625,0.99609375,1.304043995589018,0.4907228296820468,-2.813272288069129,2.438000018121553
processing_teacher_forced,no_prime,2048,15,2033,0.00732421875,0.99267578125,1.8245964936852928,0.6624747773923096,-2.2693365961313248,2.95381067102936
processing_teacher_forced,passive,2048,2,2046,0.0009765625,0.9990234375,1.5123416007570332,0.44939815604763095,-0.9727425258606672,2.2290329539895626
```

## Pairwise Condition Comparisons

```csv
prompt_template,condition_a,condition_b,passive_choice_rate_a,passive_choice_rate_b,passive_choice_rate_diff_b_minus_a,active_choice_rate_a,active_choice_rate_b,active_choice_rate_diff_b_minus_a,mean_logprob_a,mean_logprob_b,mean_logprob_diff_b_minus_a,mean_logprob_sum_a,mean_logprob_sum_b,mean_logprob_sum_diff_b_minus_a
processing_teacher_forced,active,filler,0.99267578125,0.99609375,0.00341796875,0.00732421875,0.00390625,-0.00341796875,1.2571356232618056,1.304043995589018,0.046908372327212344,-2.9372135046869516,-2.813272288069129,0.12394121661782265
processing_teacher_forced,active,no_prime,0.99267578125,0.99267578125,0.0,0.00732421875,0.00732421875,0.0,1.2571356232618056,1.8245964936852928,0.5674608704234871,-2.9372135046869516,-2.2693365961313248,0.6678769085556269
processing_teacher_forced,active,passive,0.99267578125,0.9990234375,0.00634765625,0.00732421875,0.0009765625,-0.00634765625,1.2571356232618056,1.5123416007570332,0.25520597749522755,-2.9372135046869516,-0.9727425258606672,1.9644709788262844
processing_teacher_forced,no_prime,filler,0.99267578125,0.99609375,0.00341796875,0.00732421875,0.00390625,-0.00341796875,1.8245964936852928,1.304043995589018,-0.5205524980962748,-2.2693365961313248,-2.813272288069129,-0.5439356919378042
processing_teacher_forced,passive,filler,0.9990234375,0.99609375,-0.0029296875,0.0009765625,0.00390625,0.0029296875,1.5123416007570332,1.304043995589018,-0.2082976051680152,-0.9727425258606672,-2.813272288069129,-1.8405297622084618
processing_teacher_forced,passive,no_prime,0.9990234375,0.99267578125,-0.00634765625,0.0009765625,0.00732421875,0.00634765625,1.5123416007570332,1.8245964936852928,0.3122548929282596,-0.9727425258606672,-2.2693365961313248,-1.2965940702706575
```

## Paired Significance Tests

```csv
prompt_template,metric,condition_a,condition_b,n_items,mean_diff_b_minus_a,sd_diff,effect_size_dz,t_stat,t_p_two_sided,perm_p_two_sided,bootstrap_ci95_low,bootstrap_ci95_high,mcnemar_b,mcnemar_c,mcnemar_p_exact
processing_teacher_forced,logprob_delta,active,filler,2048,0.046908372327212296,0.36507644521692084,0.12848917792913295,5.814756417457585,7.028119660573941e-09,0.0,0.030959503026087833,0.06265825195252776,,,
processing_teacher_forced,logprob_sum_delta,active,filler,2048,0.12394121661782265,2.4465157005496834,0.0506602988854621,2.292623416246243,0.02197070129807047,0.0205,0.016971331182867293,0.23069774387404318,,,
processing_teacher_forced,passive_choice_delta,active,filler,2048,0.00341796875,0.09628181395497887,0.03549962978053398,1.6065298526353593,0.10831177414944453,0.1646,-0.0009765625,0.0078125,13.0,6.0,0.1670684814453125
processing_teacher_forced,logprob_delta,active,no_prime,2048,0.5674608704234873,0.47126481246103746,1.204123149912456,54.49239325995542,0.0,0.0,0.5471485657795214,0.5879427217108212,,,
processing_teacher_forced,logprob_sum_delta,active,no_prime,2048,0.6678769085556269,2.7191778375830076,0.2456172227224689,11.115366640848917,6.529673583599612e-28,0.0,0.5483100500889122,0.7857308766338974,,,
processing_teacher_forced,passive_choice_delta,active,no_prime,2048,0.0,0.1169553503754157,0.0,0.0,1.0,1.0,-0.00537109375,0.0048828125,14.0,14.0,1.0
processing_teacher_forced,logprob_delta,active,passive,2048,0.25520597749522755,0.2269602729240866,1.1244521968855252,50.8868975064234,0.0,0.0,0.24533841755004626,0.2649106924652699,,,
processing_teacher_forced,logprob_sum_delta,active,passive,2048,1.9644709788262844,1.58020533708077,1.243174499369428,56.259655596948086,0.0,0.0,1.895041295979172,2.0332508308812978,,,
processing_teacher_forced,passive_choice_delta,active,passive,2048,0.00634765625,0.08536676374386366,0.074357466203658,3.365034789405191,0.0007794674073812532,0.0008,0.0029296875,0.01025390625,14.0,1.0,0.0009765625
processing_teacher_forced,logprob_delta,no_prime,filler,2048,-0.520552498096275,0.48456270727239065,-1.0742727211230747,-48.616033660790464,0.0,0.0,-0.5417054082614384,-0.4996448344385459,,,
processing_teacher_forced,logprob_sum_delta,no_prime,filler,2048,-0.5439356919378042,2.9229559248665353,-0.1860909661039931,-8.421515779180126,6.890617818758528e-17,0.0,-0.6711719904560596,-0.41813163277693094,,,
processing_teacher_forced,passive_choice_delta,no_prime,filler,2048,0.00341796875,0.10122858918588044,0.033764856128971364,1.528022959013364,0.12666133574258645,0.1893,-0.0009765625,0.0078125,14.0,7.0,0.18924713134765625
processing_teacher_forced,logprob_delta,passive,filler,2048,-0.20829760516801524,0.36646632498102477,-0.5683949410052906,-25.722618699325995,1.1422299236746917e-126,0.0,-0.22420662746395145,-0.19255542466919573,,,
processing_teacher_forced,logprob_sum_delta,passive,filler,2048,-1.8405297622084618,2.5302255463804255,-0.7274172710971963,-32.91914784928261,4.174465205285752e-191,0.0,-1.9504540851339698,-1.7308118461165578,,,
processing_teacher_forced,passive_choice_delta,passive,filler,2048,-0.0029296875,0.06983273293138642,-0.04195292632866797,-1.8985727166477293,0.057761189294668176,0.1155,-0.005859375,0.0,2.0,8.0,0.109375
processing_teacher_forced,logprob_delta,passive,no_prime,2048,0.3122548929282597,0.4745852637864601,0.657953199888564,29.77556283805387,3.5675994977400513e-162,0.0,0.2914226792460781,0.3325281836459088,,,
processing_teacher_forced,logprob_sum_delta,passive,no_prime,2048,-1.2965940702706575,2.8079076915853256,-0.461765204802231,-20.897107688425756,4.676237171844443e-88,0.0,-1.418875922029838,-1.174977542553097,,,
processing_teacher_forced,passive_choice_delta,passive,no_prime,2048,-0.00634765625,0.08536676374386366,-0.074357466203658,-3.365034789405191,0.0007794674073812532,0.0006,-0.01025390625,-0.0029296875,1.0,14.0,0.0009765625
```

## Sinclair-Style PE (Same Minus Other)

```csv
prompt_template,n_items,pe_active_target_logprob_same_minus_other,pe_active_target_logprob_ci95_low,pe_active_target_logprob_ci95_high,pe_active_target_logprob_p,pe_active_target_logprob_perm_p,pe_passive_target_logprob_same_minus_other,pe_passive_target_logprob_ci95_low,pe_passive_target_logprob_ci95_high,pe_passive_target_logprob_p,pe_passive_target_logprob_perm_p,pe_logprob_imbalance_passive_minus_active,pe_logprob_imbalance_ci95_low,pe_logprob_imbalance_ci95_high,pe_logprob_imbalance_p,pe_logprob_imbalance_perm_p,pe_active_target_logprob_sum_same_minus_other,pe_active_target_logprob_sum_ci95_low,pe_active_target_logprob_sum_ci95_high,pe_active_target_logprob_sum_p,pe_active_target_logprob_sum_perm_p,pe_passive_target_logprob_sum_same_minus_other,pe_passive_target_logprob_sum_ci95_low,pe_passive_target_logprob_sum_ci95_high,pe_passive_target_logprob_sum_p,pe_passive_target_logprob_sum_perm_p,pe_logprob_sum_imbalance_passive_minus_active,pe_logprob_sum_imbalance_ci95_low,pe_logprob_sum_imbalance_ci95_high,pe_logprob_sum_imbalance_p,pe_logprob_sum_imbalance_perm_p
processing_teacher_forced,2048,0.0498462337113562,0.03711376075765916,0.06228175094972053,2.229556658649464e-14,0.0,0.20535974378387134,0.19467697954219246,0.21611084307026535,1.4363250944706446e-232,0.0,0.15551351007251513,0.13424771538388633,0.17658464710331626,7.419818023848114e-44,0.0,0.3096785396337509,0.23312697671353816,0.3867015063762665,1.0079850025796603e-14,0.0,1.6547924391925335,1.5695048213005065,1.742011356074363,2.1977286407138893e-232,0.0,1.3451138995587826,1.1952632442116737,1.4985847966745496,2.8668378427019896e-64,0.0
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