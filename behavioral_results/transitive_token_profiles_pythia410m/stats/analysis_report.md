# Transitive Statistical Analysis

## Primary and Secondary Paired Effects

| condition | metric | n_items | mean_diff | bootstrap_ci95_low | bootstrap_ci95_high | perm_p_greater_than_zero | effect_size_dz |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ANOMALOUS | critical_word_pe_mean | 15000 | -0.119638 | -0.136828 | -0.103084 | 1 | -0.114057 |
| CORE | critical_word_pe_mean | 15000 | -0.192962 | -0.209213 | -0.176476 | 1 | -0.184631 |
| jabberwocky-only | critical_word_pe_mean | 15000 | 0.496374 | 0.48495 | 0.507581 | 9.999e-05 | 0.704505 |
| ANOMALOUS | post_divergence_pe_mean | 15000 | 0.00928263 | 0.00290971 | 0.015705 | 0.00229977 | 0.0233779 |
| CORE | post_divergence_pe_mean | 15000 | 0.0231065 | 0.0168901 | 0.0293977 | 9.999e-05 | 0.0584065 |
| jabberwocky-only | post_divergence_pe_mean | 15000 | 0.309341 | 0.304316 | 0.314324 | 9.999e-05 | 0.977013 |
| ANOMALOUS | sentence_pe | 15000 | 0.782961 | 0.732076 | 0.832743 | 9.999e-05 | 0.247703 |
| CORE | sentence_pe | 15000 | 1.26707 | 1.21995 | 1.31468 | 9.999e-05 | 0.426726 |
| jabberwocky-only | sentence_pe | 15000 | 7.07126 | 6.96659 | 7.17504 | 9.999e-05 | 1.08243 |
| ANOMALOUS | sentence_pe_mean | 15000 | 0.0936275 | 0.0864388 | 0.100849 | 9.999e-05 | 0.205551 |
| CORE | sentence_pe_mean | 15000 | 0.156111 | 0.149164 | 0.162812 | 9.999e-05 | 0.364604 |
| jabberwocky-only | sentence_pe_mean | 15000 | 0.396388 | 0.390247 | 0.402498 | 9.999e-05 | 1.04487 |

## Confound-Aware Delta Regression (HC3)

| term | Coef. | Std.Err. | z | P>|z| | [0.025 | 0.975] | model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Intercept | 0.0970649 | 0.00713441 | 13.6052 | 3.73096e-42 | 0.0830817 | 0.111048 | delta_ols_hc3 |
| C(condition)[T.CORE] | 0.0624932 | 0.00509976 | 12.2541 | 1.59618e-34 | 0.0524979 | 0.0724886 | delta_ols_hc3 |
| C(condition)[T.jabberwocky-only] | 0.292438 | 0.0190024 | 15.3895 | 1.92533e-53 | 0.255194 | 0.329682 | delta_ols_hc3 |
| z_delta_target_length | 0.0194451 | 0.00217913 | 8.9233 | 4.52608e-19 | 0.015174 | 0.0237161 | delta_ols_hc3 |
| z_delta_critical_word_token_count | -0.00327902 | 0.0088384 | -0.370997 | 0.71064 | -0.020602 | 0.0140439 | delta_ols_hc3 |

## Per-Condition LMM Robustness

| condition | term | coef | std_err | p_value | aic | bic | converged | n_obs | model | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ANOMALOUS | C(target_structure)[T.passive] | 0.0404689 | 0.0283303 | 0.153158 | 2105.19 | 2155.04 | True | 30000 | lmm_random_intercept |  |
| CORE | C(target_structure)[T.passive] | -0.0587749 | 0.0341037 | 0.0848137 | 5551.74 | 5601.6 | True | 30000 | lmm_random_intercept |  |
| jabberwocky-only | C(target_structure)[T.passive] | 0.222411 | 0.0133161 | 1.25949e-62 | -5480.89 | -5431.04 | True | 30000 | lmm_random_intercept |  |

## Notes

- Primary effect is passive minus active on sentence_pe_mean.
- Permutation p-values are sign-flip tests on paired item-level differences.
- Bootstrap confidence intervals are percentile 95% intervals for mean paired difference.
- Delta regression controls for target length and critical-word token-count asymmetries.