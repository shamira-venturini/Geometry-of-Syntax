# Transitive Statistical Analysis

## Primary and Secondary Paired Effects

| condition | metric | n_items | mean_diff | bootstrap_ci95_low | bootstrap_ci95_high | perm_p_greater_than_zero | effect_size_dz |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ANOMALOUS | critical_word_pe_mean | 15000 | 0.0482195 | 0.0328669 | 0.0637229 | 9.999e-05 | 0.0497836 |
| CORE | critical_word_pe_mean | 15000 | 0.0453464 | 0.0299659 | 0.0604456 | 9.999e-05 | 0.0475073 |
| jabberwocky | critical_word_pe_mean | 15000 | 0.76346 | 0.748911 | 0.778145 | 9.999e-05 | 0.822278 |
| ANOMALOUS | post_divergence_pe_mean | 15000 | 0.162111 | 0.155705 | 0.168247 | 9.999e-05 | 0.40795 |
| CORE | post_divergence_pe_mean | 15000 | 0.0827745 | 0.0771065 | 0.0884886 | 9.999e-05 | 0.230794 |
| jabberwocky | post_divergence_pe_mean | 15000 | 0.421449 | 0.416432 | 0.426448 | 9.999e-05 | 1.34075 |
| ANOMALOUS | sentence_pe | 15000 | 1.89322 | 1.84572 | 1.94039 | 9.999e-05 | 0.648192 |
| CORE | sentence_pe | 15000 | 1.20775 | 1.16621 | 1.24868 | 9.999e-05 | 0.470829 |
| jabberwocky | sentence_pe | 15000 | 8.99388 | 8.90082 | 9.0852 | 9.999e-05 | 1.54007 |
| ANOMALOUS | sentence_pe_mean | 15000 | 0.249988 | 0.243278 | 0.25662 | 9.999e-05 | 0.594185 |
| CORE | sentence_pe_mean | 15000 | 0.143378 | 0.137463 | 0.149296 | 9.999e-05 | 0.388663 |
| jabberwocky | sentence_pe_mean | 15000 | 0.497472 | 0.492054 | 0.502891 | 9.999e-05 | 1.48187 |

## Confound-Aware Delta Regression (HC3)

| term | Coef. | Std.Err. | z | P>|z| | [0.025 | 0.975] | model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Intercept | 0.188855 | 0.00756919 | 24.9504 | 2.11174e-137 | 0.174019 | 0.20369 | delta_ols_hc3 |
| C(condition)[T.CORE] | -0.106811 | 0.00456783 | -23.3832 | 6.33685e-121 | -0.115763 | -0.0978577 | delta_ols_hc3 |
| C(condition)[T.jabberwocky] | 0.431085 | 0.0208151 | 20.7102 | 2.80556e-95 | 0.390288 | 0.471882 | delta_ols_hc3 |
| z_delta_target_length | 0.000299819 | 0.0019886 | 0.150769 | 0.880158 | -0.00359777 | 0.00419741 | delta_ols_hc3 |
| z_delta_critical_word_token_count | 0.0883889 | 0.00977833 | 9.03927 | 1.57723e-19 | 0.0692238 | 0.107554 | delta_ols_hc3 |

## Per-Condition LMM Robustness

| condition | term | coef | std_err | p_value | aic | bic | converged | n_obs | model | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ANOMALOUS | C(target_structure)[T.passive] | 0.409795 | 0.0301322 | 4.01055e-42 | 5749.13 | 5798.98 | True | 30000 | lmm_random_intercept |  |
| CORE | C(target_structure)[T.passive] | 0.170425 | 0.027047 | 2.95627e-10 | -8492.76 | -8442.9 | True | 30000 | lmm_random_intercept |  |
| jabberwocky | C(target_structure)[T.passive] | 0.649746 | 0.0149174 | 0 | -16314.9 | -16265.1 | True | 30000 | lmm_random_intercept |  |

## Notes

- Primary effect is passive minus active on sentence_pe_mean.
- Permutation p-values are sign-flip tests on paired item-level differences.
- Bootstrap confidence intervals are percentile 95% intervals for mean paired difference.
- Delta regression controls for target length and critical-word token-count asymmetries.