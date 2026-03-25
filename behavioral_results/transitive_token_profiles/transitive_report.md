# Transitive Priming Report

Active condition set: CORE and jabberwocky.

## Item Summary

| condition   | target_structure   |   n_items |   mean_sentence_pe |   sd_sentence_pe |   mean_sentence_pe_mean |   sd_sentence_pe_mean |   mean_critical_word_pe |   sd_critical_word_pe |   mean_critical_word_pe_mean |   sd_critical_word_pe_mean |   mean_post_divergence_pe |   sd_post_divergence_pe |   mean_post_divergence_pe_mean |   sd_post_divergence_pe_mean |   mean_pre_divergence_pe |   sd_pre_divergence_pe |   mean_structure_region_pe |   sd_structure_region_pe |   mean_structure_region_pe_mean |   sd_structure_region_pe_mean |
|:------------|:-------------------|----------:|-------------------:|-----------------:|------------------------:|----------------------:|------------------------:|----------------------:|-----------------------------:|---------------------------:|--------------------------:|------------------------:|-------------------------------:|-----------------------------:|-------------------------:|-----------------------:|---------------------------:|-------------------------:|--------------------------------:|------------------------------:|
| CORE        | active             |     15000 |            0.19073 |          1.38465 |               0.0311782 |              0.227847 |                0.143683 |              0.689171 |                    0.130402  |                   0.658222 |                  0.411241 |                 1.21702 |                      0.0812686 |                     0.233573 |                -0.220511 |               0.541786 |                   0.473601 |                  1.06174 |                       0.115893  |                      0.260831 |
| CORE        | passive            |     15000 |            1.39848 |          1.51377 |               0.174557  |              0.188898 |                0.175748 |              0.516494 |                    0.175748  |                   0.516494 |                  1.17797  |                 1.35661 |                      0.164043  |                     0.188479 |                 0.220511 |               0.541786 |                   1.14371  |                  1.22221 |                       0.190281  |                      0.203356 |
| jabberwocky | active             |     15000 |           -1.85624 |          3.3904  |              -0.112068  |              0.204496 |                0.219599 |              1.31521  |                    0.0447444 |                   0.280831 |                 -0.968787 |                 2.95283 |                     -0.0624946 |                     0.190638 |                -0.887457 |               0.839071 |                  -0.71322  |                  2.36319 |                      -0.0642733 |                      0.2119   |
| jabberwocky | passive            |     15000 |            7.13764 |          2.99097 |               0.385404  |              0.162617 |                0.808204 |              0.865784 |                    0.808204  |                   0.865784 |                  6.25018  |                 2.81057 |                      0.358954  |                     0.162275 |                 0.887457 |               0.839071 |                   5.99377  |                  2.45083 |                       0.45852   |                      0.189842 |

## Paired Effects

|   n_items |   mean_diff |   sd_diff |   effect_size_dz |    t_stat |   t_p_two_sided |   perm_p_two_sided |   perm_p_greater_than_zero |   bootstrap_ci95_low |   bootstrap_ci95_high | condition   | metric                  |
|----------:|------------:|----------:|-----------------:|----------:|----------------:|-------------------:|---------------------------:|---------------------:|----------------------:|:------------|:------------------------|
|     15000 |   0.0453464 |  0.954516 |        0.0475073 |   5.81843 |    6.06154e-09  |          9.999e-05 |                  9.999e-05 |            0.0299659 |             0.0604456 | CORE        | critical_word_pe_mean   |
|     15000 |   0.76346   |  0.928469 |        0.822278  | 100.708   |    0            |          9.999e-05 |                  9.999e-05 |            0.748911  |             0.778145  | jabberwocky | critical_word_pe_mean   |
|     15000 |   0.0827745 |  0.358651 |        0.230794  |  28.2663  |    2.67242e-171 |          9.999e-05 |                  9.999e-05 |            0.0771065 |             0.0884886 | CORE        | post_divergence_pe_mean |
|     15000 |   0.421449  |  0.314338 |        1.34075   | 164.208   |    0            |          9.999e-05 |                  9.999e-05 |            0.416432  |             0.426448  | jabberwocky | post_divergence_pe_mean |
|     15000 |   1.20775   |  2.56515  |        0.470829  |  57.6645  |    0            |          9.999e-05 |                  9.999e-05 |            1.16621   |             1.24868   | CORE        | sentence_pe             |
|     15000 |   8.99388   |  5.83993  |        1.54007   | 188.619   |    0            |          9.999e-05 |                  9.999e-05 |            8.90082   |             9.0852    | jabberwocky | sentence_pe             |
|     15000 |   0.143378  |  0.368901 |        0.388663  |  47.6014  |    0            |          9.999e-05 |                  9.999e-05 |            0.137463  |             0.149296  | CORE        | sentence_pe_mean        |
|     15000 |   0.497472  |  0.335705 |        1.48187   | 181.491   |    0            |          9.999e-05 |                  9.999e-05 |            0.492054  |             0.502891  | jabberwocky | sentence_pe_mean        |

## LMM Coefficients

| condition   | term                           |     coef |   std_err |     p_value |       aic |      bic | converged   |   n_obs | model                |   error |
|:------------|:-------------------------------|---------:|----------:|------------:|----------:|---------:|:------------|--------:|:---------------------|--------:|
| CORE        | C(target_structure)[T.passive] | 0.170425 | 0.027047  | 2.95627e-10 |  -8492.76 |  -8442.9 | True        |   30000 | lmm_random_intercept |     nan |
| jabberwocky | C(target_structure)[T.passive] | 0.649746 | 0.0149174 | 0           | -16314.9  | -16265.1 | True        |   30000 | lmm_random_intercept |     nan |

The archived ANOMALOUS branch and the original full-condition outputs were moved to:
- /Users/shamiraventurini/PycharmProjects/Geometry-of-Syntax/_archive/anomalous_retired_2026-03-25
