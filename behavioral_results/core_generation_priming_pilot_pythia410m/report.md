# CORE Generation Priming Pilot

## First-word choice summary

```csv
prime_structure,n_generations,n_active,n_passive,n_other,valid_rate,active_rate_all,passive_rate_all,active_rate_valid,passive_rate_valid
active,90,13,6,71,0.2111111111111111,0.14444444444444443,0.06666666666666667,0.6842105263157895,0.3157894736842105
passive,90,7,4,79,0.12222222222222222,0.07777777777777778,0.044444444444444446,0.6363636363636364,0.36363636363636365
```

## Full-sentence summary

```csv
prime_structure,n_generations,n_active,n_passive,n_other,valid_rate,active_rate_all,passive_rate_all,active_rate_valid,passive_rate_valid
active,90,3,1,86,0.044444444444444446,0.03333333333333333,0.011111111111111112,0.75,0.25
passive,90,2,0,88,0.022222222222222223,0.022222222222222223,0.0,1.0,0.0
```

## Comparison

```json
{
  "passive_rate_shift_all": -0.02222222222222222,
  "active_rate_shift_all": 0.06666666666666665,
  "passive_rate_shift_valid": 0.04784688995215314
}
```

Interpretation:
- `choice_summary.csv` is the primary pilot output. It scores the first generated noun after `Sentence: the` as the structural commitment point.
- `sentence_summary.csv` is a stricter secondary output based on whole-sentence active/passive coding.
- `passive_rate_shift_all` compares passive choice rate after passive primes vs active primes.
- `active_rate_shift_all` compares active choice rate after active primes vs passive primes.
- `*_valid` rates restrict to outputs classified as active or passive.