# CORE Generation Priming Pilot

## First-word choice summary

```csv
prime_structure,n_generations,n_active,n_passive,n_other,valid_rate,active_rate_all,passive_rate_all,active_rate_valid,passive_rate_valid
active,90,11,3,76,0.15555555555555556,0.12222222222222222,0.03333333333333333,0.7857142857142857,0.21428571428571427
passive,90,10,2,78,0.13333333333333333,0.1111111111111111,0.022222222222222223,0.8333333333333334,0.16666666666666666
```

## Full-sentence summary

```csv
prime_structure,n_generations,n_active,n_passive,n_other,valid_rate,active_rate_all,passive_rate_all,active_rate_valid,passive_rate_valid
active,90,5,0,85,0.05555555555555555,0.05555555555555555,0.0,1.0,0.0
passive,90,2,0,88,0.022222222222222223,0.022222222222222223,0.0,1.0,0.0
```

## Comparison

```json
{
  "passive_rate_shift_all": -0.01111111111111111,
  "active_rate_shift_all": 0.011111111111111113,
  "passive_rate_shift_valid": -0.047619047619047616
}
```

Interpretation:
- `choice_summary.csv` is the primary pilot output. It scores the first generated noun after `Sentence: the` as the structural commitment point.
- `sentence_summary.csv` is a stricter secondary output based on whole-sentence active/passive coding.
- `passive_rate_shift_all` compares passive choice rate after passive primes vs active primes.
- `active_rate_shift_all` compares active choice rate after active primes vs passive primes.
- `*_valid` rates restrict to outputs classified as active or passive.