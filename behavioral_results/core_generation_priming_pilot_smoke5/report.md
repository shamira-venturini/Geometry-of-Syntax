# CORE Generation Priming Pilot

## First-word choice summary

```csv
prime_structure,n_generations,n_active,n_passive,n_other,valid_rate,active_rate_all,passive_rate_all,active_rate_valid,passive_rate_valid
active,5,0,1,4,0.2,0.0,0.2,0.0,1.0
passive,5,0,0,5,0.0,0.0,0.0,,
```

## Full-sentence summary

```csv
prime_structure,n_generations,n_active,n_passive,n_other,valid_rate,active_rate_all,passive_rate_all,active_rate_valid,passive_rate_valid
active,5,0,0,5,0.0,0.0,0.0,,
passive,5,0,0,5,0.0,0.0,0.0,,
```

## Comparison

```json
{
  "passive_rate_shift_all": -0.2,
  "active_rate_shift_all": 0.0,
  "passive_rate_shift_valid": null
}
```

Interpretation:
- `choice_summary.csv` is the primary pilot output. It scores the first generated noun after `Sentence: the` as the structural commitment point.
- `sentence_summary.csv` is a stricter secondary output based on whole-sentence active/passive coding.
- `passive_rate_shift_all` compares passive choice rate after passive primes vs active primes.
- `active_rate_shift_all` compares active choice rate after active primes vs passive primes.
- `*_valid` rates restrict to outputs classified as active or passive.