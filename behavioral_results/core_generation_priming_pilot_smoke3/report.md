# CORE Generation Priming Pilot

## Summary by prime structure

```csv
prime_structure,n_generations,n_active,n_passive,n_other,valid_rate,active_rate_all,passive_rate_all,active_rate_valid,passive_rate_valid
active,5,0,0,5,0.0,0.0,0.0,,
passive,5,0,0,5,0.0,0.0,0.0,,
```

## Comparison

```json
{
  "passive_rate_shift_all": 0.0,
  "active_rate_shift_all": 0.0,
  "passive_rate_shift_valid": null
}
```

Interpretation:
- `passive_rate_shift_all` compares passive output rate after passive primes vs active primes across all generations.
- `active_rate_shift_all` compares active output rate after active primes vs passive primes across all generations.
- `*_valid` rates restrict to outputs classified as active or passive.