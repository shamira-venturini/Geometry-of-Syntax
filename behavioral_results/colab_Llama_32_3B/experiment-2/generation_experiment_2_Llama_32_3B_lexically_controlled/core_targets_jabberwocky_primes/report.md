# Demo Prompt Generation Audit

Model metadata:
```json
{
  "model_name": "meta-llama/Llama-3.2-3B",
  "prompt_csv": "/content/Geometry-of-Syntax/corpora/transitive/experiment_2_core_targets_jabberwocky_primes_demo_prompts_lexically_controlled.csv",
  "max_items": 2048,
  "prompt_columns": [
    "prompt_active",
    "prompt_passive",
    "prompt_no_prime",
    "prompt_filler"
  ],
  "batch_size": 4,
  "max_new_tokens": 24,
  "device": "cuda",
  "torch_dtype": "torch.float16",
  "local_files_only": false,
  "seed": 13
}
```

Generation quality summary:
```csv
prompt_column,prime_condition,n_items,active_like_rate,passive_like_rate,exact_rate,prefix_rate,structural_rate,congruent_rate
prompt_active,active,2048,0.435546875,0.0732421875,0.10400390625,0.0,0.40478515625,0.435546875
prompt_filler,filler,2048,0.1474609375,0.046875,0.04296875,0.0,0.1513671875,0.0
prompt_no_prime,no_prime,2048,0.30859375,0.0068359375,0.08154296875,0.0,0.23388671875,0.0
prompt_passive,passive,2048,0.107421875,0.23046875,0.0341796875,0.0,0.3037109375,0.23046875
```

Generation class summary:
```csv
prompt_column,prime_condition,generation_class,n_items,total_items,share
prompt_active,active,active_exact,166,2048,0.0810546875
prompt_active,active,active_structural,726,2048,0.3544921875
prompt_active,active,other,1006,2048,0.4912109375
prompt_active,active,passive_exact,47,2048,0.02294921875
prompt_active,active,passive_structural,103,2048,0.05029296875
prompt_filler,filler,active_exact,82,2048,0.0400390625
prompt_filler,filler,active_structural,220,2048,0.107421875
prompt_filler,filler,other,1650,2048,0.8056640625
prompt_filler,filler,passive_exact,6,2048,0.0029296875
prompt_filler,filler,passive_structural,90,2048,0.0439453125
prompt_no_prime,no_prime,active_exact,164,2048,0.080078125
prompt_no_prime,no_prime,active_structural,468,2048,0.228515625
prompt_no_prime,no_prime,other,1402,2048,0.6845703125
prompt_no_prime,no_prime,passive_exact,3,2048,0.00146484375
prompt_no_prime,no_prime,passive_structural,11,2048,0.00537109375
prompt_passive,passive,active_exact,42,2048,0.0205078125
prompt_passive,passive,active_structural,178,2048,0.0869140625
prompt_passive,passive,other,1356,2048,0.662109375
prompt_passive,passive,passive_exact,28,2048,0.013671875
prompt_passive,passive,passive_structural,444,2048,0.216796875
```