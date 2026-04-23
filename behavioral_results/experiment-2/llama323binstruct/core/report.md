# Demo Prompt Generation Audit

Model metadata:
```json
{
  "model_name": "meta-llama/Llama-3.2-3B-Instruct",
  "prompt_csv": "/content/Geometry-of-Syntax/corpora/transitive/experiment_2_core_demo_prompts_lexically_controlled.csv",
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
prompt_active,active,2048,0.8330078125,0.0009765625,0.35107421875,0.0,0.48291015625,0.8330078125
prompt_filler,filler,2048,0.1875,0.02978515625,0.099609375,0.0,0.11767578125,0.0
prompt_no_prime,no_prime,2048,0.349609375,0.0166015625,0.1298828125,0.0,0.236328125,0.0
prompt_passive,passive,2048,0.18359375,0.67529296875,0.3173828125,0.0,0.54150390625,0.67529296875
```

Generation class summary:
```csv
prompt_column,prime_condition,generation_class,n_items,total_items,share
prompt_active,active,active_exact,717,2048,0.35009765625
prompt_active,active,active_structural,989,2048,0.48291015625
prompt_active,active,other,340,2048,0.166015625
prompt_active,active,passive_exact,2,2048,0.0009765625
prompt_filler,filler,active_exact,177,2048,0.08642578125
prompt_filler,filler,active_structural,207,2048,0.10107421875
prompt_filler,filler,other,1603,2048,0.78271484375
prompt_filler,filler,passive_exact,27,2048,0.01318359375
prompt_filler,filler,passive_structural,34,2048,0.0166015625
prompt_no_prime,no_prime,active_exact,262,2048,0.1279296875
prompt_no_prime,no_prime,active_structural,454,2048,0.2216796875
prompt_no_prime,no_prime,other,1298,2048,0.6337890625
prompt_no_prime,no_prime,passive_exact,4,2048,0.001953125
prompt_no_prime,no_prime,passive_structural,30,2048,0.0146484375
prompt_passive,passive,active_exact,341,2048,0.16650390625
prompt_passive,passive,active_structural,35,2048,0.01708984375
prompt_passive,passive,other,289,2048,0.14111328125
prompt_passive,passive,passive_exact,309,2048,0.15087890625
prompt_passive,passive,passive_structural,1074,2048,0.5244140625
```