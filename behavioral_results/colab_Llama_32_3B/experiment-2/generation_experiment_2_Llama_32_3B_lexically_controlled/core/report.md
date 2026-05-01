# Demo Prompt Generation Audit

Model metadata:
```json
{
  "model_name": "meta-llama/Llama-3.2-3B",
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
prompt_active,active,2048,0.68994140625,0.0498046875,0.10595703125,0.0,0.6337890625,0.68994140625
prompt_filler,filler,2048,0.13916015625,0.03759765625,0.05029296875,0.0,0.12646484375,0.0
prompt_no_prime,no_prime,2048,0.30908203125,0.00634765625,0.08154296875,0.0,0.23388671875,0.0
prompt_passive,passive,2048,0.216796875,0.68701171875,0.0810546875,0.0,0.82275390625,0.68701171875
```

Generation class summary:
```csv
prompt_column,prime_condition,generation_class,n_items,total_items,share
prompt_active,active,active_exact,208,2048,0.1015625
prompt_active,active,active_structural,1205,2048,0.58837890625
prompt_active,active,other,533,2048,0.26025390625
prompt_active,active,passive_exact,9,2048,0.00439453125
prompt_active,active,passive_structural,93,2048,0.04541015625
prompt_filler,filler,active_exact,97,2048,0.04736328125
prompt_filler,filler,active_structural,188,2048,0.091796875
prompt_filler,filler,other,1686,2048,0.8232421875
prompt_filler,filler,passive_exact,6,2048,0.0029296875
prompt_filler,filler,passive_structural,71,2048,0.03466796875
prompt_no_prime,no_prime,active_exact,164,2048,0.080078125
prompt_no_prime,no_prime,active_structural,469,2048,0.22900390625
prompt_no_prime,no_prime,other,1402,2048,0.6845703125
prompt_no_prime,no_prime,passive_exact,3,2048,0.00146484375
prompt_no_prime,no_prime,passive_structural,10,2048,0.0048828125
prompt_passive,passive,active_exact,152,2048,0.07421875
prompt_passive,passive,active_structural,292,2048,0.142578125
prompt_passive,passive,other,197,2048,0.09619140625
prompt_passive,passive,passive_exact,14,2048,0.0068359375
prompt_passive,passive,passive_structural,1393,2048,0.68017578125
```