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
prompt_active,active,2048,0.830078125,0.0,0.16796875,0.0,0.662109375,0.830078125
prompt_filler,filler,2048,0.19921875,0.0302734375,0.0498046875,0.0,0.1796875,0.0
prompt_no_prime,no_prime,2048,0.35595703125,0.02490234375,0.06298828125,0.0,0.31787109375,0.0
prompt_passive,passive,2048,0.1630859375,0.6923828125,0.15087890625,0.0,0.70458984375,0.6923828125
```

Generation class summary:
```csv
prompt_column,prime_condition,generation_class,n_items,total_items,share
prompt_active,active,active_exact,344,2048,0.16796875
prompt_active,active,active_structural,1356,2048,0.662109375
prompt_active,active,other,348,2048,0.169921875
prompt_filler,filler,active_exact,93,2048,0.04541015625
prompt_filler,filler,active_structural,315,2048,0.15380859375
prompt_filler,filler,other,1578,2048,0.7705078125
prompt_filler,filler,passive_exact,9,2048,0.00439453125
prompt_filler,filler,passive_structural,53,2048,0.02587890625
prompt_no_prime,no_prime,active_exact,127,2048,0.06201171875
prompt_no_prime,no_prime,active_structural,602,2048,0.2939453125
prompt_no_prime,no_prime,other,1268,2048,0.619140625
prompt_no_prime,no_prime,passive_exact,2,2048,0.0009765625
prompt_no_prime,no_prime,passive_structural,49,2048,0.02392578125
prompt_passive,passive,active_exact,150,2048,0.0732421875
prompt_passive,passive,active_structural,184,2048,0.08984375
prompt_passive,passive,other,296,2048,0.14453125
prompt_passive,passive,passive_exact,159,2048,0.07763671875
prompt_passive,passive,passive_structural,1259,2048,0.61474609375
```