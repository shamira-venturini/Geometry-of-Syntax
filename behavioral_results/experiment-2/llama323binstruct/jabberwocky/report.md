# Demo Prompt Generation Audit

Model metadata:
```json
{
  "model_name": "meta-llama/Llama-3.2-3B-Instruct",
  "prompt_csv": "/content/Geometry-of-Syntax/corpora/transitive/experiment_2_jabberwocky_demo_prompts_lexically_controlled.csv",
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
prompt_active,active,2048,0.833984375,0.0,0.36572265625,0.0,0.46826171875,0.833984375
prompt_filler,filler,2048,0.2265625,0.02734375,0.1171875,0.0,0.13671875,0.0
prompt_no_prime,no_prime,2048,0.34765625,0.0166015625,0.13037109375,0.0,0.23388671875,0.0
prompt_passive,passive,2048,0.21142578125,0.63232421875,0.34716796875,0.0,0.49658203125,0.63232421875
```

Generation class summary:
```csv
prompt_column,prime_condition,generation_class,n_items,total_items,share
prompt_active,active,active_exact,749,2048,0.36572265625
prompt_active,active,active_structural,959,2048,0.46826171875
prompt_active,active,other,340,2048,0.166015625
prompt_filler,filler,active_exact,216,2048,0.10546875
prompt_filler,filler,active_structural,248,2048,0.12109375
prompt_filler,filler,other,1528,2048,0.74609375
prompt_filler,filler,passive_exact,24,2048,0.01171875
prompt_filler,filler,passive_structural,32,2048,0.015625
prompt_no_prime,no_prime,active_exact,263,2048,0.12841796875
prompt_no_prime,no_prime,active_structural,449,2048,0.21923828125
prompt_no_prime,no_prime,other,1302,2048,0.6357421875
prompt_no_prime,no_prime,passive_exact,4,2048,0.001953125
prompt_no_prime,no_prime,passive_structural,30,2048,0.0146484375
prompt_passive,passive,active_exact,405,2048,0.19775390625
prompt_passive,passive,active_structural,28,2048,0.013671875
prompt_passive,passive,other,320,2048,0.15625
prompt_passive,passive,passive_exact,306,2048,0.1494140625
prompt_passive,passive,passive_structural,989,2048,0.48291015625
```