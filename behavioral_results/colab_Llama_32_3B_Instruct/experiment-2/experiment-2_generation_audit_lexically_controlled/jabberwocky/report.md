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
prompt_active,active,2048,0.294921875,0.0,0.0,0.0,0.294921875,0.294921875
prompt_filler,filler,2048,0.01953125,0.0,0.0,0.0,0.01953125,0.0
prompt_no_prime,no_prime,2048,0.0,0.0,0.0,0.0,0.0,0.0
prompt_passive,passive,2048,0.0537109375,0.369140625,0.021484375,0.0,0.4013671875,0.369140625
```

Generation class summary:
```csv
prompt_column,prime_condition,generation_class,n_items,total_items,share
prompt_active,active,active_structural,604,2048,0.294921875
prompt_active,active,other,1444,2048,0.705078125
prompt_filler,filler,active_structural,40,2048,0.01953125
prompt_filler,filler,other,2008,2048,0.98046875
prompt_no_prime,no_prime,other,2048,2048,1.0
prompt_passive,passive,active_exact,44,2048,0.021484375
prompt_passive,passive,active_structural,66,2048,0.0322265625
prompt_passive,passive,other,1182,2048,0.5771484375
prompt_passive,passive,passive_structural,756,2048,0.369140625
```