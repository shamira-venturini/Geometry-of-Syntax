# Demo Prompt Generation Audit

Model metadata:
```json
{
  "model_name": "meta-llama/Llama-3.2-3B-Instruct",
  "prompt_csv": "/content/Geometry-of-Syntax/corpora/transitive/experiment_2_core_demo_prompts_lexically_controlled.csv",
  "max_items": 2080,
  "prompt_columns": [
    "prompt_active",
    "prompt_passive",
    "prompt_no_demo",
    "prompt_filler"
  ],
  "batch_size": 2,
  "max_new_tokens": 24,
  "device": "cuda",
  "torch_dtype": "torch.float16",
  "local_files_only": false,
  "seed": 13,
  "classification_version": "structural_v2"
}
```

Generation quality summary:
```csv
prompt_column,prime_condition,n_items,active_like_rate,passive_like_rate,exact_rate,prefix_rate,structural_rate,congruent_rate
prompt_active,active,2080,0.7850961538461538,0.0610576923076923,0.0,0.0,0.8461538461538461,0.7850961538461538
prompt_filler,filler,2080,0.2326923076923077,0.0259615384615384,0.0,0.0,0.2586538461538462,0.0
prompt_no_demo,no_demo,2080,0.3990384615384615,0.0269230769230769,0.0,0.0,0.4259615384615384,0.0
prompt_passive,passive,2080,0.26875,0.5326923076923077,0.0,0.0,0.8014423076923077,0.5326923076923077
```

Generation class summary:
```csv
prompt_column,prime_condition,generation_class,n_items,total_items,share
prompt_active,active,active_structural,1633,2080,0.7850961538461538
prompt_active,active,other,320,2080,0.1538461538461538
prompt_active,active,passive_structural,127,2080,0.0610576923076923
prompt_filler,filler,active_structural,484,2080,0.2326923076923077
prompt_filler,filler,other,1542,2080,0.7413461538461539
prompt_filler,filler,passive_structural,54,2080,0.0259615384615384
prompt_no_demo,no_demo,active_structural,830,2080,0.3990384615384615
prompt_no_demo,no_demo,other,1194,2080,0.5740384615384615
prompt_no_demo,no_demo,passive_structural,56,2080,0.0269230769230769
prompt_passive,passive,active_structural,559,2080,0.26875
prompt_passive,passive,other,413,2080,0.1985576923076923
prompt_passive,passive,passive_structural,1108,2080,0.5326923076923077
```