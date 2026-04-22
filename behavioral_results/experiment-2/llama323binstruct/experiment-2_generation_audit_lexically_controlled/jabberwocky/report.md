# Demo Prompt Generation Audit

Model metadata:
```json
{
  "model_name": "meta-llama/Llama-3.2-3B-Instruct",
  "prompt_csv": "/content/Geometry-of-Syntax/corpora/transitive/experiment_2_jabberwocky_demo_prompts_lexically_controlled.csv",
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
prompt_active,active,2080,0.8514423076923077,0.0072115384615384,0.0,0.0,0.8586538461538461,0.8514423076923077
prompt_filler,filler,2080,0.3283653846153846,0.0264423076923076,0.0,0.0,0.3548076923076923,0.0
prompt_no_demo,no_demo,2080,0.3995192307692308,0.0269230769230769,0.0,0.0,0.4264423076923077,0.0
prompt_passive,passive,2080,0.0865384615384615,0.7567307692307692,0.0,0.0,0.8432692307692308,0.7567307692307692
```

Generation class summary:
```csv
prompt_column,prime_condition,generation_class,n_items,total_items,share
prompt_active,active,active_structural,1771,2080,0.8514423076923077
prompt_active,active,other,294,2080,0.1413461538461538
prompt_active,active,passive_structural,15,2080,0.0072115384615384
prompt_filler,filler,active_structural,683,2080,0.3283653846153846
prompt_filler,filler,other,1342,2080,0.6451923076923077
prompt_filler,filler,passive_structural,55,2080,0.0264423076923076
prompt_no_demo,no_demo,active_structural,831,2080,0.3995192307692308
prompt_no_demo,no_demo,other,1193,2080,0.5735576923076923
prompt_no_demo,no_demo,passive_structural,56,2080,0.0269230769230769
prompt_passive,passive,active_structural,180,2080,0.0865384615384615
prompt_passive,passive,other,326,2080,0.1567307692307692
prompt_passive,passive,passive_structural,1574,2080,0.7567307692307692
```