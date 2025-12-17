import pandas as pd
import numpy as np

NULL_PATH = "/Users/shamiraventurini/PycharmProjects/Geometry-of-Syntax/pca_results/PCA_null_tests_all_torch.csv"

df_null = pd.read_csv(NULL_PATH)

def summarize_condition(cond, layer_min=1, layer_max=5):
    sub = df_null[(df_null['structure'] == 'transitive') &
                  (df_null['condition'] == cond)].copy()

    sub['z_rand'] = (sub['real_var'] - sub['rand_mean']) / sub['rand_std'].replace(0, np.nan)

    print(f"\n=== {cond} (transitive): per-layer snapshot (first 10 layers) ===")
    print(sub[['layer','real_var','rand_mean','rand_std','z_rand']].head(10))

    early = sub[(sub['layer'] >= layer_min) & (sub['layer'] <= layer_max)]
    summary = {
        'layers': f"{layer_min}–{layer_max}",
        'real_var_mean': early['real_var'].mean(),
        'real_var_min':  early['real_var'].min(),
        'real_var_max':  early['real_var'].max(),
        'rand_mean_mean': early['rand_mean'].mean(),
        'rand_mean_max':  early['rand_mean'].max(),
        'z_rand_min':  early['z_rand'].min(),
        'z_rand_mean': early['z_rand'].mean()
    }
    print(f"\n=== Early-layer summary ({cond}, transitive) ===")
    for k,v in summary.items():
        print(f"{k}: {v}")

    top_layers = sub.sort_values('real_var', ascending=False).head(5)
    print(f"\n=== Top-5 layers by real_var ({cond}, transitive) ===")
    print(top_layers[['layer','real_var','rand_mean','z_rand']])

# Run for Core and Jabberwocky
summarize_condition('CORE')
summarize_condition('jabberwocky')
