import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm

# --- CONFIGURATION ---
INPUT_DIR = "/pca_results/activations"
OUTPUT_DIR = "/Users/shamiraventurini/PycharmProjects/Geometry-of-Syntax/pca_results/null_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STRUCTURES = ["transitive", "dative"]
CONDITIONS = ["CORE", "ANOMALOUS", "jabberwocky"]
N_LAYERS = 37
N_SPLITS = 5
N_SHUFFLES = 20   # number of random permutations per layer/condition
N_RANDOM = 100    # number of random directions per layer/condition


def load_deltas(structure, condition):
    files = [f for f in os.listdir(INPUT_DIR) if structure in f and condition in f]
    if not files:
        return None
    f1 = next((f for f in files if "struct1" in f), None)
    f2 = next((f for f in files if "struct2" in f), None)
    if f1 is None or f2 is None:
        return None
    d1 = np.load(os.path.join(INPUT_DIR, f1))
    d2 = np.load(os.path.join(INPUT_DIR, f2))
    return np.concatenate([d1, d2], axis=0)  # [items, layers, dim]


def center(data_layer):
    return data_layer - data_layer.mean(axis=0, keepdims=True)


def real_pca_cv(data_layer):
    """Your existing real-data KFold PCA, returning mean test variance."""
    centered = center(data_layer)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    vars_test = []

    for train_idx, test_idx in kf.split(centered):
        X_train = centered[train_idx]
        X_test = centered[test_idx]

        pca = PCA(n_components=1)
        pca.fit(X_train)
        pc1 = pca.components_[0]
        proj = X_test @ pc1
        recon = np.outer(proj, pc1)
        vars_test.append(np.var(recon) / np.var(X_test))

    return float(np.mean(vars_test))


def pca_cv_label_shuffled(data_layer, n_shuffles=N_SHUFFLES):
    """
    Null 1: shuffle struct1/struct2 labels across items.
    Operationally, we randomly flip the sign of each delta vector,
    which is equivalent if your delta is (struct2 - struct1).
    """
    centered = center(data_layer)
    n_items = centered.shape[0]
    vars_shuf = []

    rng = np.random.default_rng(123)

    for _ in range(n_shuffles):
        # randomly flip sign per item
        signs = rng.choice([-1.0, 1.0], size=n_items).astype(np.float32)
        shuffled = centered * signs[:, None]

        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        fold_vars = []
        for train_idx, test_idx in kf.split(shuffled):
            X_train = shuffled[train_idx]
            X_test = shuffled[test_idx]

            pca = PCA(n_components=1)
            pca.fit(X_train)
            pc1 = pca.components_[0]
            proj = X_test @ pc1
            recon = np.outer(proj, pc1)
            fold_vars.append(np.var(recon) / np.var(X_test))

        vars_shuf.append(np.mean(fold_vars))

    return float(np.mean(vars_shuf)), float(np.std(vars_shuf))


def random_direction_baseline(data_layer, n_random=N_RANDOM):
    """
    Null 2: how much variance does a random unit vector explain?
    Evaluated on full centered data (no CV needed).
    """
    centered = center(data_layer)
    dim = centered.shape[1]
    rng = np.random.default_rng(456)
    vals = []

    for _ in range(n_random):
        v = rng.normal(size=dim)
        v = v / np.linalg.norm(v)
        proj = centered @ v
        recon = np.outer(proj, v)
        vals.append(np.var(recon) / np.var(centered))

    return float(np.mean(vals)), float(np.std(vals))


# --- MAIN LOOP: run tests and save CSVs ---
rows = []

for struct in STRUCTURES:
    print(f"\n=== Null tests for {struct.upper()} ===")
    for cond in CONDITIONS:
        deltas = load_deltas(struct, cond)
        if deltas is None:
            print(f"  No data for {struct}, {cond}, skipping.")
            continue

        for layer in tqdm(range(N_LAYERS), desc=f"{cond} ({struct})"):
            layer_data = deltas[:, layer, :]  # [items, dim]

            # skip degenerate layers (all zeros)
            if np.allclose(layer_data, 0):
                rows.append({
                    "structure": struct,
                    "condition": cond,
                    "layer": layer,
                    "real_var": 0.0,
                    "shuffle_mean": 0.0,
                    "shuffle_std": 0.0,
                    "rand_mean": 0.0,
                    "rand_std": 0.0,
                })
                continue

            real_var = real_pca_cv(layer_data)
            shuf_mean, shuf_std = pca_cv_label_shuffled(layer_data)
            rand_mean, rand_std = random_direction_baseline(layer_data)

            rows.append({
                "structure": struct,
                "condition": cond,
                "layer": layer,
                "real_var": real_var,
                "shuffle_mean": shuf_mean,
                "shuffle_std": shuf_std,
                "rand_mean": rand_mean,
                "rand_std": rand_std,
            })

# Save combined results
null_df = pd.DataFrame(rows)
out_path = os.path.join(OUTPUT_DIR, "PCA_null_tests_all.csv")
null_df.to_csv(out_path, index=False)
print(f"\nSaved null-test results to {out_path}")
