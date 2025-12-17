import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm

# --- CONFIGURATION ---
INPUT_DIR = "/content/Geometry-of-Syntax/activations"
OUTPUT_DIR = "/content/Geometry-of-Syntax/plots/geometryKFOLD"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STRUCTURES = ["transitive", "dative"]
CONDITIONS = ["CORE", "ANOMALOUS_chomsky", "jabberwocky"]
N_LAYERS = 37
N_SPLITS = 5


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


def get_pca_kfold_stats(data_layer, n_splits=N_SPLITS):
    """
    Performs K-Fold CV PCA on one layer.
    Returns:
      - mean explained variance on held-out data
      - std of explained variance
      - unit-norm PC1 vector fitted on full centered data
    """
    # 1. Center data
    mean_vec = np.mean(data_layer, axis=0)
    centered_data = data_layer - mean_vec

    # 2. K-Fold CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    test_variances = []

    for train_idx, test_idx in kf.split(centered_data):
        X_train = centered_data[train_idx]
        X_test = centered_data[test_idx]

        pca = PCA(n_components=1)
        pca.fit(X_train)

        pc1 = pca.components_[0]
        projections = X_test @ pc1
        reconstruction = np.outer(projections, pc1)

        var_test = np.var(reconstruction) / np.var(X_test)
        test_variances.append(var_test)

    # 3. Global PC1 on full data for geometry
    pca_full = PCA(n_components=1)
    pca_full.fit(centered_data)
    global_pc1 = pca_full.components_[0]
    global_pc1 = global_pc1 / np.linalg.norm(global_pc1)

    return np.mean(test_variances), np.std(test_variances), global_pc1


# --- MAIN LOOP ---
for struct in STRUCTURES:
    print(f"\n=== Analyzing {struct.upper()} Geometry (5-Fold CV) ===")

    # Storage for per-layer stats
    layer_stats = {cond: {"mean": [], "std": [], "pc1": []} for cond in CONDITIONS}

    # Load all deltas for this structure
    data_map = {}
    for cond in CONDITIONS:
        data = load_deltas(struct, cond)
        if data is not None:
            data_map[cond] = data

    # Skip if nothing found
    if not data_map:
        print(f"No data for {struct}, skipping.")
        continue

    # Per-layer PCA
    for layer in tqdm(range(N_LAYERS), desc=f"Scanning Layers ({struct})"):
        for cond in CONDITIONS:
            if cond not in data_map:
                continue

            layer_data = data_map[cond][:, layer, :]  # [items, dim]
            mean_var, std_var, pc1 = get_pca_kfold_stats(layer_data)

            layer_stats[cond]["mean"].append(mean_var)
            layer_stats[cond]["std"].append(std_var)
            layer_stats[cond]["pc1"].append(pc1)

    # ---------- SAVE NUMERICAL RESULTS ----------

    # 1) Summary CSV of variance stats
    rows = []
    for cond in CONDITIONS:
        if cond not in layer_stats or not layer_stats[cond]["mean"]:
            continue
        for layer in range(N_LAYERS):
            rows.append({
                "structure": struct,
                "condition": cond,
                "layer": layer,
                "mean_var": layer_stats[cond]["mean"][layer],
                "std_var": layer_stats[cond]["std"][layer],
            })

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(OUTPUT_DIR, f"PCA_summary_{struct}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary stats to {summary_path}")

    # 2) PC1 vectors per condition
    for cond in CONDITIONS:
        if cond not in layer_stats or not layer_stats[cond]["pc1"]:
            continue
        pc1_array = np.stack(layer_stats[cond]["pc1"], axis=0)  # [layers, dim]
        pc1_path = os.path.join(OUTPUT_DIR, f"PC1_{struct}_{cond}.npy")
        np.save(pc1_path, pc1_array)
        print(f"Saved PC1 vectors to {pc1_path}")

    # ---------- PLOTTING ----------

    # Plot A: variance with error bars
    plt.figure(figsize=(10, 5))
    for cond in CONDITIONS:
        if cond not in layer_stats or not layer_stats[cond]["mean"]:
            continue

        label = (
            "Core" if "CORE" in cond
            else "Anomalous" if "ANOMALOUS" in cond
            else "Jabberwocky"
        )

        means = np.array(layer_stats[cond]["mean"])
        stds = np.array(layer_stats[cond]["std"])

        plt.plot(range(N_LAYERS), means, label=label, marker='o', markersize=3)
        plt.fill_between(range(N_LAYERS), means - stds, means + stds, alpha=0.2)

    plt.title(f"{struct.capitalize()}: Syntax Subspace Strength (5-Fold CV)", fontsize=14)
    plt.xlabel("Layer")
    plt.ylabel("Explained Variance (Test Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    var_fig_path = os.path.join(OUTPUT_DIR, f"PCA_Variance_KFold_{struct}.png")
    plt.savefig(var_fig_path, bbox_inches="tight", dpi=200)
    plt.show()
    print(f"Saved variance plot to {var_fig_path}")

    # Plot B: cosine similarity Core vs Jabberwocky
    if "CORE" in layer_stats and "jabberwocky" in layer_stats \
       and layer_stats["CORE"]["pc1"] and layer_stats["jabberwocky"]["pc1"]:

        similarities = []
        for i in range(N_LAYERS):
            v_core = layer_stats["CORE"]["pc1"][i]
            v_jabb = layer_stats["jabberwocky"]["pc1"][i]
            sim = float(np.abs(np.dot(v_core, v_jabb)))
            similarities.append(sim)

        plt.figure(figsize=(10, 5))
        plt.plot(similarities, color='purple', linewidth=2, marker='s')
        plt.title(f"{struct.capitalize()}: Geometric Alignment (Core vs. Jabberwocky)", fontsize=14)
        plt.xlabel("Layer")
        plt.ylabel("Cosine Similarity")
        plt.ylim(0, 1.1)
        plt.axhline(0.8, color='green', linestyle='--', label="High Similarity Threshold")
        plt.grid(True, alpha=0.3)
        plt.legend()
        sim_fig_path = os.path.join(OUTPUT_DIR, f"PCA_Similarity_{struct}.png")
        plt.savefig(sim_fig_path, bbox_inches="tight", dpi=200)
        plt.show()
        print(f"Saved similarity plot to {sim_fig_path}")

        peak_layer = int(np.argmax(similarities))
        print(f"  Peak Alignment Layer: {peak_layer} (Sim: {similarities[peak_layer]:.3f})")
