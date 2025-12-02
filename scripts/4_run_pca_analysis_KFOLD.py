import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm
import itertools

# --- CONFIGURATION ---
INPUT_DIR = "activations"
OUTPUT_DIR = "plots/geometry"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STRUCTURES = ["transitive", "dative"]
CONDITIONS = ["CORE", "ANOMALOUS_chomsky", "jabberwocky"]


def load_deltas(structure, condition):
    files = [f for f in os.listdir(INPUT_DIR) if structure in f and condition in f]
    if not files: return None
    f1 = next((f for f in files if "struct1" in f), None)
    f2 = next((f for f in files if "struct2" in f), None)
    if f1 is None or f2 is None: return None
    d1 = np.load(os.path.join(INPUT_DIR, f1))
    d2 = np.load(os.path.join(INPUT_DIR, f2))
    return np.concatenate([d1, d2], axis=0)


def analyze_layer_stability(data_layer, n_splits=5):
    """
    Calculates Variance Explained AND Cross-Fold Directional Stability.
    """
    # Center
    mean_vec = np.mean(data_layer, axis=0)
    centered_data = data_layer - mean_vec

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_pc1s = []
    test_variances = []

    # 1. Run CV
    for train_idx, test_idx in kf.split(centered_data):
        X_train = centered_data[train_idx]
        X_test = centered_data[test_idx]

        pca = PCA(n_components=1)
        pca.fit(X_train)

        # Store PC1 (Direction)
        fold_pc1s.append(pca.components_[0])

        # Store Variance (Strength)
        pc1 = pca.components_[0]
        projections = X_test @ pc1
        reconstruction = np.outer(projections, pc1)
        var_test = np.var(reconstruction) / np.var(X_test)
        test_variances.append(var_test)

    # 2. Calculate Stability (Cosine Sim between all pairs of folds)
    # If PC1 is stable, all folds should point the same way (or 180 flip)
    similarities = []
    for v1, v2 in itertools.combinations(fold_pc1s, 2):
        sim = np.abs(np.dot(v1, v2))  # Abs because sign is arbitrary
        similarities.append(sim)

    return np.mean(test_variances), np.mean(similarities)


# --- MAIN LOOP ---
for struct in STRUCTURES:
    print(f"\n=== Analyzing {struct.upper()} Stability ===")

    stability_scores = {cond: [] for cond in CONDITIONS}
    variance_scores = {cond: [] for cond in CONDITIONS}

    # Load Data
    data_map = {}
    for cond in CONDITIONS:
        data = load_deltas(struct, cond)
        if data is not None:
            data_map[cond] = data

    # Scan Layers
    for layer in tqdm(range(37), desc="Scanning Layers"):
        for cond in CONDITIONS:
            if cond not in data_map: continue

            layer_data = data_map[cond][:, layer, :]
            mean_var, mean_sim = analyze_layer_stability(layer_data)

            variance_scores[cond].append(mean_var)
            stability_scores[cond].append(mean_sim)

    # --- PLOTTING ---
    # Plot 1: Directional Stability (The New Metric)
    plt.figure(figsize=(10, 5))
    for cond in CONDITIONS:
        if cond in stability_scores and stability_scores[cond]:
            label = "Core" if "CORE" in cond else ("Anomalous" if "ANOMALOUS" in cond else "Jabberwocky")
            plt.plot(stability_scores[cond], label=label, marker='o', markersize=3)

    plt.title(f"{struct.capitalize()}: Stability of Syntax Axis (Cross-Fold Cosine Sim)", fontsize=14)
    plt.xlabel("Layer")
    plt.ylabel("Directional Stability (0-1)")
    plt.axhline(0.9, color='green', linestyle='--', label="Robust Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, f"PCA_Stability_{struct}.png"))
    plt.show()