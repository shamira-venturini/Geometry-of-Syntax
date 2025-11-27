import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm

# --- CONFIGURATION ---
INPUT_DIR = "/content/Geometry-of-Syntax/activations"
OUTPUT_DIR = "/content/Geometry-of-Syntax/plots/geometryKFOLD"
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


def get_pca_kfold_stats(data_layer, n_splits=5):
    """
    Performs 5-Fold CV PCA.
    Returns:
    - Mean Explained Variance (Test)
    - Std Explained Variance (Test)
    - The PC1 vector (computed on full data for geometry comparison)
    """
    # 1. Center Data
    mean_vec = np.mean(data_layer, axis=0)
    centered_data = data_layer - mean_vec

    # 2. K-Fold Loop
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    test_variances = []

    for train_idx, test_idx in kf.split(centered_data):
        X_train = centered_data[train_idx]
        X_test = centered_data[test_idx]

        # Fit on Train
        pca = PCA(n_components=1)
        pca.fit(X_train)

        # Evaluate on Test
        pc1 = pca.components_[0]
        projections = X_test @ pc1
        reconstruction = np.outer(projections, pc1)

        # Calculate Variance Explained in Test Set
        # (Variance of Reconstruction / Variance of Original Data)
        var_test = np.var(reconstruction) / np.var(X_test)
        test_variances.append(var_test)

    # 3. Compute "Global" PC1 for Cosine Similarity
    # (We use the full dataset to get the most stable direction for comparison)
    pca_full = PCA(n_components=1)
    pca_full.fit(centered_data)
    global_pc1 = pca_full.components_[0]

    return np.mean(test_variances), np.std(test_variances), global_pc1


# --- MAIN LOOP ---
for struct in STRUCTURES:
    print(f"\n=== Analyzing {struct.upper()} Geometry (5-Fold CV) ===")

    # Storage
    layer_stats = {cond: {"mean": [], "std": [], "pc1": []} for cond in CONDITIONS}

    # Load Data
    data_map = {}
    for cond in CONDITIONS:
        data = load_deltas(struct, cond)
        if data is not None:
            data_map[cond] = data

    # Iterate Layers
    n_layers = 37
    for layer in tqdm(range(n_layers), desc="Scanning Layers"):
        for cond in CONDITIONS:
            if cond not in data_map: continue

            layer_data = data_map[cond][:, layer, :]

            mean_var, std_var, pc1 = get_pca_kfold_stats(layer_data)

            layer_stats[cond]["mean"].append(mean_var)
            layer_stats[cond]["std"].append(std_var)
            layer_stats[cond]["pc1"].append(pc1)

    # --- PLOTTING ---

    # Plot A: Variance with Error Bars
    plt.figure(figsize=(10, 5))
    for cond in CONDITIONS:
        if cond in layer_stats and layer_stats[cond]["mean"]:
            label = "Core" if "CORE" in cond else ("Anomalous" if "ANOMALOUS" in cond else "Jabberwocky")

            means = np.array(layer_stats[cond]["mean"])
            stds = np.array(layer_stats[cond]["std"])

            plt.plot(range(n_layers), means, label=label, marker='o', markersize=3)
            plt.fill_between(range(n_layers), means - stds, means + stds, alpha=0.2)

    plt.title(f"{struct.capitalize()}: Syntax Subspace Strength (5-Fold CV)", fontsize=14)
    plt.xlabel("Layer")
    plt.ylabel("Explained Variance (Test Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, f"PCA_Variance_KFold_{struct}.png"))
    plt.show()

    # Plot B: Cosine Similarity (Core vs Jabberwocky)
    if "CORE" in layer_stats and "jabberwocky" in layer_stats:
        similarities = []
        for i in range(n_layers):
            v_core = layer_stats["CORE"]["pc1"][i]
            v_jabb = layer_stats["jabberwocky"]["pc1"][i]
            sim = np.abs(np.dot(v_core, v_jabb))
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
        plt.savefig(os.path.join(OUTPUT_DIR, f"PCA_Similarity_{struct}.png"))
        plt.show()

        peak_layer = np.argmax(similarities)
        print(f"  Peak Alignment Layer: {peak_layer} (Sim: {similarities[peak_layer]:.3f})")