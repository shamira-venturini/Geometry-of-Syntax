import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm

# --- CONFIGURATION ---
INPUT_DIR = "activations"
OUTPUT_DIR = "plots/geometry"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Structure Definitions
# We analyze Transitive and Dative separately
STRUCTURES = ["transitive", "dative"]
CONDITIONS = ["CORE", "ANOMALOUS_chomsky", "jabberwocky"]


def load_deltas(structure, condition):
    """Loads and concatenates struct1 (Active/DO) and struct2 (Passive/PO) deltas."""
    # Find matching files
    files = [f for f in os.listdir(INPUT_DIR) if structure in f and condition in f]
    if not files:
        print(f"  Warning: No files found for {structure} {condition}")
        return None

    # Load struct1 and struct2
    # Note: Filenames created by extraction script: delta_[original_name]_struct1.npy
    f1 = next((f for f in files if "struct1" in f), None)
    f2 = next((f for f in files if "struct2" in f), None)

    if f1 is None or f2 is None:
        return None

    d1 = np.load(os.path.join(INPUT_DIR, f1))  # [N, 37, 1280]
    d2 = np.load(os.path.join(INPUT_DIR, f2))

    # Combine them to find the common axis of variation
    # We stack them: [2*N, 37, 1280]
    return np.concatenate([d1, d2], axis=0)


def get_pca_stats(data_layer):
    """
    Performs Cross-Validated PCA on a specific layer.
    Returns: PC1 Vector, Explained Variance (Train), Explained Variance (Test)
    """
    # 1. Center the data (Crucial!)
    mean_vec = np.mean(data_layer, axis=0)
    centered_data = data_layer - mean_vec

    # 2. Cross-Validation Split
    # We use ShuffleSplit to simulate "Held-out" data
    ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(ss.split(centered_data))

    X_train = centered_data[train_idx]
    X_test = centered_data[test_idx]

    # 3. Fit PCA on Train
    pca = PCA(n_components=1)
    pca.fit(X_train)

    # 4. Evaluate
    # Explained Variance Ratio on Train
    var_train = pca.explained_variance_ratio_[0]

    # Explained Variance on Test (Manual Calculation)
    # How much of the Test variance is captured by the Train PC1 axis?
    # Project test data onto PC1
    pc1 = pca.components_[0]
    projections = X_test @ pc1
    reconstruction = np.outer(projections, pc1)

    # Var(Reconstruction) / Var(Original)
    var_test = np.var(reconstruction) / np.var(X_test)

    return pc1, var_train, var_test


# --- MAIN LOOP ---
results = []

for struct in STRUCTURES:
    print(f"\n=== Analyzing {struct.upper()} Geometry ===")

    # Storage for layer-wise metrics
    layer_stats = {cond: {"var_test": [], "pc1": []} for cond in CONDITIONS}

    # 1. Load Data for all conditions first (to ensure alignment)
    data_map = {}
    for cond in CONDITIONS:
        data = load_deltas(struct, cond)
        if data is not None:
            data_map[cond] = data
            print(f"  Loaded {cond}: {data.shape}")

    # 2. Iterate Layers
    n_layers = 37
    for layer in tqdm(range(n_layers), desc="Scanning Layers"):

        # Calculate PCA for each condition independently
        for cond in CONDITIONS:
            if cond not in data_map: continue

            # Extract specific layer: [N, 1280]
            layer_data = data_map[cond][:, layer, :]

            pc1, var_train, var_test = get_pca_stats(layer_data)

            layer_stats[cond]["var_test"].append(var_test)
            layer_stats[cond]["pc1"].append(pc1)

    # 3. Calculate Cosine Similarities (The Autonomy Test)
    # We compare Core vs Jabberwocky at every layer
    similarities = []
    if "CORE" in layer_stats and "jabberwocky" in layer_stats:
        for i in range(n_layers):
            v_core = layer_stats["CORE"]["pc1"][i]
            v_jabb = layer_stats["jabberwocky"]["pc1"][i]

            # Cosine Sim = (A . B) / (|A|*|B|)
            # Vectors from sklearn PCA are already normalized
            sim = np.abs(np.dot(v_core, v_jabb))  # Abs because sign is arbitrary in PCA
            similarities.append(sim)

    # 4. Plotting Results for this Structure

    # Plot A: Explained Variance (Where is the Syntax?)
    plt.figure(figsize=(10, 5))
    for cond in CONDITIONS:
        if cond in layer_stats and layer_stats[cond]["var_test"]:
            label = "Core" if "CORE" in cond else ("Anomalous" if "ANOMALOUS" in cond else "Jabberwocky")
            plt.plot(layer_stats[cond]["var_test"], label=label, marker='o', markersize=4)

    plt.title(f"{struct.capitalize()}: Strength of Syntax Subspace (PCA Explained Variance)", fontsize=14)
    plt.xlabel("Layer")
    plt.ylabel("Variance Explained by PC1 (Test Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, f"PCA_Variance_{struct}.png"))
    plt.show()

    # Plot B: Cosine Similarity (Is it the same Syntax?)
    if similarities:
        plt.figure(figsize=(10, 5))
        plt.plot(similarities, color='purple', linewidth=2, marker='s')
        plt.title(f"{struct.capitalize()}: Geometric Alignment (Core vs. Jabberwocky)", fontsize=14)
        plt.xlabel("Layer")
        plt.ylabel("Cosine Similarity (|PC1_core . PC1_jabb|)")
        plt.ylim(0, 1.1)
        plt.axhline(0.8, color='green', linestyle='--', label="High Similarity Threshold")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, f"PCA_Similarity_{struct}.png"))
        plt.show()

        # Find Peak Similarity
        peak_layer = np.argmax(similarities)
        print(f"  Peak Alignment Layer: {peak_layer} (Sim: {similarities[peak_layer]:.3f})")