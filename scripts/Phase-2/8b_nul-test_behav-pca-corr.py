import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from scipy.stats import pearsonr

# --- CONFIGURATION ---
BASE_DIR = "/content/Geometry-of-Syntax"  # adjust
ACTIV_DIR = os.path.join(BASE_DIR, "activations")           # per-item deltas
PC1_DIR   = os.path.join(BASE_DIR, "pca_results/activations")
BEHAV_DIR = os.path.join(BASE_DIR, "behavioral_results/s-PE")
OUT_DIR   = os.path.join(BASE_DIR, "pca_results/correlations")
os.makedirs(OUT_DIR, exist_ok=True)

STRUCTURES = ["transitive", "dative"]
CONDITIONS = ["CORE", "ANOMALOUS_chomsky", "jabberwocky"]
N_LAYERS = 37


def load_deltas(structure, condition):
    """Load per-item deltas [items, layers, dim] for a given structure/condition."""
    files = [f for f in os.listdir(ACTIV_DIR)
             if structure in f and condition in f and f.endswith(".npy")]
    if not files:
        return None
    f1 = next((f for f in files if "struct1" in f), None)
    f2 = next((f for f in files if "struct2" in f), None)
    if f1 is None or f2 is None:
        return None
    d1 = np.load(os.path.join(ACTIV_DIR, f1))
    d2 = np.load(os.path.join(ACTIV_DIR, f2))
    return np.concatenate([d1, d2], axis=0)  # [items, layers, dim]


def load_pc1(structure, condition):
    """Load PC1 array [layers, dim] for structure/condition."""
    fname = f"PC1_{structure}_{condition}.npy"
    path = os.path.join(PC1_DIR, fname)
    if not os.path.exists(path):
        return None
    pc1 = np.load(path)
    # ensure unit norm per layer
    norms = np.linalg.norm(pc1, axis=1, keepdims=True) + 1e-12
    return pc1 / norms


def load_behavior(structure, condition):
    """Load behavioral s-PE per item in matching order."""
    files = [f for f in os.listdir(BEHAV_DIR)
             if structure in f and condition in f and f.endswith(".csv")]
    if not files:
        return None
    # if multiple, take first
    df = pd.read_csv(os.path.join(BEHAV_DIR, files[0]))
    return df


def pick_pe_column(structure):
    """Choose which s-PE column to correlate with PC1 strength."""
    if structure == "transitive":
        # Passive is the marked structure with strongest effects
        return "s_PE_Passive"
    else:  # dative
        # You can change to s_PE_PO if that is your main focus
        return "s_PE_DO"


rows = []

for struct in STRUCTURES:
    print(f"\n=== Correlations for {struct.upper()} ===")
    pe_col = pick_pe_column(struct)

    for cond in CONDITIONS:
        deltas = load_deltas(struct, cond)
        pc1 = load_pc1(struct, cond)
        behav = load_behavior(struct, cond)

        if deltas is None or pc1 is None or behav is None:
            print(f"  Missing data for {struct}, {cond}, skipping.")
            continue

        # sanity: number of items should match behavioral rows
        n_items = deltas.shape[0]
        if len(behav) != n_items:
            print(f"  WARNING: item count mismatch for {struct}, {cond} "
                  f"({n_items} activations vs {len(behav)} behavior)")
            n_items = min(n_items, len(behav))
            deltas = deltas[:n_items]
            behav = behav.iloc[:n_items].reset_index(drop=True)

        pe_vals = behav[pe_col].values

        for layer in tqdm(range(N_LAYERS), desc=f"{cond} ({struct})"):
            layer_deltas = deltas[:, layer, :]            # [items, dim]
            v = pc1[layer]                                # [dim]

            # projection strength per item (absolute)
            proj = layer_deltas @ v
            strength = np.abs(proj)

            if np.allclose(strength, 0):
                r, p = np.nan, np.nan
            else:
                r, p = pearsonr(strength, pe_vals)

            rows.append({
                "structure": struct,
                "condition": cond,
                "layer": layer,
                "pe_column": pe_col,
                "corr_r": r,
                "p_value": p,
                "mean_strength": float(strength.mean()),
                "std_strength": float(strength.std()),
            })

# save results
corr_df = pd.DataFrame(rows)
out_path = os.path.join(OUT_DIR, "PC1_behavior_correlations.csv")
corr_df.to_csv(out_path, index=False)
print(f"\nSaved correlations to {out_path}")
