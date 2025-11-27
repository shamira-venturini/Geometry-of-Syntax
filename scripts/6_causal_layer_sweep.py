import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# --- CONFIGURATION ---
MODEL_NAME = "gpt2-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIR = "activations"
OUTPUT_DIR = "plots/steering"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA = 15.0  # Increased Alpha to force the effect

# Define Target Sets (The "Category")
PASSIVE_AUXILIARIES = ["was", "is", "were", "are", "been", "be"]
DATIVE_PREPOSITIONS = ["to", "for"]

EXPERIMENTS = [
    {
        "name": "Transitive",
        "prompt": "The box",
        "targets": PASSIVE_AUXILIARIES,
        "vector_file_1": "delta_jabberwocky_transitive_struct1.npy",
        "vector_file_2": "delta_jabberwocky_transitive_struct2.npy"
    },
    {
        "name": "Dative",
        "prompt": "The girl gave the book",
        "targets": DATIVE_PREPOSITIONS,
        "vector_file_1": "delta_jabberwocky_dative_struct1.npy",
        "vector_file_2": "delta_jabberwocky_dative_struct2.npy"
    }
]

# --- LOAD MODEL ---
print(f"Loading {MODEL_NAME}...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()


# --- HELPER: GET PC1 ---
def get_layer_pc1(f1, f2, layer_idx):
    d1 = np.load(os.path.join(INPUT_DIR, f1))[:, layer_idx, :]
    d2 = np.load(os.path.join(INPUT_DIR, f2))[:, layer_idx, :]
    data = np.concatenate([d1, d2], axis=0)
    mean_vec = np.mean(data, axis=0)
    centered = data - mean_vec
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(centered)
    return torch.tensor(pca.components_[0], dtype=torch.float32).to(DEVICE)


# --- HELPER: HOOK ---
def get_hook(vector, alpha):
    def hook(module, input, output):
        output[0][:, -1, :] += alpha * vector
        return output

    return hook


# --- MAIN LOOP ---
for exp in EXPERIMENTS:
    print(f"\nRunning Layer Sweep for {exp['name']}...")

    # Get Target IDs
    # Note: We add space prefix " " to match GPT-2 tokenization
    target_ids = [tokenizer.encode(" " + t)[0] for t in exp['targets']]

    inputs = tokenizer(exp['prompt'], return_tensors="pt").to(DEVICE)

    # Baseline
    with torch.no_grad():
        base_logits = model(**inputs).logits[0, -1, :]
        base_probs = torch.softmax(base_logits, dim=0)
        base_score = sum(base_probs[tid].item() for tid in target_ids)

    layer_effects = []

    # Sweep Layers 0 to 35 (GPT-2 Large has 36 layers, indices 0-35)
    # Note: Our extraction had 37 layers (0=Embed, 1-36=Blocks).
    # model.transformer.h has 36 blocks.
    # So Extraction Layer 1 corresponds to model.transformer.h[0].
    # We skip Extraction Layer 0 (Embeddings) for steering.

    for layer_idx in range(36):
        # We use Extraction Layer (layer_idx + 1) because Extraction 0 is embeddings
        pc1 = get_layer_pc1(exp['vector_file_1'], exp['vector_file_2'], layer_idx + 1)

        probs = []
        for sign in [1, -1]:
            eff_alpha = ALPHA * sign
            layer_module = model.transformer.h[layer_idx]
            handle = layer_module.register_forward_hook(get_hook(pc1, eff_alpha))

            with torch.no_grad():
                logits = model(**inputs).logits[0, -1, :]
                probs_all = torch.softmax(logits, dim=0)
                score = sum(probs_all[tid].item() for tid in target_ids)

            probs.append(score)
            handle.remove()

        best_score = max(probs)
        fold_change = best_score / base_score
        layer_effects.append(fold_change)

        print(f"  Layer {layer_idx}: Base={base_score:.4f} -> Steered={best_score:.4f} (x{fold_change:.1f})")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(36), layer_effects, marker='o', linewidth=2, color='#8e44ad')
    plt.title(f"Causal Layer Sweep: {exp['name']}", fontsize=14)
    plt.xlabel("Layer Injection Site")
    plt.ylabel("Probability Multiplier (x Baseline)")
    plt.axhline(1.0, color='black', linestyle='--')
    plt.grid(True, alpha=0.3)

    peak_layer = np.argmax(layer_effects)
    peak_val = layer_effects[peak_layer]
    plt.plot(peak_layer, peak_val, marker='*', color='red', markersize=15)
    plt.text(peak_layer, peak_val + 0.1, f"Peak: L{peak_layer}", ha='center', fontweight='bold')

    plt.savefig(os.path.join(OUTPUT_DIR, f"Sweep_{exp['name']}_FIXED.png"))
    plt.show()