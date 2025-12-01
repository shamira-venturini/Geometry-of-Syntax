import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from scipy import stats

# --- CONFIGURATION ---
MODEL_NAME = "gpt2-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIR = "activations"
OUTPUT_DIR = "plots/steering"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# We use a strong alpha to force the effect
ALPHA = 10.0

# 20 Jabberwocky Prompts (The "Ingredients")
PROMPTS = [
    "The dax", "The wug", "The fep", "The zom", "The blicket",
    "The glorp", "The tove", "The slithy", "The borogove", "The mome",
    "The rath", "The jubjub", "The tumtum", "The snark", "The boojum",
    "The vorpal", "The uffish", "The whiffling", "The burbled", "The galumph"
]

# The Target we want to boost (Passive Marker)
TARGET_TOKEN = "was"

# --- LOAD MODEL ---
print(f"Loading {MODEL_NAME}...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()


# --- HELPER: GET PC1 ---
def get_layer_pc1(layer_idx):
    # Load Transitive Jabberwocky Vectors
    f1 = "delta_jabberwocky_transitive_struct1.npy"  # Active
    f2 = "delta_jabberwocky_transitive_struct2.npy"  # Passive

    d1 = np.load(os.path.join(INPUT_DIR, f1))[:, layer_idx, :]
    d2 = np.load(os.path.join(INPUT_DIR, f2))[:, layer_idx, :]

    data = np.concatenate([d1, d2], axis=0)
    mean_vec = np.mean(data, axis=0)
    centered = data - mean_vec

    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(centered)

    # We need to ensure the vector points towards PASSIVE.
    # In our subtraction (Active - Passive), Passive is Negative.
    # So we want the Negative direction of PC1 (or Positive if PCA flipped it).
    # We will try both +Alpha and -Alpha in the loop and pick the one that works.
    return torch.tensor(pca.components_[0], dtype=torch.float32).to(DEVICE)


# --- HELPER: HOOK ---
def get_hook(vector, alpha):
    def hook(module, input, output):
        output[0][:, -1, :] += alpha * vector
        return output

    return hook


# --- MAIN EXPERIMENT ---
print("Running Layer Sweep with Significance Testing...")

layer_t_stats = []
layer_p_values = []
layer_mean_diffs = []

target_id = tokenizer.encode(" " + TARGET_TOKEN)[0]

# Sweep Layers 0 to 35
for layer in range(36):
    # 1. Get Vector
    # Use Extraction Layer (layer+1) to match Model Layer (layer)
    pc1 = get_layer_pc1(layer + 1)

    # 2. Collect Data for this Layer
    base_probs = []
    steered_probs = []

    # We determine the "Passive Direction" using the first prompt
    # Try +Alpha and -Alpha, see which one boosts "was"
    direction = 0
    for sign in [1, -1]:
        temp_hook = model.transformer.h[layer].register_forward_hook(get_hook(pc1, ALPHA * sign))
        inputs = tokenizer(PROMPTS[0], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
            prob = torch.softmax(logits, dim=0)[target_id].item()
        temp_hook.remove()

        # Get baseline for first prompt
        with torch.no_grad():
            base_logits = model(**inputs).logits[0, -1, :]
            base_p = torch.softmax(base_logits, dim=0)[target_id].item()

        if prob > base_p:
            direction = sign
            break

    if direction == 0: direction = 1  # Default if no change

    # 3. Run Batch
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # Baseline
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
            p_base = torch.softmax(logits, dim=0)[target_id].item()
            base_probs.append(p_base)

        # Steered
        hook = model.transformer.h[layer].register_forward_hook(get_hook(pc1, ALPHA * direction))
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
            p_steer = torch.softmax(logits, dim=0)[target_id].item()
            steered_probs.append(p_steer)
        hook.remove()

    # 4. Statistical Test (Paired T-Test)
    # H0: Mean(Steered) - Mean(Base) = 0
    t_stat, p_val = stats.ttest_rel(steered_probs, base_probs)
    mean_diff = np.mean(steered_probs) - np.mean(base_probs)

    layer_t_stats.append(t_stat)
    layer_p_values.append(p_val)
    layer_mean_diffs.append(mean_diff)

    sig_marker = "*" if p_val < 0.001 else ""
    print(f"L{layer}: Mean Diff={mean_diff:.4f}, T={t_stat:.2f}, p={p_val:.2e} {sig_marker}")

# --- PLOTTING ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot T-Statistic (Significance)
color = 'tab:blue'
ax1.set_xlabel('Layer')
ax1.set_ylabel('T-Statistic (Significance)', color=color, fontsize=12)
ax1.plot(range(36), layer_t_stats, color=color, marker='o', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.axhline(3.5, color='gray', linestyle='--', label="p < 0.001 Threshold")  # Approx T for p=0.001 with df=19

# Highlight Peak
peak_layer = np.argmax(layer_t_stats)
plt.title(f"Causal Significance of Syntax Vector (Transitive)", fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.savefig(os.path.join(OUTPUT_DIR, "Figure_Steering_Significance.png"))
plt.show()

print(f"Most Significant Causal Layer: {peak_layer} (T={layer_t_stats[peak_layer]:.2f})")