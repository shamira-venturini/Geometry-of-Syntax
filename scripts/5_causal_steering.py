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

# Steering Targets
# We use the Peak Layers found in PCA
TARGETS = [
    {"struct": "transitive", "layer": 4, "prompt": "The box", "target_token": "was"},
    {"struct": "dative", "layer": 35, "prompt": "The girl gave the boy", "target_token": "to"}
]

# --- LOAD MODEL ---
print(f"Loading {MODEL_NAME}...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()


# --- HELPER: LOAD PC1 VECTOR ---
def get_pc1_vector(structure, layer):
    # Load Jabberwocky Deltas (We steer with the "Pure Syntax" vector)
    # Filename format: delta_jabberwocky_[struct]_struct1.npy
    f1 = f"delta_jabberwocky_{structure}_struct1.npy"
    f2 = f"delta_jabberwocky_{structure}_struct2.npy"

    d1 = np.load(os.path.join(INPUT_DIR, f1))[:, layer, :]
    d2 = np.load(os.path.join(INPUT_DIR, f2))[:, layer, :]

    # Combine and Center
    data = np.concatenate([d1, d2], axis=0)
    mean_vec = np.mean(data, axis=0)
    centered = data - mean_vec

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(centered)

    # The direction is PC1.
    # Note: PCA sign is arbitrary. We might need to flip it.
    # We assume Positive = Active/DO, Negative = Passive/PO based on subtraction order.
    # Delta = Active - Passive. So Positive -> Active.
    # To steer towards Passive, we need to SUBTRACT this vector (or add negative).
    return torch.tensor(pca.components_[0], dtype=torch.float32).to(DEVICE)


# --- HELPER: STEERING HOOK ---
def get_steering_hook(vector, alpha):
    def hook(module, input, output):
        # Output is tuple (hidden_state, ...)
        # Hidden state is [Batch, Seq, Dim]
        # We add the vector to the LAST token position
        # shape of vector: [1280]

        # Broadcast addition
        # We add to the entire sequence or just the last token?
        # Usually adding to the last token is sufficient for next-token prediction.
        output[0][:, -1, :] += alpha * vector
        return output

    return hook


# --- MAIN LOOP ---
for target in TARGETS:
    struct = target["struct"]
    layer_idx = target["layer"]
    prompt = target["prompt"]
    token_str = target["target_token"]

    print(f"\nSteering {struct.upper()} at Layer {layer_idx}...")

    # 1. Get Vector
    pc1 = get_pc1_vector(struct, layer_idx)

    # 2. Define Sweep
    alphas = np.linspace(-15, 15, 20)  # Sweep from -15 to +15
    probs = []

    # Token ID to measure
    target_id = tokenizer.encode(" " + token_str)[0]

    # 3. Run Sweep
    for alpha in alphas:
        # Register Hook
        # GPT-2 Layers are in model.transformer.h
        layer_module = model.transformer.h[layer_idx]
        handle = layer_module.register_forward_hook(get_steering_hook(pc1, alpha))

        # Run Inference
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get Probability of Target Token
        # Logits: [Batch, Seq, Vocab] -> Last token logits
        last_token_logits = logits[0, -1, :]
        probs_all = torch.softmax(last_token_logits, dim=0)
        prob = probs_all[target_id].item()
        probs.append(prob)

        # Remove Hook
        handle.remove()

    # 4. Plot
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, probs, marker='o', linewidth=2, color='#d35400')
    plt.title(f"Causal Steering: {struct.capitalize()} (Layer {layer_idx})", fontsize=14)
    plt.xlabel("Steering Coefficient (alpha)")
    plt.ylabel(f"Probability of '{token_str}'")
    plt.grid(True, alpha=0.3)

    # Add annotation
    plt.text(0.05, 0.95, f"Prompt: '{prompt}'", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(OUTPUT_DIR, f"Steering_{struct}.png"))
    plt.show()