import torch
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm.notebook import tqdm
import os

# --- CONFIGURATION ---
MODEL_NAME = "gpt2-large"
BATCH_SIZE = 16
SUBSET_SIZE = 3000  # Optimization: Run on 3k items per corpus
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Exact paths based on your Colab structure
FILES = [
    "/content/Geometry-of-Syntax/corpora/transitive/jabberwocky_transitive.csv",
    "/content/Geometry-of-Syntax/corpora/dative/jabberwocky_dative.csv",
    "/content/Geometry-of-Syntax/corpora/transitive/CORE_transitive_15000sampled_10-1.csv",
    "/content/Geometry-of-Syntax/corpora/dative/CORE_dative_15000sampled_10-1.csv",
    "/content/Geometry-of-Syntax/corpora/transitive/ANOMALOUS_chomsky_transitive_15000sampled_10-1.csv",
    "/content/Geometry-of-Syntax/corpora/dative/ANOMALOUS_chomsky_dative_15000sampled_10-1.csv"
]

OUTPUT_DIR = "activations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD MODEL ---
print(f"Loading {MODEL_NAME} on {DEVICE} (FP16)...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
# output_hidden_states=True is REQUIRED for extraction
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, output_hidden_states=True)
model.to(DEVICE)
model.eval()


def get_critical_index_relative(tokenizer, target_text, structure_type):
    """
    Calculates the token index of the critical word relative to the start of the target.
    FIXED LOGIC: Counts tokens of the prefix words.
    """
    words = target_text.strip().split()

    if "transitive" in structure_type:
        # Template: "The(0) dax(1) [gimbled](2)..."
        word_idx = 2
    elif "dative" in structure_type:
        # Template: "The(0) dax(1) gimbled(2) the(3) wug(4) [to](5)..."
        word_idx = 5
    else:
        word_idx = 2

    # Reconstruct prefix to count tokens
    prefix_str = " " + " ".join(words[:word_idx])
    prefix_tokens = tokenizer.encode(prefix_str)
    return len(prefix_tokens)


def extract_batch(batch_primes, batch_targets, structure_type):
    """
    Runs the model and extracts the hidden state at the critical token for all 37 layers.
    Returns: Numpy array [Batch, 37, 1280]
    """
    full_texts = [p + " " + t for p, t in zip(batch_primes, batch_targets)]
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # Stack layers: [37, Batch, Seq, Dim] -> [Batch, 37, Seq, Dim]
    all_layers = torch.stack(outputs.hidden_states).permute(1, 0, 2, 3)

    batch_vectors = []

    for i in range(len(batch_primes)):
        prime_len = len(tokenizer.encode(batch_primes[i]))

        # Calculate dynamic offset for this specific sentence
        offset = get_critical_index_relative(tokenizer, batch_targets[i], structure_type)

        # The target starts at `prime_len`. The critical token is `prime_len + offset`
        crit_idx = prime_len + offset

        # Safety clamp
        crit_idx = min(crit_idx, int(inputs.attention_mask[i].sum() - 1))

        # Extract [37, 1280] for this item
        vectors = all_layers[i, :, crit_idx, :].cpu().numpy()  # [37, 1280]
        batch_vectors.append(vectors)

    return np.array(batch_vectors)  # [Batch, 37, 1280]


# --- MAIN LOOP ---
for filename in FILES:
    if not os.path.exists(filename):
        print(f"Skipping {filename} (not found)")
        continue

    print(f"Processing {filename}...")

    df = pd.read_csv(filename)
    # Safety: Normalize columns
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {'p_do': 'pdo', 'p_po': 'ppo', 't_do': 'tdo', 't_po': 'tpo'}
    df.rename(columns=rename_map, inplace=True)

    # 1. SAMPLING (Optimization)
    if len(df) > SUBSET_SIZE:
        print(f"  Sampling {SUBSET_SIZE} items...")
        df = df.sample(n=SUBSET_SIZE, random_state=42).reset_index(drop=True)

    structure_type = "transitive" if "transitive" in filename else "dative"

    # Storage for Subtraction Vectors (Deltas)
    deltas_struct_1 = []  # Active or DO
    deltas_struct_2 = []  # Passive or PO

    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[i: i + BATCH_SIZE]

        if structure_type == "transitive":
            pa, pp = batch['pa'].tolist(), batch['pp'].tolist()
            ta, tp = batch['ta'].tolist(), batch['tp'].tolist()

            # Analysis A: Target Active (ta)
            # Delta = h(ta|pa) - h(ta|pp)
            h_cong = extract_batch(pa, ta, structure_type)
            h_inc = extract_batch(pp, ta, structure_type)
            deltas_struct_1.append(h_cong - h_inc)

            # Analysis B: Target Passive (tp)
            # Delta = h(tp|pp) - h(tp|pa)
            h_cong = extract_batch(pp, tp, structure_type)
            h_inc = extract_batch(pa, tp, structure_type)
            deltas_struct_2.append(h_cong - h_inc)

        else:  # Dative
            # Use normalized column names (pdo, ppo...)
            p_do, p_po = batch['pdo'].tolist(), batch['ppo'].tolist()
            t_do, t_po = batch['tdo'].tolist(), batch['tpo'].tolist()

            # Analysis A: Target DO
            # Delta = h(t_do|p_do) - h(t_do|p_po)
            h_cong = extract_batch(p_do, t_do, structure_type)
            h_inc = extract_batch(p_po, t_do, structure_type)
            deltas_struct_1.append(h_cong - h_inc)

            # Analysis B: Target PO
            # Delta = h(t_po|p_po) - h(t_po|p_do)
            h_cong = extract_batch(p_po, t_po, structure_type)
            h_inc = extract_batch(p_do, t_po, structure_type)
            deltas_struct_2.append(h_cong - h_inc)

    # Concatenate and Save
    final_delta_1 = np.concatenate(deltas_struct_1, axis=0)
    final_delta_2 = np.concatenate(deltas_struct_2, axis=0)

    base_name = os.path.basename(filename).replace(".csv", "")

    # Save Active/DO Deltas
    np.save(os.path.join(OUTPUT_DIR, f"delta_{base_name}_struct1.npy"), final_delta_1)
    # Save Passive/PO Deltas
    np.save(os.path.join(OUTPUT_DIR, f"delta_{base_name}_struct2.npy"), final_delta_2)

    print(f"Saved deltas for {base_name}. Shape: {final_delta_1.shape}")

print("Extraction Complete.")