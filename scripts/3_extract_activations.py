import torch
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm.notebook import tqdm
import os

# --- CONFIGURATION ---
MODEL_NAME = "gpt2-large"
BATCH_SIZE = 16
SUBSET_SIZE = 3000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Updated File List
FILES = [
    # TRANSITIVE
    "/content/Geometry-of-Syntax/corpora/transitive/jabberwocky_transitive.csv",
    "/content/Geometry-of-Syntax/corpora/transitive/CORE_transitive_15000sampled_10-1.csv",
    "/content/Geometry-of-Syntax/corpora/transitive/ANOMALOUS_chomsky_transitive_15000sampled_10-1.csv",

    # DATIVE
    "/content/Geometry-of-Syntax/corpora/dative/jabberwocky_dative.csv",
    "/content/Geometry-of-Syntax/corpora/dative/CORE_dative_15000sampled_10-1.csv",
    "/content/Geometry-of-Syntax/corpora/dative/ANOMALOUS_chomsky_dative_15000sampled_10-1.csv",

    # ERGATIVE (New)
    "/content/Geometry-of-Syntax/corpora/transitive/jabberwocky_ergative.csv",
    "/content/Geometry-of-Syntax/corpora/transitive/CORE_causative.csv"
]

OUTPUT_DIR = "activations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD MODEL ---
print(f"Loading {MODEL_NAME} on {DEVICE} (FP16)...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, output_hidden_states=True)
model.to(DEVICE)
model.eval()


def get_structure_type(filename):
    if "ergative" in filename or "causative" in filename:
        return "ergative"
    elif "dative" in filename:
        return "dative"
    elif "transitive" in filename:
        return "transitive"
    return "unknown"


def get_critical_index_relative(tokenizer, target_text, structure_type):
    """
    Calculates the token index of the critical word relative to the start of the target.
    """
    words = target_text.strip().split()

    if structure_type == "transitive":
        # "The(0) boy(1) [kicked](2)..."
        word_idx = 2
    elif structure_type == "dative":
        # "The(0) girl(1) gave(2) the(3) boy(4) [to](5)..."
        word_idx = 5
    elif structure_type == "ergative":
        # Transitive: "The(0) dax(1) [gimbled](2)..."
        # Intransitive: "The(0) wug(1) [gimbled](2)..."
        word_idx = 2
    else:
        word_idx = 2

    prefix_str = " " + " ".join(words[:word_idx])
    prefix_tokens = tokenizer.encode(prefix_str)
    return len(prefix_tokens)


def extract_batch(batch_primes, batch_targets, structure_type):
    full_texts = [p + " " + t for p, t in zip(batch_primes, batch_targets)]
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    all_layers = torch.stack(outputs.hidden_states).permute(1, 0, 2, 3)
    batch_vectors = []

    for i in range(len(batch_primes)):
        prime_len = len(tokenizer.encode(batch_primes[i]))
        offset = get_critical_index_relative(tokenizer, batch_targets[i], structure_type)
        crit_idx = prime_len + offset
        crit_idx = min(crit_idx, int(inputs.attention_mask[i].sum() - 1))

        vectors = all_layers[i, :, crit_idx, :].cpu().numpy()
        batch_vectors.append(vectors)

    return np.array(batch_vectors)


# --- MAIN LOOP ---
for filename in FILES:
    if not os.path.exists(filename):
        print(f"Skipping {filename} (not found)")
        continue

    print(f"Processing {os.path.basename(filename)}...")
    df = pd.read_csv(filename)

    # Normalize columns
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {
        'p_do': 'pdo', 'p_po': 'ppo', 't_do': 'tdo', 't_po': 'tpo',
        'p_caus': 'p_trans', 'p_inch': 'p_intrans',
        't_caus': 't_trans', 't_inch': 't_intrans'
    }
    df.rename(columns=rename_map, inplace=True)

    # Sampling
    if len(df) > SUBSET_SIZE:
        df = df.sample(n=SUBSET_SIZE, random_state=42).reset_index(drop=True)

    structure_type = get_structure_type(filename)

    deltas_struct_1 = []  # Active / DO / Transitive
    deltas_struct_2 = []  # Passive / PO / Intransitive

    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[i: i + BATCH_SIZE]

        if structure_type == "transitive":
            pa, pp = batch['pa'].tolist(), batch['pp'].tolist()
            ta, tp = batch['ta'].tolist(), batch['tp'].tolist()

            # Active Target (Active - Passive)
            h_cong = extract_batch(pa, ta, structure_type)
            h_inc = extract_batch(pp, ta, structure_type)
            deltas_struct_1.append(h_cong - h_inc)

            # Passive Target (Passive - Active)
            h_cong = extract_batch(pp, tp, structure_type)
            h_inc = extract_batch(pa, tp, structure_type)
            deltas_struct_2.append(h_cong - h_inc)

        elif structure_type == "dative":
            p_do, p_po = batch['pdo'].tolist(), batch['ppo'].tolist()
            t_do, t_po = batch['tdo'].tolist(), batch['tpo'].tolist()

            # DO Target (DO - PO)
            h_cong = extract_batch(p_do, t_do, structure_type)
            h_inc = extract_batch(p_po, t_do, structure_type)
            deltas_struct_1.append(h_cong - h_inc)

            # PO Target (PO - DO)
            h_cong = extract_batch(p_po, t_po, structure_type)
            h_inc = extract_batch(p_do, t_po, structure_type)
            deltas_struct_2.append(h_cong - h_inc)

        elif structure_type == "ergative":
            p_trans = batch['p_trans'].tolist()
            p_intrans = batch['p_intrans'].tolist()
            t_trans = batch['t_trans'].tolist()
            t_intrans = batch['t_intrans'].tolist()

            # Transitive Target (Transitive - Intransitive)
            h_cong = extract_batch(p_trans, t_trans, structure_type)
            h_inc = extract_batch(p_intrans, t_trans, structure_type)
            deltas_struct_1.append(h_cong - h_inc)

            # Intransitive Target (Intransitive - Transitive)
            h_cong = extract_batch(p_intrans, t_intrans, structure_type)
            h_inc = extract_batch(p_trans, t_intrans, structure_type)
            deltas_struct_2.append(h_cong - h_inc)

    # Save
    final_delta_1 = np.concatenate(deltas_struct_1, axis=0)
    final_delta_2 = np.concatenate(deltas_struct_2, axis=0)

    base_name = os.path.basename(filename).replace(".csv", "")
    np.save(os.path.join(OUTPUT_DIR, f"delta_{base_name}_struct1.npy"), final_delta_1)
    np.save(os.path.join(OUTPUT_DIR, f"delta_{base_name}_struct2.npy"), final_delta_2)

    print(f"Saved deltas for {base_name}")

print("Extraction Complete.")