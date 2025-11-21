import torch
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import os

# --- CONFIGURATION ---
MODEL_NAME = "gpt2-large"
BATCH_SIZE = 16  # Adjust based on GPU (8-32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Files to process
FILES = [
    # TRANSITIVE
    "corpora/transitive/ANOMALOUS_chomsky_transitive_15000sampled_10-1.csv",
    "corpora/transitive/CORE_transitive_15000sampled_10-1.csv",
    "corpora/transitive/jabberwocky_transitive.csv",
    # DITRANSITIVE
    "corpora/ditransitive/ANOMALOUS_chomsky_ditransitive_15000sampled_10-1.csv",
    "corpora/ditransitive/CORE_dative_15000sampled_10-1.csv",
    "jabberwocky_dative.csv"
]

OUTPUT_DIR = "results_behavioral"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD MODEL ---
print(f"Loading {MODEL_NAME} on {DEVICE} (FP16)...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load in FP16 for speed/memory efficiency
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model.to(DEVICE)
model.eval()


def get_critical_offset(structure_type):
    """
    Returns the token index offset for the critical word relative to the target start.
    Based on Jabberwocky/Core templates.
    """
    if "transitive" in structure_type:
        # Template: " The(0) boy(1) [kicked](2)..."
        # Critical: Main Verb (Active) or Aux (Passive)
        return 2
    elif "dative" in structure_type:
        # Template: " The(0) girl(1) gave(2) the(3) boy(4) [to](5)..."
        # Critical: Preposition (PO) or 2nd Det (DO)
        return 5
    return 2


def calculate_log_probs(batch_primes, batch_targets, structure_type):
    """
    Calculates s-PE (sum of target log probs) and w-PE (critical token log prob).
    Replicates logic from Jumelet et al. (2024).
    """
    # 1. Prepare Inputs (Prime + Space + Target)
    full_texts = [p + " " + t for p, t in zip(batch_primes, batch_targets)]

    # Tokenize
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)

    # 2. Run Model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [Batch, Seq, Vocab]

    # 3. Shift Logits for Prediction (Standard LM logic)
    # Logit at [i] predicts Token at [i+1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs.input_ids[:, 1:].contiguous()

    # Calculate Log Softmax (Probabilities)
    log_probs_all = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Gather the log prob of the actual token that appeared
    # This matches Jumelet's: token_probs[range(size), token_ids]
    target_log_probs = log_probs_all.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

    # 4. Extract Specific Scores
    s_scores = []
    w_scores = []

    offset = get_critical_offset(structure_type)

    for i in range(len(batch_primes)):
        # Find where the target starts
        # We re-tokenize the prime to get its exact length in tokens
        # Note: We strip the padding from the prime calculation
        prime_tokens = tokenizer.encode(batch_primes[i])
        prime_len = len(prime_tokens)

        target_tokens = tokenizer.encode(" " + batch_targets[i])
        target_len = len(target_tokens)

        # The target starts at index `prime_len` in the `input_ids`
        # In `shift_labels` (which is shifted by 1), the target starts at `prime_len - 1`
        start_idx = prime_len - 1
        end_idx = start_idx + target_len

        # Safety check for padding boundaries
        valid_len = inputs.attention_mask[i].sum() - 1
        end_idx = min(end_idx, int(valid_len))

        # s-PE: Sum of log probs for the whole target
        sentence_score = target_log_probs[i, start_idx:end_idx].sum().item()
        s_scores.append(sentence_score)

        # w-PE: Critical Token
        # The critical token is at `start_idx + offset`
        crit_idx = start_idx + offset

        if crit_idx < end_idx:
            token_score = target_log_probs[i, crit_idx].item()
        else:
            # Fallback if sentence is unexpectedly short
            token_score = float('nan')

        w_scores.append(token_score)

    return s_scores, w_scores


# --- MAIN LOOP ---
for filename in FILES:
    if not os.path.exists(filename):
        print(f"Skipping {filename} (not found)")
        continue

    print(f"Processing {filename}...")
    df = pd.read_csv(filename)
    structure_type = "transitive" if "transitive" in filename else "dative"

    results = []

    # Process in Batches
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[i: i + BATCH_SIZE]

        if structure_type == "transitive":
            pa, pp = batch['pa'].tolist(), batch['pp'].tolist()
            ta, tp = batch['ta'].tolist(), batch['tp'].tolist()

            # 1. Target Active (ta)
            s_aa, w_aa = calculate_log_probs(pa, ta, structure_type)  # Congruent
            s_ba, w_ba = calculate_log_probs(pp, ta, structure_type)  # Incongruent

            # 2. Target Passive (tp)
            s_ab, w_ab = calculate_log_probs(pa, tp, structure_type)  # Incongruent
            s_bb, w_bb = calculate_log_probs(pp, tp, structure_type)  # Congruent

            for j in range(len(batch)):
                results.append({
                    "s_PE_Active": s_aa[j] - s_ba[j],
                    "s_PE_Passive": s_bb[j] - s_ab[j],
                    "w_PE_Active": w_aa[j] - w_ba[j],
                    "w_PE_Passive": w_bb[j] - w_ab[j]
                })

        else:  # Dative
            p_do, p_po = batch['p_do'].tolist(), batch['p_po'].tolist()
            t_do, t_po = batch['t_do'].tolist(), batch['t_po'].tolist()

            # 1. Target DO
            s_dd, w_dd = calculate_log_probs(p_do, t_do, structure_type)  # Congruent
            s_pd, w_pd = calculate_log_probs(p_po, t_do, structure_type)  # Incongruent

            # 2. Target PO
            s_dp, w_dp = calculate_log_probs(p_do, t_po, structure_type)  # Incongruent
            s_pp, w_pp = calculate_log_probs(p_po, t_po, structure_type)  # Congruent

            for j in range(len(batch)):
                results.append({
                    "s_PE_DO": s_dd[j] - s_pd[j],
                    "s_PE_PO": s_pp[j] - s_dp[j],
                    "w_PE_DO": w_dd[j] - w_pd[j],
                    "w_PE_PO": w_pp[j] - w_dp[j]
                })

    # Save Results
    res_df = pd.DataFrame(results)
    save_path = os.path.join(OUTPUT_DIR, f"scores_{filename}")
    res_df.to_csv(save_path, index=False)
    print(f"Saved scores to {save_path}")

    # Quick Stats
    print("Mean s-PE:", res_df.mean().to_dict())