import torch
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm.notebook import tqdm
import os

# --- CONFIGURATION ---
MODEL_NAME = "gpt2-large"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Files to process (Including Hybrid)
FILES = [
    # TRANSITIVE
    "/content/Geometry-of-Syntax/corpora/transitive/ANOMALOUS_chomsky_transitive_15000sampled_10-1.csv",
    "/content/Geometry-of-Syntax/corpora/transitive/CORE_transitive_15000sampled_10-1.csv",
    "/content/Geometry-of-Syntax/corpora/transitive/jabberwocky_transitive.csv",
    # DITRANSITIVE
    "/content/Geometry-of-Syntax/corpora/dative/ANOMALOUS_chomsky_dative_15000sampled_10-1.csv",
    "/content/Geometry-of-Syntax/corpora/dative/CORE_dative_15000sampled_10-1.csv",
    "/content/Geometry-of-Syntax/corpora/dative/jabberwocky_dative.csv",
    "/content/Geometry-of-Syntax/corpora/dative/jabberwocky_hybrid_dative.csv"
]

OUTPUT_DIR = "results/results_behavioral_raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD MODEL ---
print(f"Loading {MODEL_NAME} on {DEVICE} (FP16)...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model.to(DEVICE)
model.eval()


def get_critical_index_relative(tokenizer, target_text, structure_type):
    """
    Calculates the token index of the critical word relative to the start of the target.
    """
    words = target_text.strip().split()

    if "transitive" in structure_type:
        word_idx = 2  # Verb/Aux
    elif "dative" in structure_type:
        word_idx = 5  # Prep/Det
    else:
        word_idx = 2

    prefix_str = " " + " ".join(words[:word_idx])
    prefix_tokens = tokenizer.encode(prefix_str)
    return len(prefix_tokens)


def calculate_log_probs(batch_primes, batch_targets, structure_type):
    """
    Returns w-PE log probabilities for s-PE (sum) and w-PE (critical token).
    """
    full_texts = [p + " " + t for p, t in zip(batch_primes, batch_targets)]
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs.input_ids[:, 1:].contiguous()
    log_probs_all = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    target_log_probs = log_probs_all.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

    s_scores, w_scores = [], []

    for i in range(len(batch_primes)):
        prime_len = len(tokenizer.encode(batch_primes[i]))
        target_len = len(tokenizer.encode(" " + batch_targets[i]))

        start_idx = prime_len - 1
        end_idx = min(start_idx + target_len, int(inputs.attention_mask[i].sum() - 1))

        # s-PE Score (Sum)
        s_scores.append(target_log_probs[i, start_idx:end_idx].sum().item())

        # w-PE Score (Token)
        offset = get_critical_index_relative(tokenizer, batch_targets[i], structure_type)
        crit_idx = start_idx + offset

        if crit_idx < end_idx:
            w_scores.append(target_log_probs[i, crit_idx].item())
        else:
            w_scores.append(float('nan'))

    return s_scores, w_scores


# --- MAIN LOOP ---
for filename in FILES:
    if not os.path.exists(filename):
        print(f"Skipping {filename} (not found)")
        continue

    print(f"Processing {filename}...")
    df = pd.read_csv(filename)
    # Normalize columns
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {'p_do': 'pdo', 'p_po': 'ppo', 't_do': 'tdo', 't_po': 'tpo'}
    df.rename(columns=rename_map, inplace=True)

    structure_type = "transitive" if "transitive" in filename else "dative"
    results = []

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
                    # Raw Scores (Active Target)
                    "s_cong_Active": s_aa[j], "s_inc_Active": s_ba[j],
                    "w_cong_Active": w_aa[j], "w_inc_Active": w_ba[j],

                    # Raw Scores (Passive Target)
                    "s_cong_Passive": s_bb[j], "s_inc_Passive": s_ab[j],
                    "w_cong_Passive": w_bb[j], "w_inc_Passive": w_ab[j],

                    # Calculated PE (for convenience)
                    "s_PE_Active": s_aa[j] - s_ba[j],
                    "s_PE_Passive": s_bb[j] - s_ab[j]
                })

        else:  # Dative
            p_do, p_po = batch['pdo'].tolist(), batch['ppo'].tolist()
            t_do, t_po = batch['tdo'].tolist(), batch['tpo'].tolist()

            # 1. Target DO
            s_dd, w_dd = calculate_log_probs(p_do, t_do, structure_type)  # Congruent
            s_pd, w_pd = calculate_log_probs(p_po, t_do, structure_type)  # Incongruent

            # 2. Target PO
            s_dp, w_dp = calculate_log_probs(p_do, t_po, structure_type)  # Incongruent
            s_pp, w_pp = calculate_log_probs(p_po, t_po, structure_type)  # Congruent

            for j in range(len(batch)):
                results.append({
                    # Raw Scores (DO Target)
                    "s_cong_DO": s_dd[j], "s_inc_DO": s_pd[j],
                    "w_cong_DO": w_dd[j], "w_inc_DO": w_pd[j],

                    # Raw Scores (PO Target)
                    "s_cong_PO": s_pp[j], "s_inc_PO": s_dp[j],
                    "w_cong_PO": w_pp[j], "w_inc_PO": w_dp[j],

                    # Calculated PE
                    "s_PE_DO": s_dd[j] - s_pd[j],
                    "s_PE_PO": s_pp[j] - s_dp[j]
                })

    # Save Results
    res_df = pd.DataFrame(results)
    base_name = os.path.basename(filename)
    save_path = os.path.join(OUTPUT_DIR, f"scores_raw_{base_name}")
    res_df.to_csv(save_path, index=False)
    print(f"Saved w-PE scores to {save_path}")