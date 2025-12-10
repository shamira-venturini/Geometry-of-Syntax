import torch
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import os

# --- CONFIGURATION ---
MODEL_NAME = "gpt2-large"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MASTER FILE LIST
FILES = [
    # TRANSITIVE
    "/content/Geometry-of-Syntax/corpora/transitive/ANOMALOUS_chomsky_transitive_15000sampled_10-1.csv",
    "/content/Geometry-of-Syntax/corpora/transitive/CORE_transitive_15000sampled_10-1.csv",
    "/content/Geometry-of-Syntax/corpora/transitive/jabberwocky_transitive.csv",

    # DATIVE
    "/content/Geometry-of-Syntax/corpora/dative/ANOMALOUS_chomsky_dative_15000sampled_10-1.csv",
    "/content/Geometry-of-Syntax/corpora/dative/CORE_dative_15000sampled_10-1.csv",
    "/content/Geometry-of-Syntax/corpora/dative/jabberwocky_dative.csv",
    "/content/Geometry-of-Syntax/corpora/dative/jabberwocky_hybrid_dative.csv",

    # ERGATIVE (Causative)
    "/content/Geometry-of-Syntax/corpora/ergative/CORE_ergative.csv",
    "/content/Geometry-of-Syntax/corpora/ergative/jabberwocky_ergative.csv"
]

OUTPUT_DIR = "behavioral_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD MODEL ---
print(f"Loading {MODEL_NAME} on {DEVICE} (FP16)...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model.to(DEVICE)
model.eval()


def get_structure_type(filename):
    if "causative" in filename or "ergative" in filename:
        return "ergative"
    elif "dative" in filename:
        return "dative"
    elif "transitive" in filename:
        return "transitive"
    return "unknown"


def get_critical_index_relative(tokenizer, target_text, structure_type):
    """Calculates token index of critical word relative to target start."""
    words = target_text.strip().split()

    if structure_type == "transitive":
        word_idx = 2  # Verb/Aux
    elif structure_type == "dative":
        word_idx = 5  # Prep/Det
    elif structure_type == "ergative":
        word_idx = 2  # Verb
    else:
        word_idx = 2

    prefix_str = " " + " ".join(words[:word_idx])
    prefix_tokens = tokenizer.encode(prefix_str)
    return len(prefix_tokens)


def calculate_log_probs(batch_primes, batch_targets, structure_type):
    """Returns raw log probabilities for s-PE (sum) and w-PE (critical token)."""
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

    print(f"Processing {os.path.basename(filename)}...")
    df = pd.read_csv(filename)

    # --- ROBUST COLUMN NORMALIZATION ---
    df.columns = df.columns.str.strip().str.lower()

    # Map everything to standard internal names
    rename_map = {
        'p_do': 'pdo', 'p_po': 'ppo', 't_do': 'tdo', 't_po': 'tpo',
        'p_caus': 'ptrans', 'p_inch': 'pintrans', 'p_trans': 'ptrans', 'p_intrans': 'pintrans',
        't_caus': 'ttrans', 't_inch': 'tintrans', 't_trans': 'ttrans', 't_intrans': 'tintrans'
    }
    df.rename(columns=rename_map, inplace=True)

    structure_type = get_structure_type(filename)
    results = []

    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[i: i + BATCH_SIZE]

        # --- TRANSITIVE ---
        if structure_type == "transitive":
            pa, pp = batch['pa'].tolist(), batch['pp'].tolist()
            ta, tp = batch['ta'].tolist(), batch['tp'].tolist()

            # 1. Target Active
            s_aa, w_aa = calculate_log_probs(pa, ta, structure_type)  # Congruent
            s_ba, w_ba = calculate_log_probs(pp, ta, structure_type)  # Incongruent

            # 2. Target Passive
            s_ab, w_ab = calculate_log_probs(pa, tp, structure_type)  # Incongruent
            s_bb, w_bb = calculate_log_probs(pp, tp, structure_type)  # Congruent

            for j in range(len(batch)):
                results.append({
                    # Raw
                    "s_cong_Active": s_aa[j], "s_inc_Active": s_ba[j],
                    "w_cong_Active": w_aa[j], "w_inc_Active": w_ba[j],
                    "s_cong_Passive": s_bb[j], "s_inc_Passive": s_ab[j],
                    "w_cong_Passive": w_bb[j], "w_inc_Passive": w_ab[j],
                    # Aggregated
                    "s_PE_Active": s_aa[j] - s_ba[j],
                    "s_PE_Passive": s_bb[j] - s_ab[j],
                    "w_PE_Active": w_aa[j] - w_ba[j],
                    "w_PE_Passive": w_bb[j] - w_ab[j]
                })

        # --- DATIVE ---
        elif structure_type == "dative":
            p_do, p_po = batch['pdo'].tolist(), batch['ppo'].tolist()
            t_do, t_po = batch['tdo'].tolist(), batch['tpo'].tolist()

            s_dd, w_dd = calculate_log_probs(p_do, t_do, structure_type)
            s_pd, w_pd = calculate_log_probs(p_po, t_do, structure_type)
            s_dp, w_dp = calculate_log_probs(p_do, t_po, structure_type)
            s_pp, w_pp = calculate_log_probs(p_po, t_po, structure_type)

            for j in range(len(batch)):
                results.append({
                    # Raw
                    "s_cong_DO": s_dd[j], "s_inc_DO": s_pd[j],
                    "w_cong_DO": w_dd[j], "w_inc_DO": w_pd[j],
                    "s_cong_PO": s_pp[j], "s_inc_PO": s_dp[j],
                    "w_cong_PO": w_pp[j], "w_inc_PO": w_dp[j],
                    # Aggregated
                    "s_PE_DO": s_dd[j] - s_pd[j],
                    "s_PE_PO": s_pp[j] - s_dp[j],
                    "w_PE_DO": w_dd[j] - w_pd[j],
                    "w_PE_PO": w_pp[j] - w_dp[j]
                })

        # --- ERGATIVE ---
        elif structure_type == "ergative":
            p_trans = batch['ptrans'].tolist()
            p_intrans = batch['pintrans'].tolist()
            t_trans = batch['ttrans'].tolist()
            t_intrans = batch['tintrans'].tolist()

            # 1. Target Transitive
            s_tt, w_tt = calculate_log_probs(p_trans, t_trans, structure_type)  # Congruent
            s_it, w_it = calculate_log_probs(p_intrans, t_trans, structure_type)  # Incongruent

            # 2. Target Intransitive
            s_ii, w_ii = calculate_log_probs(p_intrans, t_intrans, structure_type)  # Congruent
            s_ti, w_ti = calculate_log_probs(p_trans, t_intrans, structure_type)  # Incongruent

            for j in range(len(batch)):
                results.append({
                    # Raw
                    "s_cong_Trans": s_tt[j], "s_inc_Trans": s_it[j],
                    "w_cong_Trans": w_tt[j], "w_inc_Trans": w_it[j],
                    "s_cong_Intrans": s_ii[j], "s_inc_Intrans": s_ti[j],
                    "w_cong_Intrans": w_ii[j], "w_inc_Intrans": w_ti[j],
                    # Aggregated
                    "s_PE_Transitive": s_tt[j] - s_it[j],
                    "s_PE_Intransitive": s_ii[j] - s_ti[j],
                    "w_PE_Transitive": w_tt[j] - w_it[j],
                    "w_PE_Intransitive": w_ii[j] - w_ti[j]
                })

    # Save Results
    res_df = pd.DataFrame(results)
    base_name = os.path.basename(filename)
    save_path = os.path.join(OUTPUT_DIR, f"scores_master_{base_name}")
    res_df.to_csv(save_path, index=False)
    print(f"Saved master scores to {save_path}")