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

# Files to process
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

    # CAUSATIVE / ERGATIVE
    "/content/Geometry-of-Syntax/corpora/transitive/CORE_causative.csv",
    "/content/Geometry-of-Syntax/corpora/transitive/jabberwocky_causative.csv",
    "/content/Geometry-of-Syntax/corpora/transitive/jabberwocky_ergative.csv"  # Just in case
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


def get_structure_type(filename):
    if "causative" in filename or "ergative" in filename:
        return "causative"
    elif "dative" in filename:
        return "dative"
    elif "transitive" in filename:
        return "transitive"
    return "unknown"


def get_critical_index_relative(tokenizer, target_text, structure_type):
    words = target_text.strip().split()
    if structure_type == "transitive":
        word_idx = 2
    elif structure_type == "dative":
        word_idx = 5
    elif structure_type == "causative":
        word_idx = 2
    else:
        word_idx = 2
    prefix_str = " " + " ".join(words[:word_idx])
    prefix_tokens = tokenizer.encode(prefix_str)
    return len(prefix_tokens)


def calculate_log_probs(batch_primes, batch_targets, structure_type):
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

        s_scores.append(target_log_probs[i, start_idx:end_idx].sum().item())

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
        continue

    print(f"Processing {os.path.basename(filename)}...")
    df = pd.read_csv(filename)

    # --- ROBUST COLUMN NORMALIZATION ---
    df.columns = df.columns.str.strip().str.lower()

    # Map everything to standard names
    rename_map = {
        'p_do': 'pdo', 'p_po': 'ppo', 't_do': 'tdo', 't_po': 'tpo',
        'p_trans': 'p_caus', 'p_intrans': 'p_inch',  # Handle "Ergative" naming
        't_trans': 't_caus', 't_intrans': 't_inch'
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

            s_aa, w_aa = calculate_log_probs(pa, ta, structure_type)
            s_ba, w_ba = calculate_log_probs(pp, ta, structure_type)
            s_ab, w_ab = calculate_log_probs(pa, tp, structure_type)
            s_bb, w_bb = calculate_log_probs(pp, tp, structure_type)

            for j in range(len(batch)):
                results.append({
                    "s_cong_Active": s_aa[j], "s_inc_Active": s_ba[j],
                    "w_cong_Active": w_aa[j], "w_inc_Active": w_ba[j],
                    "s_cong_Passive": s_bb[j], "s_inc_Passive": s_ab[j],
                    "w_cong_Passive": w_bb[j], "w_inc_Passive": w_ab[j],
                    "s_PE_Active": s_aa[j] - s_ba[j],
                    "s_PE_Passive": s_bb[j] - s_ab[j]
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
                    "s_cong_DO": s_dd[j], "s_inc_DO": s_pd[j],
                    "w_cong_DO": w_dd[j], "w_inc_DO": w_pd[j],
                    "s_cong_PO": s_pp[j], "s_inc_PO": s_dp[j],
                    "w_cong_PO": w_pp[j], "w_inc_PO": w_dp[j],
                    "s_PE_DO": s_dd[j] - s_pd[j],
                    "s_PE_PO": s_pp[j] - s_dp[j]
                })

        # --- CAUSATIVE ---
        elif structure_type == "causative":
            p_c = batch['p_caus'].tolist()
            p_i = batch['p_inch'].tolist()

            # Handle Ambiguous Target (Jabberwocky) vs Separate Targets (Core)
            if 't_ambiguous' in batch.columns:
                # Jabberwocky: One target for both
                t_amb = batch['t_ambiguous'].tolist()
                s_cc, w_cc = calculate_log_probs(p_c, t_amb, structure_type)  # Congruent
                s_ic, w_ic = calculate_log_probs(p_i, t_amb, structure_type)  # Incongruent

                for j in range(len(batch)):
                    results.append({
                        "s_cong_Caus": s_cc[j], "s_inc_Caus": s_ic[j],
                        "w_cong_Caus": w_cc[j], "w_inc_Caus": w_ic[j],
                        "s_PE_Causative": s_cc[j] - s_ic[j],
                        # Inchoative is N/A for ambiguous target
                        "s_PE_Inchoative": np.nan
                    })
            else:
                # Core: Separate targets
                t_c = batch['t_caus'].tolist()
                t_i = batch['t_inch'].tolist()

                s_cc, w_cc = calculate_log_probs(p_c, t_c, structure_type)
                s_ic, w_ic = calculate_log_probs(p_i, t_c, structure_type)
                s_ii, w_ii = calculate_log_probs(p_i, t_i, structure_type)
                s_ci, w_ci = calculate_log_probs(p_c, t_i, structure_type)

                for j in range(len(batch)):
                    results.append({
                        "s_cong_Caus": s_cc[j], "s_inc_Caus": s_ic[j],
                        "w_cong_Caus": w_cc[j], "w_inc_Caus": w_ic[j],
                        "s_cong_Inch": s_ii[j], "s_inc_Inch": s_ci[j],
                        "w_cong_Inch": w_ii[j], "w_inc_Inch": w_ci[j],
                        "s_PE_Causative": s_cc[j] - s_ic[j],
                        "s_PE_Inchoative": s_ii[j] - s_ci[j]
                    })

    # Save Results
    res_df = pd.DataFrame(results)
    base_name = os.path.basename(filename)
    save_path = os.path.join(OUTPUT_DIR, f"scores_raw_{base_name}")
    res_df.to_csv(save_path, index=False)
    print(f"Saved raw scores to {save_path}")