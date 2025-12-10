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
    "/content/Geometry-of-Syntax/corpora/transitive/ANOMALOUS_chomsky_transitive_15000sampled_10-1.csv",
    "/content/Geometry-of-Syntax/corpora/transitive/CORE_transitive_15000sampled_10-1.csv",
    "/content/Geometry-of-Syntax/corpora/transitive/jabberwocky_transitive.csv",
    # DITRANSITIVE
    "/content/Geometry-of-Syntax/corpora/dative/ANOMALOUS_chomsky_dative_15000sampled_10-1.csv",
    "/content/Geometry-of-Syntax/corpora/dative/CORE_dative_15000sampled_10-1.csv",
    "/content/Geometry-of-Syntax/corpora/dative/jabberwocky_dative.csv"
]

OUTPUT_DIR = "results/results_behavioral"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD MODEL ---
print(f"Loading {MODEL_NAME} on {DEVICE} (FP16)...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load in FP16 for speed/memory efficiency
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model.to(DEVICE)
model.eval()


def get_critical_index_relative(tokenizer, target_text, structure_type):
    """
    Calculates the token index of the critical word relative to the start of the target.
    Strategy: Split sentence into words, identify critical word position,
    tokenize the prefix, and count tokens.
    """
    # 1. Split into words (simple space split works for this clean corpus)
    words = target_text.strip().split()

    # 2. Identify which word number is critical (0-indexed)
    if "transitive" in structure_type:
        # Template: "The(0) dax(1) [gimbled](2)..." or "The(0) wug(1) [was](2)..."
        word_idx = 2
    elif "dative" in structure_type:
        # Template: "The(0) dax(1) gimbled(2) the(3) wug(4) [to](5)..."
        # Template: "The(0) dax(1) gimbled(2) the(3) wug(4) [the](5)..."
        word_idx = 5
    else:
        word_idx = 2  # Fallback

    # 3. Reconstruct the prefix (words BEFORE the critical one)
    # We add the leading space back because the tokenizer needs to know it's not start-of-sentence
    prefix_str = " " + " ".join(words[:word_idx])

    # 4. Count tokens in the prefix
    # The critical token is the NEXT token after this prefix
    prefix_tokens = tokenizer.encode(prefix_str)
    return len(prefix_tokens)


def calculate_log_probs(batch_primes, batch_targets, structure_type):
    """
    Calculates s-PE (sum of target log probs) and w-PE (critical token log prob).
    """
    # 1. Prepare Inputs (Prime + Space + Target)
    full_texts = [p + " " + t for p, t in zip(batch_primes, batch_targets)]

    # Tokenize
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)

    # 2. Run Model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [Batch, Seq, Vocab]

    # 3. Shift Logits for Prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs.input_ids[:, 1:].contiguous()

    # Calculate Log Softmax
    log_probs_all = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Gather the log prob of the actual token that appeared
    target_log_probs = log_probs_all.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

    # 4. Extract Specific Scores
    s_scores = []
    w_scores = []

    for i in range(len(batch_primes)):
        # Find where the target starts
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

        # w-PE: Critical Token (DYNAMIC CALCULATION)
        # We calculate the offset specifically for THIS sentence
        offset = get_critical_index_relative(tokenizer, batch_targets[i], structure_type)
        crit_idx = start_idx + offset

        if crit_idx < end_idx:
            token_score = target_log_probs[i, crit_idx].item()
        else:
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
    df.columns = df.columns.str.strip()
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


        else:  # dative

            pdo, ppo = batch['pdo'].tolist(), batch['ppo'].tolist()
            tdo, tpo = batch['tdo'].tolist(), batch['tpo'].tolist()

            # 1. Target DO
            s_dd, w_dd = calculate_log_probs(pdo, tdo, structure_type)  # Congruent
            s_pd, w_pd = calculate_log_probs(ppo, tdo, structure_type)  # Incongruent

            # 2. Target PO

            s_dp, w_dp = calculate_log_probs(pdo, tpo, structure_type)  # Incongruent

            s_pp, w_pp = calculate_log_probs(ppo, tpo, structure_type)  # Congruent

            for j in range(len(batch)):
                results.append({

                    "s_PE_DO": s_dd[j] - s_pd[j],

                    "s_PE_PO": s_pp[j] - s_dp[j],

                    "w_PE_DO": w_dd[j] - w_pd[j],

                    "w_PE_PO": w_pp[j] - w_dp[j]

                })

    # Save Results
    res_df = pd.DataFrame(results)
    base_name = os.path.basename(filename)
    save_path = os.path.join(OUTPUT_DIR, f"scores_{base_name}")
    res_df.to_csv(save_path, index=False)
    print(f"Saved scores to {save_path}")

    # Quick Stats
    print("Mean s-PE:", res_df.mean().to_dict())