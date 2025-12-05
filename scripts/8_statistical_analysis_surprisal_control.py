import pandas as pd
import statsmodels.formula.api as smf
import os
import numpy as np

# --- CONFIGURATION ---
RESULTS_DIR = "results/results_behavioral"
CORPORA_DIR = "corpora"

# File Map (Same as before)
FILE_MAP = [
    ("Core", "corpora/transitive/CORE_transitive_15000sampled_10-1.csv",
     "corpora/dative/CORE_dative_15000sampled_10-1.csv"),
    ("Anomalous", "corpora/transitive/ANOMALOUS_chomsky_transitive_15000sampled_10-1.csv",
     "corpora/dative/ANOMALOUS_chomsky_dative_15000sampled_10-1.csv"),
    ("Jabberwocky", "corpora/transitive/jabberwocky_transitive.csv", "corpora/dative/jabberwocky_dative.csv")
]


def extract_verb(sentence):
    try:
        return sentence.strip().split()[2]
    except:
        return "unknown"


def prepare_data_with_surprisal(structure_type):
    long_data = []
    print(f"Preparing data for {structure_type}...")

    for condition, trans_path, dat_path in FILE_MAP:
        corpus_path = trans_path if structure_type == "Transitive" else dat_path
        if not os.path.exists(corpus_path): continue

        df_corpus = pd.read_csv(corpus_path)
        df_corpus.columns = df_corpus.columns.str.strip()

        base_name = os.path.basename(corpus_path)
        score_path = os.path.join(RESULTS_DIR, f"scores_{base_name}")
        if not os.path.exists(score_path): continue

        df_scores = pd.read_csv(score_path)

        # Merge
        min_len = min(len(df_corpus), len(df_scores))
        df = pd.concat([df_corpus.iloc[:min_len].reset_index(drop=True),
                        df_scores.iloc[:min_len].reset_index(drop=True)], axis=1)

        # CALCULATE SURPRISAL
        # Surprisal = -1 * LogProb(Target | Incongruent Prime)
        # This is the "Baseline Difficulty" of the sentence

        if structure_type == "Transitive":
            if 'ta' not in df.columns: continue

            # Active Rows (Baseline is Passive Prime -> Active Target)
            # s_ba is LogProb(TargetActive | PrimePassive)
            # Note: We need to reconstruct this from the PE if not saved directly.
            # Wait, the previous script saved s_aa, s_ba? No, it saved PE.
            # We need to re-run extraction OR approximate.
            # Actually, s_PE = s_cong - s_inc.
            # We don't have raw s_inc in the saved CSV.

            # CRITICAL FIX: We need the raw scores to calculate surprisal.
            # If you didn't save raw s_aa, s_ba in the CSV, we can't do this exact analysis
            # without re-running the behavioral script to save raw columns.

            # ASSUMING you can re-run behavioral script to save raw columns:
            # Let's assume df has 's_logprob_cong' and 's_logprob_inc'
            # For now, I will write the code assuming we have them.
            # IF NOT, YOU MUST MODIFY 2_behavioral_priming.py TO SAVE RAW SCORES.
            pass

    return pd.DataFrame()