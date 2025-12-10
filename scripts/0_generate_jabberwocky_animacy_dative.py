import random
import csv
import os

# --- CONFIGURATION ---
OUTPUT_DIR = "corpora/dative"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "jabberwocky_hybrid_dative.csv")
NUM_SAMPLES = 15000
random.seed(42)

# --- SANITIZED VOCABULARY (Matches Standard Clean Corpus) ---
# Nouns (No overlap with verb roots)
NOUNS = [
    "wug", "dax", "fep", "blicket", "glorp", "tove", "slithy", "borogove",
    "mome", "rath", "jubjub", "bandersnatch", "tumtum", "tulgey", "frabjous",
    "snark", "boojum", "vorpal", "manxome", "uffish", "kiki", "bouba",
    "zorp", "quib", "narp", "vlim", "crunk", "dril", "frob", "zib"
]

# Verbs (Past Tense)
VERBS = [
    "gimbled", "gorped", "whiffled", "burbled", "snickered", "galumphed",
    "pilked", "strugged", "yomped", "fazzed", "quilliged", "chortled",
    "outgrabed", "crinked", "blemed", "porked", "glorked"
]

# Animate Pronouns (The "Semantic Scaffold")
PRONOUNS = ["him", "her", "them", "us", "me", "you"]


def generate_hybrid_row():
    random.shuffle(NOUNS)
    random.shuffle(VERBS)

    # 1. Select Nouns & Verbs (Disjoint)
    p_agent, p_theme = NOUNS[0], NOUNS[1]
    t_agent, t_theme = NOUNS[2], NOUNS[3]

    p_verb = VERBS[0]
    t_verb = VERBS[1]

    # 2. Select Pronouns (Disjoint)
    # CRITICAL FIX: Ensure Prime and Target use DIFFERENT pronouns
    # to avoid lexical priming of the pronoun itself.
    p_recip, t_recip = random.sample(PRONOUNS, 2)

    # 3. Determiners (Flip)
    if random.random() > 0.5:
        p_det, t_det = "the", "a"
    else:
        p_det, t_det = "a", "the"

    # 4. Preposition Flip (Strict Disjointness)
    if random.random() > 0.5:
        p_prep, t_prep = "to", "for"
    else:
        p_prep, t_prep = "for", "to"

    # --- Construct Sentences ---

    # DO: Agent V [Pronoun] Theme
    # Note: No determiner before pronoun
    p_do = f"{p_det} {p_agent} {p_verb} {p_recip} {p_det} {p_theme} ."
    t_do = f"{t_det} {t_agent} {t_verb} {t_recip} {t_det} {t_theme} ."

    # PO: Agent V Theme Prep [Pronoun]
    p_po = f"{p_det} {p_agent} {p_verb} {p_det} {p_theme} {p_prep} {p_recip} ."
    t_po = f"{t_det} {t_agent} {t_verb} {t_det} {t_theme} {t_prep} {t_recip} ."

    return [p_do, p_po, t_do, t_po]


# --- EXECUTION ---
print(f"Generating {NUM_SAMPLES} Clean Hybrid Jabberwocky pairs...")
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["p_do", "p_po", "t_do", "t_po"])
    for _ in range(NUM_SAMPLES):
        writer.writerow(generate_hybrid_row())
print("Done.")