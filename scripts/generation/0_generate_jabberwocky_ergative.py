import random
import csv
import os

# --- CONFIGURATION ---
OUTPUT_DIR = "/corpora/ergative"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "jabberwocky_ergative.csv")
NUM_SAMPLES = 15000
random.seed(42)

# --- SANITIZED VOCABULARY ---
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
    "outgrabed", "crinked", "blemed", "porked", "zibbed", "glorked"
]


def generate_row():
    random.shuffle(NOUNS)
    random.shuffle(VERBS)

    # Prime Nouns
    p_n1, p_n2 = NOUNS[0], NOUNS[1]
    p_v = VERBS[0]

    # Target Nouns
    # We use the SAME subject for the ambiguous target to test the structure
    t_subj = NOUNS[2]
    t_obj = NOUNS[3]
    t_v = VERBS[1]

    # Determiners (Flip)
    p_det = "the" if random.random() > 0.5 else "a"
    t_det = "a" if p_det == "the" else "the"

    # --- SENTENCES ---

    # Prime Causative: "The dax gimbled the wug."
    p_caus = f"{p_det} {p_n1} {p_v} {p_det} {p_n2} ."

    # Prime Inchoative: "The wug gimbled."
    # (Patient becomes Subject)
    p_inch = f"{p_det} {p_n2} {p_v} ."

    # Target (Ambiguous Start): "The fep pilked..."
    # We provide the full Causative sentence as the target string for extraction.
    # The extraction script will look at the verb "pilked".
    # If primed by Inchoative, the model expects "." after "pilked".
    # If primed by Causative, the model expects "the" (object) after "pilked".
    t_ambiguous = f"{t_det} {t_subj} {t_v} {t_det} {t_obj} ."

    return [p_caus, p_inch, t_ambiguous]


# --- EXECUTION ---
print(f"Generating {NUM_SAMPLES} Clean Jabberwocky Causative pairs...")
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    # Headers
    writer.writerow(["p_caus", "p_inch", "t_ambiguous"])
    for _ in range(NUM_SAMPLES):
        writer.writerow(generate_row())
print("Done.")