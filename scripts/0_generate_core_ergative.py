import random
import csv
import os

# --- CONFIGURATION ---
OUTPUT_DIR = "corpora/transitive"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "jabberwocky_ergative.csv")  # Renamed to 'ergative'
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
    p_agent, p_patient = NOUNS[0], NOUNS[1]
    p_v = VERBS[0]

    # Target Nouns (Disjoint from Prime)
    t_agent, t_patient = NOUNS[2], NOUNS[3]
    t_v = VERBS[1]

    # Determiners (Flip)
    p_det = "the" if random.random() > 0.5 else "a"
    t_det = "a" if p_det == "the" else "the"

    # --- SENTENCES ---

    # Prime Transitive: "The dax gimbled the wug."
    p_trans = f"{p_det} {p_agent} {p_v} {p_det} {p_patient} ."

    # Prime Intransitive: "The wug gimbled."
    # (Patient becomes Subject, mirroring Core structure)
    p_intrans = f"{p_det} {p_patient} {p_v} ."

    # Target Transitive
    t_trans = f"{t_det} {t_agent} {t_v} {t_det} {t_patient} ."

    # Target Intransitive
    t_intrans = f"{t_det} {t_patient} {t_v} ."

    return [p_trans, p_intrans, t_trans, t_intrans]


# --- EXECUTION ---
print(f"Generating {NUM_SAMPLES} Symmetric Jabberwocky Ergative pairs...")
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    # Headers match the logic: Transitive / Intransitive
    writer.writerow(["p_trans", "p_intrans", "t_trans", "t_intrans"])
    for _ in range(NUM_SAMPLES):
        writer.writerow(generate_row())
print("Done.")