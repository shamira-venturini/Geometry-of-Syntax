import random
import csv

# --- CONFIGURATION ---
OUTPUT_FILE = "../corpora/dative/jabberwocky_dative.csv"
NUM_SAMPLES = 15000
random.seed(42)

# --- VOCABULARY ---
NOUNS = [
    "wug", "dax", "fep", "blicket", "glorp", "tove", "slithy", "borogove",
    "mome", "rath", "jubjub", "bandersnatch", "tumtum", "tulgey", "frabjous",
    "snark", "boojum", "vorpal", "manxome", "uffish", "whiffling", "burbled",
    "snicker", "galumph", "beamish", "frumious", "gimble", "mimsy", "kiki",
    "bouba", "zorp", "quib", "narp", "vlim", "crunk", "dril", "frob", "zib"
]

# Verbs (Past Tense forms)
VERBS = [
    "gimbled", "gorped", "whiffled", "burbled", "snickered", "galumphed",
    "pilked", "strugged", "yomped", "fazzed", "quilliged", "chortled",
    "outgrabed", "crinked", "blemed", "porked", "zibbed", "glorked"
]


def generate_dative_row():
    random.shuffle(NOUNS)
    random.shuffle(VERBS)

    # 1. Lexical Disjointness (Constraint 1)
    # Prime: Indices 0,1,2. Target: Indices 3,4,5.
    p_agent, p_recip, p_theme = NOUNS[0], NOUNS[1], NOUNS[2]
    t_agent, t_recip, t_theme = NOUNS[3], NOUNS[4], NOUNS[5]

    p_verb = VERBS[0]
    t_verb = VERBS[1]

    # 2. Determiner Control (Constraint 3)
    if random.random() > 0.5:
        p_det, t_det = "the", "a"
    else:
        p_det, t_det = "a", "the"

    # 3. Function Word Control (Constraint 2: Preposition)
    # If Prime uses "to", Target uses "for" (and vice versa)
    if random.random() > 0.5:
        p_prep, t_prep = "to", "for"
    else:
        p_prep, t_prep = "for", "to"

    # 4. Construct Sentences
    # Prime DO: Agent V Recipient Theme
    p_do = f"{p_det} {p_agent} {p_verb} {p_det} {p_recip} {p_det} {p_theme} ."
    # Prime PO: Agent V Theme Prep Recipient
    p_po = f"{p_det} {p_agent} {p_verb} {p_det} {p_theme} {p_prep} {p_det} {p_recip} ."

    # Target DO
    t_do = f"{t_det} {t_agent} {t_verb} {t_det} {t_recip} {t_det} {t_theme} ."
    # Target PO
    t_po = f"{t_det} {t_agent} {t_verb} {t_det} {t_theme} {t_prep} {t_det} {t_recip} ."

    return [p_do, p_po, t_do, t_po]


# --- EXECUTION ---
print(f"Generating {NUM_SAMPLES} dative Jabberwocky pairs...")
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["p_do", "p_po", "t_do", "t_po"])
    for _ in range(NUM_SAMPLES):
        writer.writerow(generate_dative_row())
print("Done.")