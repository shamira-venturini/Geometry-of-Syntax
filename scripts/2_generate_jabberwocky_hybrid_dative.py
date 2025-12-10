import random
import csv

# --- CONFIGURATION ---
OUTPUT_FILE = "../corpora/dative/jabberwocky_hybrid_dative.csv"
NUM_SAMPLES = 15000
random.seed(42)

# --- VOCABULARY ---
NOUNS = ["wug", "dax", "fep", "blicket", "glorp", "tove", "slithy", "borogove", "mome", "rath", "jubjub",
         "bandersnatch", "tumtum", "tulgey", "frabjous", "snark", "boojum", "vorpal", "manxome", "uffish", "whiffling",
         "burbled", "snicker", "galumph", "beamish", "frumious", "gimble", "mimsy", "kiki", "bouba", "zorp", "quib",
         "narp", "vlim", "crunk", "dril", "frob", "zib"]
VERBS = ["gimbled", "gorped", "whiffled", "burbled", "snickered", "galumphed", "pilked", "strugged", "yomped", "fazzed",
         "quilliged", "chortled", "outgrabed", "crinked", "blemed", "porked", "zibbed", "glorked"]

# Animate Pronouns (The "Semantic Scaffold")
PRONOUNS = ["him", "her", "them", "us", "me", "you"]


def generate_hybrid_row():
    random.shuffle(NOUNS)
    random.shuffle(VERBS)

    # Prime: Standard Jabberwocky (No Animacy)
    # We want to see if the TARGET benefits from Animacy, regardless of Prime.
    # Actually, to test scaffolding, we should probably make BOTH Prime and Target Hybrid
    # to maximize the priming signal.

    p_agent, p_theme = NOUNS[0], NOUNS[1]
    t_agent, t_theme = NOUNS[2], NOUNS[3]

    p_recip = random.choice(PRONOUNS)
    t_recip = random.choice(PRONOUNS)

    p_verb = VERBS[0]
    t_verb = VERBS[1]

    # Determiners (Only for Agent/Theme, Pronouns don't take dets)
    if random.random() > 0.5:
        p_det, t_det = "the", "a"
    else:
        p_det, t_det = "a", "the"

    # Preposition Flip
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
print(f"Generating {NUM_SAMPLES} Hybrid Jabberwocky pairs...")
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["p_do", "p_po", "t_do", "t_po"])
    for _ in range(NUM_SAMPLES):
        writer.writerow(generate_hybrid_row())
print("Done.")