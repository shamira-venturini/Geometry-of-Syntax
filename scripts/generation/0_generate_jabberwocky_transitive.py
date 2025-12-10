import random
import csv

<<<<<<< HEAD:scripts/generation/0_generate_jabberwocky_transitive.py
OUTPUT_DIR = "/corpora/transitive"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "jabberwocky_transitive.csv")
=======
# --- CONFIGURATION ---
OUTPUT_FILE = "../corpora/transitive/jabberwocky_transitive.csv"
>>>>>>> parent of c39105d (fixed generation script with vocabulary control):scripts/0_generate_jabberwocky_transitive.py
NUM_SAMPLES = 15000
random.seed(42)  # Reproducibility (Phase 5)

# --- VOCABULARY ---
# Nouns (Singular)
NOUNS = [
    "wug", "dax", "fep", "blicket", "glorp", "tove", "slithy", "borogove",
    "mome", "rath", "jubjub", "bandersnatch", "tumtum", "tulgey", "frabjous",
    "snark", "boojum", "vorpal", "manxome", "uffish", "whiffling", "burbled",
    "snicker", "galumph", "beamish", "frumious", "gimble", "mimsy", "kiki",
    "bouba", "zorp", "quib", "narp", "vlim", "crunk", "dril", "frob"
]

# Verbs: (Present_3rd_Person, Past_Participle)
# e.g., "The wug [gorps]..." vs "The wug was [gorped]..."
VERBS = [
    ("gorps", "gorped"), ("chortles", "chortled"), ("gyres", "gyred"),
    ("gimbles", "gimbled"), ("outgrabes", "outgrabed"), ("whiffles", "whiffled"),
    ("burbles", "burbled"), ("snickers", "snickered"), ("galumphs", "galumphed"),
    ("crinks", "crinked"), ("blems", "blemed"), ("pilks", "pilked"),
    ("strugs", "strugged"), ("yomps", "yomped"), ("fazzes", "fazzed"),
    ("quilligs", "quilliged")
]


def generate_transitive_row():
    # 1. Shuffle Vocabulary
    random.shuffle(NOUNS)
    random.shuffle(VERBS)

    # 2. Lexical Disjointness (Constraint 1)
    # Prime uses indices 0,1; Target uses 2,3
    p_n1, p_n2 = NOUNS[0], NOUNS[1]
    t_n1, t_n2 = NOUNS[2], NOUNS[3]

    p_verb_tuple = VERBS[0]
    t_verb_tuple = VERBS[1]

    # 3. Function Word Control (Constraint 2: Tense/Aux)
    # Randomly decide if Prime is Present or Past
    if random.random() > 0.5:
        # Prime = Present ("is"), Target = Past ("was")
        p_aux = "is"
        p_v_act = p_verb_tuple[0]  # "gorps"
        p_v_pass = p_verb_tuple[1]  # "gorped"

        t_aux = "was"
        t_v_act = t_verb_tuple[1]  # Past tense active ("gorped")
        t_v_pass = t_verb_tuple[1]  # Past participle ("gorped")
    else:
        # Prime = Past ("was"), Target = Present ("is")
        p_aux = "was"
        p_v_act = p_verb_tuple[1]
        p_v_pass = p_verb_tuple[1]

        t_aux = "is"
        t_v_act = t_verb_tuple[0]
        t_v_pass = t_verb_tuple[1]

    # 4. Determiner Control (Constraint 3)
    if random.random() > 0.5:
        p_det, t_det = "the", "a"
    else:
        p_det, t_det = "a", "the"

    # 5. Construct Sentences
    # Prime Active (PA)
    pa = f"{p_det} {p_n1} {p_v_act} {p_det} {p_n2} ."
    # Prime Passive (PP)
    pp = f"{p_det} {p_n2} {p_aux} {p_v_pass} by {p_det} {p_n1} ."

    # Target Active (TA)
    ta = f"{t_det} {t_n1} {t_v_act} {t_det} {t_n2} ."
    # Target Passive (TP)
    tp = f"{t_det} {t_n2} {t_aux} {t_v_pass} by {t_det} {t_n1} ."

    return [pa, pp, ta, tp]


# --- EXECUTION ---
print(f"Generating {NUM_SAMPLES} Transitive Jabberwocky pairs...")
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["pa", "pp", "ta", "tp"])
    for _ in range(NUM_SAMPLES):
        writer.writerow(generate_transitive_row())
print("Done.")