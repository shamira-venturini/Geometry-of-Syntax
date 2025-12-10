import random
import csv
import os

OUTPUT_DIR = "/corpora/transitive"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "jabberwocky_transitive.csv")
NUM_SAMPLES = 15000
random.seed(42)

# --- STRICTLY DISJOINT VOCABULARY ---
# Nouns (Entities)
NOUNS = [
    "wug", "dax", "fep", "blicket", "glorp", "tove", "slithy", "borogove",
    "mome", "rath", "jubjub", "bandersnatch", "tumtum", "tulgey", "frabjous",
    "snark", "boojum", "vorpal", "manxome", "uffish", "kiki", "bouba",
    "zorp", "quib", "narp", "vlim", "crunk", "dril", "frob", "zib"
]

# Verbs (Actions) - NO overlap with Nouns
VERBS = [
    ("gimbles", "gimbled"), ("gorps", "gorped"), ("whiffles", "whiffled"),
    ("burbles", "burbled"), ("snickers", "snickered"), ("galumphs", "galumphed"),
    ("pilks", "pilked"), ("strugs", "strugged"), ("yomps", "yomped"),
    ("fazzes", "fazzed"), ("quilligs", "quilliged"), ("chortles", "chortled"),
    ("outgrabes", "outgrabed"), ("crinks", "crinked"), ("blems", "blemed"),
    ("porks", "porked"), ("glorks", "glorked")
]


def generate_row():
    random.shuffle(NOUNS)
    random.shuffle(VERBS)

    p_n1, p_n2 = NOUNS[0], NOUNS[1]
    t_n1, t_n2 = NOUNS[2], NOUNS[3]

    p_v = VERBS[0]
    t_v = VERBS[1]

    # Tense Flip
    if random.random() > 0.5:
        p_aux, p_va, p_vp = "is", p_v[0], p_v[1]
        t_aux, t_va, t_vp = "was", t_v[1], t_v[1]
    else:
        p_aux, p_va, p_vp = "was", p_v[1], p_v[1]
        t_aux, t_va, t_vp = "is", t_v[0], t_v[1]

    # Det Flip
    p_det, t_det = ("the", "a") if random.random() > 0.5 else ("a", "the")

    pa = f"{p_det} {p_n1} {p_va} {p_det} {p_n2} ."
    pp = f"{p_det} {p_n2} {p_aux} {p_vp} by {p_det} {p_n1} ."
    ta = f"{t_det} {t_n1} {t_va} {t_det} {t_n2} ."
    tp = f"{t_det} {t_n2} {t_aux} {t_vp} by {t_det} {t_n1} ."

    return [pa, pp, ta, tp]


with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["pa", "pp", "ta", "tp"])
    for _ in range(NUM_SAMPLES):
        writer.writerow(generate_row())
print("Clean Transitive Generated.")