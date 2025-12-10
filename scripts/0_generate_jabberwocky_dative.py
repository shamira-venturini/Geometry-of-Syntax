import random
import csv
import os

OUTPUT_DIR = "/Users/shamiraventurini/PycharmProjects/Geometry-of-Syntax/corpora/dative"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "jabberwocky_dative.csv")
NUM_SAMPLES = 15000
random.seed(42)

# Same Disjoint Lists
NOUNS = [
    "wug", "dax", "fep", "blicket", "glorp", "tove", "slithy", "borogove",
    "mome", "rath", "jubjub", "bandersnatch", "tumtum", "tulgey", "frabjous",
    "snark", "boojum", "vorpal", "manxome", "uffish", "kiki", "bouba",
    "zorp", "quib", "narp", "vlim", "crunk", "dril", "frob", "zib"
]

VERBS = [
    "gimbled", "gorped", "whiffled", "burbled", "snickered", "galumphed",
    "pilked", "strugged", "yomped", "fazzed", "quilliged", "chortled",
    "outgrabed", "crinked", "blemed", "porked", "glorked"
]


def generate_row():
    random.shuffle(NOUNS)
    random.shuffle(VERBS)

    p_a, p_r, p_t = NOUNS[0], NOUNS[1], NOUNS[2]
    t_a, t_r, t_t = NOUNS[3], NOUNS[4], NOUNS[5]
    p_v, t_v = VERBS[0], VERBS[1]

    p_det, t_det = ("the", "a") if random.random() > 0.5 else ("a", "the")
    p_prep, t_prep = ("to", "for") if random.random() > 0.5 else ("for", "to")

    p_do = f"{p_det} {p_a} {p_v} {p_det} {p_r} {p_det} {p_t} ."
    p_po = f"{p_det} {p_a} {p_v} {p_det} {p_t} {p_prep} {p_det} {p_r} ."
    t_do = f"{t_det} {t_a} {t_v} {t_det} {t_r} {t_det} {t_t} ."
    t_po = f"{t_det} {t_a} {t_v} {t_det} {t_t} {t_prep} {t_det} {t_r} ."

    return [p_do, p_po, t_do, t_po]


with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["p_do", "p_po", "t_do", "t_po"])
    for _ in range(NUM_SAMPLES):
        writer.writerow(generate_row())
print("Clean Dative Generated.")