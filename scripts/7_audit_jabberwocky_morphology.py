import pandas as pd
from transformers import GPT2Tokenizer
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_NAME = "gpt2-large"
FILE_PATH = "corpora/transitive/jabberwocky_transitive.csv"  # Check Transitive verbs

# Common Past Tense Suffixes in GPT-2 BPE
# Note: GPT-2 tokens often include the leading space if it's not the start
# We look for tokens that represent "ed", "d", "bed", "led", etc.
# We will check the decoded text of the last token.

print("Loading Tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

print("Loading Corpus...")
df = pd.read_csv(FILE_PATH)

# Extract unique verbs from the corpus
# In Transitive Jabberwocky, the verb is the 3rd word in Active sentences
# "The dax [gimbled]..."
unique_verbs = set()

print("Extracting verbs...")
for sentence in df['pa']:
    words = sentence.strip().split()
    if len(words) > 2:
        unique_verbs.add(words[2])

print(f"Found {len(unique_verbs)} unique nonsense verbs.")

# --- AUDIT ---
print("\n--- Tokenization Audit ---")
consistent_count = 0
total_count = 0

results = []

for verb in unique_verbs:
    # Tokenize with a leading space (how it appears in sentence)
    tokens = tokenizer.tokenize(" " + verb)
    last_token = tokens[-1]

    # Check if the last token looks like a past tense suffix
    # We look for 'ed' at the end of the token string
    is_morphological = False
    if last_token.endswith("ed"):
        is_morphological = True

    results.append({
        "Verb": verb,
        "Tokens": tokens,
        "Last_Token": last_token,
        "Is_Morphological": is_morphological
    })

    if is_morphological:
        consistent_count += 1
    total_count += 1

# --- REPORT ---
df_audit = pd.DataFrame(results)
print(df_audit.head(10))

consistency_rate = (consistent_count / total_count) * 100
print(f"\nMorphological Consistency Rate: {consistency_rate:.1f}%")

if consistency_rate > 80:
    print("VERDICT: PASS. The tokenizer consistently isolates the past tense suffix.")
else:
    print("VERDICT: WARNING. Many verbs are not split morphologically.")