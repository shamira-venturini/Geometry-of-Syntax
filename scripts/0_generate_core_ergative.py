import random
import csv
import os

# --- CONFIGURATION ---
OUTPUT_DIR = "/Users/shamiraventurini/PycharmProjects/Geometry-of-Syntax/corpora/ergative"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "CORE_ergative.csv")
NUM_SAMPLES = 3000
random.seed(42)

# --- 1. AGENTS (Stripped of "the") ---
AGENTS = [
    "woman", "man", "child", "student", "neighbor", "stranger", "friend",
    "teacher", "doctor", "nurse", "scientist", "engineer", "artist", "writer",
    "actor", "singer", "athlete", "pilot", "driver", "farmer", "lawyer",
    "judge", "manager", "chef", "waiter", "guard", "tourist", "visitor",
    "clerk", "mechanic", "government", "administration", "company",
    "committee", "council", "police", "army", "team", "school", "hospital",
    "hero", "villain", "captain", "chief", "boss", "wizard", "giant", "ghost"
]

# --- 2. VERB & PATIENT MAPPING (Tiers 1-6) ---
# Key: Base Verb, Value: List of valid Patients
VOCAB = {
    # Tier 1
    "begin": ["class", "meeting", "show", "project", "story", "game", "season", "shift", "lesson", "campaign"],
    "break": ["glass", "window", "vase", "plate", "bone", "cable", "wire", "toy", "machine", "lock"],
    "change": ["weather", "situation", "mood", "plan", "schedule", "policy", "color", "shape", "price", "style"],
    "close": ["door", "window", "shop", "store", "office", "eye", "account", "file", "case", "session"],
    "decrease": ["price", "sale", "temperature", "speed", "population", "demand", "pressure", "income", "volume",
                 "level"],  # Singularized for consistency
    "develop": ["problem", "relationship", "habit", "disease", "symptom", "pattern", "skill", "idea", "conflict",
                "plan"],
    "end": ["show", "movie", "meeting", "war", "relationship", "game", "season", "speech", "story", "contract"],
    "improve": ["health", "performance", "situation", "skill", "mood", "quality", "economy", "climate", "relation",
                "accuracy"],
    "increase": ["price", "demand", "sale", "temperature", "speed", "population", "pressure", "volume", "rate",
                 "income"],
    "move": ["car", "chair", "table", "crowd", "furniture", "train", "bus", "ship", "object", "line"],
    "open": ["door", "window", "store", "eye", "mouth", "account", "file", "curtain", "gate", "flower"],
    "start": ["engine", "car", "show", "movie", "meeting", "game", "project", "machine", "service", "campaign"],
    "stop": ["car", "music", "noise", "machine", "rain", "traffic", "service", "show", "game", "conversation"],
    "turn": ["wheel", "car", "head", "page", "knob", "key", "tide", "situation", "light", "handle"],

    # Tier 2
    "bend": ["branch", "arm", "knee", "leg", "pipe", "bar", "wire", "rod", "nail", "road"],
    "boil": ["water", "soup", "milk", "sauce", "liquid", "kettle", "mixture", "syrup", "stock", "tea"],
    "burn": ["toast", "bread", "paper", "wood", "house", "building", "forest", "candle", "skin", "fuel"],
    "cook": ["rice", "meat", "vegetable", "pasta", "sauce", "bean", "fish", "egg", "stew", "soup"],
    "crack": ["glass", "wall", "window", "plate", "bone", "mirror", "pavement", "screen", "ice", "ceiling"],
    "dry": ["cloth", "hair", "paint", "ground", "floor", "towel", "soil", "leaf", "ink", "shoe"],
    "explode": ["bomb", "mine", "device", "shell", "tank", "building", "volcano", "engine", "tire", "battery"],
    "freeze": ["water", "lake", "river", "ground", "pond", "pipe", "screen", "computer", "engine", "food"],
    "melt": ["ice", "snow", "butter", "chocolate", "plastic", "wax", "metal", "cheese", "glue", "fat"],
    "roll": ["ball", "car", "wheel", "stone", "tire", "coin", "barrel", "cart", "trolley", "bottle"],
    "split": ["wood", "rock", "party", "group", "company", "team", "community", "log", "board", "road"],
    "stretch": ["rope", "fabric", "rubber", "skin", "road", "wire", "line", "shirt", "muscle", "budget"],
    "swing": ["door", "gate", "arm", "leg", "pendulum", "sign", "branch", "lamp", "rope", "mood"],
    "tear": ["paper", "cloth", "shirt", "page", "curtain", "bag", "envelope", "jeans", "fabric", "poster"],

    # Tier 3
    "drop": ["price", "temperature", "demand", "sale", "level", "speed", "pressure", "rate", "volume", "profit"],
    "expand": ["business", "company", "market", "empire", "lung", "chest", "balloon", "network", "range", "universe"],
    "fall": ["leaf", "price", "temperature", "rain", "snow", "tree", "glass", "object", "stock", "rate"],
    "grow": ["plant", "tree", "crop", "population", "business", "demand", "hair", "beard", "city", "child"],
    "lift": ["fog", "ban", "restriction", "curtain", "mood", "cloud", "tax", "rule", "sanction", "weight"],
    "lower": ["price", "standard", "expectation", "temperature", "speed", "volume", "voice", "head", "rate", "barrier"],
    "raise": ["price", "standard", "curtain", "barrier", "flag", "hand", "rate", "speed", "level", "temperature"],
    "reduce": ["cost", "price", "speed", "risk", "noise", "waste", "weight", "pollution", "pressure", "volume"],
    "rise": ["sun", "temperature", "price", "level", "river", "smoke", "tide", "stock", "rate", "demand"],
    "shift": ["focus", "attention", "position", "gear", "opinion", "power", "trend", "schedule", "weight",
              "responsibility"],
    "sink": ["ship", "boat", "raft", "platform", "heart", "mood", "stone", "object", "chair", "island"],
    "slow": ["traffic", "growth", "economy", "process", "pace", "recovery", "progress", "train", "car", "heartbeat"],
    "speed": ["recovery", "process", "growth", "reaction", "development", "delivery", "service", "change", "production",
              "response"],
    "spread": ["fire", "disease", "virus", "rumor", "news", "smoke", "stain", "smell", "oil", "fear"],

    # Tier 4
    "accumulate": ["dust", "evidence", "debt", "wealth", "data", "knowledge", "snow", "sediment", "point",
                   "experience"],
    "adapt": ["species", "population", "system", "program", "design", "method", "process", "organism", "product",
              "behavior"],
    "adjust": ["eye", "price", "setting", "schedule", "policy", "plan", "position", "volume", "focus", "strategy"],
    "alter": ["plan", "schedule", "design", "gene", "result", "pattern", "structure", "layout", "behavior", "route"],
    "attach": ["label", "file", "document", "tag", "note", "picture", "sensor", "cable", "rope", "handle"],
    "combine": ["ingredient", "force", "factor", "color", "style", "data", "feature", "element", "function", "effort"],
    "concentrate": ["student", "troop", "capital", "effort", "power", "resource", "traffic", "population", "pollutant",
                    "light"],
    "connect": ["room", "town", "idea", "device", "wire", "pipe", "line", "network", "road", "system"],
    "contrast": ["color", "style", "idea", "image", "result", "approach", "feature", "culture", "texture", "element"],
    "convert": ["data", "energy", "file", "unit", "currency", "building", "vehicle", "space", "code", "power"],
    "emerge": ["pattern", "trend", "leader", "problem", "idea", "theme", "picture", "consensus", "winner", "species"],
    "evolve": ["language", "species", "system", "culture", "idea", "society", "technology", "pattern", "theory",
               "practice"],
    "focus": ["attention", "camera", "lens", "debate", "spotlight", "discussion", "effort", "light", "beam",
              "campaign"],
    "strengthen": ["economy", "muscle", "bone", "relationship", "system", "argument", "position", "bond", "alliance",
                   "structure"],
    "vary": ["price", "rate", "result", "color", "size", "quality", "temperature", "form", "pattern", "speed"],
    "weaken": ["signal", "structure", "system", "economy", "position", "muscle", "system", "argument", "support",
               "body"],
    "worsen": ["symptom", "condition", "crisis", "situation", "weather", "pain", "relationship", "problem", "conflict",
               "mood"],

    # Tier 5
    "bake": ["bread", "cake", "cookie", "pie", "pizza", "potato", "pastry", "fish", "chicken", "loaf"],
    "fade": ["color", "memory", "light", "sound", "image", "tattoo", "scar", "signal", "voice", "smell"],
    "fold": ["chair", "table", "paper", "map", "towel", "clothes", "wing", "arm", "tent", "stroller"],
    "fry": ["potato", "fish", "chicken", "egg", "bacon", "onion", "vegetable", "tofu", "dough", "pancake"],
    "grill": ["meat", "steak", "chicken", "fish", "vegetable", "sausage", "shrimp", "burger", "corn", "kebab"],
    "heal": ["wound", "cut", "injury", "bruise", "scar", "bone", "skin", "tissue", "relationship", "spirit"],
    "lock": ["door", "gate", "car", "phone", "computer", "window", "drawer", "safe", "bike", "account"],
    "reverse": ["car", "trend", "decision", "policy", "process", "position", "result", "ruling", "direction", "change"],
    "ring": ["bell", "phone", "alarm", "doorbell", "chime", "clock", "tone", "buzzer", "tower", "line"],
    "roast": ["chicken", "beef", "pork", "vegetable", "potato", "turkey", "nut", "bean", "pepper", "lamb"],
    "shut": ["door", "gate", "window", "shop", "factory", "eye", "mouth", "lid", "book", "laptop"],
    "slam": ["door", "gate", "window", "lid", "trunk", "drawer", "book", "door", "cupboard", "hatch"],
    "sound": ["alarm", "bell", "siren", "horn", "buzzer", "whistle", "note", "chime", "tone", "signal"],
    "unlock": ["door", "phone", "computer", "gate", "screen", "account", "safe", "car", "suitcase", "file"],
    "wake": ["baby", "child", "sleeper", "patient", "roommate", "dog", "cat", "crowd", "town", "city"],
    "wash": ["clothes", "hand", "dish", "car", "hair", "face", "floor", "window", "vegetable", "fabric"],

    # Tier 6
    "accelerate": ["car", "train", "process", "growth", "heartbeat", "reaction", "program", "decline", "development",
                   "engine"],
    "brighten": ["sky", "room", "face", "mood", "color", "screen", "display", "light", "day", "weather"],
    "compress": ["gas", "air", "file", "data", "spring", "sponge", "schedule", "space", "snow", "soil"],
    "condense": ["vapor", "steam", "mist", "cloud", "gas", "text", "report", "idea", "argument", "note"],
    "darken": ["sky", "room", "screen", "image", "mood", "color", "stain", "shadow", "glass", "water"],
    "decelerate": ["car", "train", "vehicle", "rate", "process", "growth", "rotation", "program", "bike", "plane"],
    "dissolve": ["sugar", "salt", "powder", "tablet", "crystal", "dye", "solid", "sediment", "soap", "fat"],
    "evaporate": ["water", "moisture", "liquid", "sweat", "solvent", "fuel", "spirit", "alcohol", "rain", "puddle"],
    "rotate": ["wheel", "earth", "planet", "disk", "fan", "turbine", "gear", "camera", "screen", "image"],
    "shatter": ["glass", "window", "bottle", "mirror", "screen", "plate", "windshield", "vase", "crystal", "jar"],
    "shrink": ["fabric", "shirt", "sweater", "tumor", "market", "budget", "population", "empire", "stomach",
               "waistline"],
    "solidify": ["mixture", "lava", "metal", "wax", "chocolate", "fat", "cement", "plan", "idea", "belief"],
    "spin": ["wheel", "top", "disk", "coin", "fan", "ball", "planet", "chair", "rotor", "propeller"],
    "thaw": ["ice", "ground", "food", "chicken", "beef", "fish", "relationship", "heart", "river", "frost"],
    "widen": ["road", "gap", "river", "crack", "path", "margin", "difference", "street", "corridor", "opening"]
}

VERB_LIST = list(VOCAB.keys())

# --- 3. MORPHOLOGY HELPER (Past Tense) ---
IRREGULARS = {
    "begin": "began", "break": "broke", "freeze": "froze", "grow": "grew",
    "rise": "rose", "sink": "sank", "fall": "fell", "fly": "flew",
    "ring": "rang", "sing": "sang", "shake": "shook", "spin": "spun",
    "swing": "swung", "tear": "tore", "wake": "woke", "wear": "wore",
    "write": "wrote", "shut": "shut", "spread": "spread", "split": "split",
    "cut": "cut", "hit": "hit", "hurt": "hurt", "let": "let", "put": "put",
    "quit": "quit", "read": "read", "set": "set", "shed": "shed"
}


def get_past_tense(verb):
    if verb in IRREGULARS:
        return IRREGULARS[verb]
    if verb.endswith("e"):
        return verb + "d"
    if verb.endswith("y") and verb[-2] not in "aeiou":
        return verb[:-1] + "ied"
    # Simple doubling rule for CVC (simplified)
    if verb in ["drop", "stop", "slam", "plan", "rob", "rub", "step", "slip"]:
        return verb + "ped"  # Special case for p/b
    return verb + "ed"


# --- 4. DETERMINER HELPER ---
def get_det(word):
    if word[0].lower() in ['a', 'e', 'i', 'o', 'u']:
        return "an"
    return "a"


# --- 5. GENERATOR ---
def generate_core_row():
    # Select Verbs
    v_prime, v_target = random.sample(VERB_LIST, 2)

    # Select Nouns
    p_agent = random.choice(AGENTS)
    p_patient = random.choice(VOCAB[v_prime])

    # Target Nouns (Disjoint)
    t_agent = random.choice([a for a in AGENTS if a != p_agent])
    t_patient = random.choice([p for p in VOCAB[v_target] if p != p_patient])

    # Determiners
    if random.random() > 0.5:
        p_det_type = "definite"
        t_det_type = "indefinite"
    else:
        p_det_type = "indefinite"
        t_det_type = "definite"

    def build_np(noun, det_type, capitalize=False):
        if det_type == "definite":
            det = "the"
        else:
            det = get_det(noun)
        if capitalize: det = det.capitalize()
        return f"{det} {noun}"

    # Conjugate
    vp_past = get_past_tense(v_prime)
    vt_past = get_past_tense(v_target)

    # Sentences
    # Prime Causative: "The woman broke the glass."
    p_caus = f"{build_np(p_agent, p_det_type, True)} {vp_past} {build_np(p_patient, 'definite')} ."

    # Prime Inchoative: "The glass broke."
    p_inch = f"{build_np(p_patient, p_det_type, True)} {vp_past} ."

    # Target Causative
    t_caus = f"{build_np(t_agent, t_det_type, True)} {vt_past} {build_np(t_patient, 'definite')} ."

    # Target Inchoative
    t_inch = f"{build_np(t_patient, t_det_type, True)} {vt_past} ."

    return [p_caus, p_inch, t_caus, t_inch]


# --- EXECUTION ---
print(f"Generating {NUM_SAMPLES} Core Causative pairs...")
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["p_caus", "p_inch", "t_caus", "t_inch"])
    for _ in range(NUM_SAMPLES):
        writer.writerow(generate_core_row())
print("Done.")