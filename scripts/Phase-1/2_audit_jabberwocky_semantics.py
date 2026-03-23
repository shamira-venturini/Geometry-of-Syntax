import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
VOCAB_PATH = REPO_ROOT / "corpora" / "jabberwocky_transitive" / "jabberwocky_transitive_strict_vocabulary.json"
NOUN_REF_PATH = REPO_ROOT / "PrimeLM" / "vocabulary_lists" / "nounlist_usf_freq.csv"
VERB_REF_PATH = REPO_ROOT / "PrimeLM" / "vocabulary_lists" / "verblist_T_usf_freq.csv"
OUTPUT_DIR = REPO_ROOT / "behavioral_results" / "jabberwocky_semantic_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit semantic leakage of strict Jabberwocky vocabulary with model representations."
    )
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vocab-path", type=Path, default=VOCAB_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def get_device(user_device: Optional[str]) -> str:
    if user_device:
        return user_device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_nonce_vocabulary(path: Path) -> Dict[str, List[str]]:
    payload = json.loads(path.read_text())
    nouns = payload["nouns"]

    if "verb_stems" in payload:
        verb_present = [stem + "s" for stem in payload["verb_stems"]]
        verb_past = [stem + "ed" for stem in payload["verb_stems"]]
    elif "verb_present" in payload and "verb_past" in payload:
        verb_present = payload["verb_present"]
        verb_past = payload["verb_past"]
    else:
        raise ValueError("Vocabulary JSON must contain either verb_stems or both verb_present and verb_past.")

    return {
        "nouns": nouns,
        "verb_present": verb_present,
        "verb_past": verb_past,
    }


def load_reference_nouns() -> List[str]:
    frame = pd.read_csv(NOUN_REF_PATH, sep=";")
    return sorted(frame["nouns"].str.strip().str.lower().unique().tolist())


def load_reference_verbs() -> Dict[str, List[str]]:
    frame = pd.read_csv(VERB_REF_PATH, sep=";")
    return {
        "present": sorted(frame["pres_3s"].str.strip().str.lower().unique().tolist()),
        "past": sorted(frame["past_A"].str.strip().str.lower().unique().tolist()),
    }


def mean_pool_rows(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.mean(dim=0)


def token_span(tokenizer, prefix: str, word_surface: str) -> Tuple[List[int], List[int], int, int]:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    word_ids = tokenizer.encode(" " + word_surface, add_special_tokens=False)
    start = len(prefix_ids)
    end = start + len(word_ids)
    return prefix_ids, word_ids, start, end


def lexical_representation(tokenizer, embedding_weight: torch.Tensor, word_surface: str) -> torch.Tensor:
    word_ids = tokenizer.encode(" " + word_surface, add_special_tokens=False)
    if not word_ids:
        raise ValueError(f"Tokenizer produced no ids for {word_surface}")
    rows = embedding_weight[word_ids]
    return mean_pool_rows(rows)


def contextual_representation(tokenizer, model, device: str, prefix: str, word_surface: str, suffix: str) -> torch.Tensor:
    text = prefix + " " + word_surface + suffix
    _, word_ids, start, end = token_span(tokenizer, prefix, word_surface)
    encoded = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0]

    observed = encoded.input_ids[0, start:end].tolist()
    if observed != word_ids:
        raise ValueError(f"Token span mismatch for {word_surface}: {observed} != {word_ids}")

    return mean_pool_rows(hidden[start:end].cpu())


def batch_contextual_representations(
    tokenizer,
    model,
    device: str,
    words: Sequence[str],
    prefix: str,
    suffix: str,
) -> Dict[str, torch.Tensor]:
    representations: Dict[str, torch.Tensor] = {}
    for word in words:
        representations[word] = contextual_representation(
            tokenizer=tokenizer,
            model=model,
            device=device,
            prefix=prefix,
            word_surface=word,
            suffix=suffix,
        )
    return representations


def lexical_representations(tokenizer, embedding_weight: torch.Tensor, words: Sequence[str]) -> Dict[str, torch.Tensor]:
    return {word: lexical_representation(tokenizer, embedding_weight, word) for word in words}


def similarity_table(
    nonce_reps: Dict[str, torch.Tensor],
    ref_reps: Dict[str, torch.Tensor],
    top_k: int,
    category: str,
    representation_type: str,
) -> pd.DataFrame:
    ref_words = list(ref_reps.keys())
    ref_matrix = torch.stack([ref_reps[word] for word in ref_words], dim=0)
    ref_matrix = F.normalize(ref_matrix, dim=1)

    rows = []
    for nonce_word, nonce_rep in nonce_reps.items():
        query = F.normalize(nonce_rep.unsqueeze(0), dim=1)
        sims = torch.matmul(query, ref_matrix.T).squeeze(0)
        values, indices = torch.topk(sims, k=min(top_k, sims.shape[0]))
        neighbors = [
            f"{ref_words[index]}:{float(value):.4f}"
            for value, index in zip(values.tolist(), indices.tolist())
        ]
        rows.append(
            {
                "category": category,
                "representation_type": representation_type,
                "nonce_word": nonce_word,
                "max_cosine": float(values[0]),
                "mean_topk_cosine": float(values.mean()),
                "nearest_neighbors": ";".join(neighbors),
            }
        )
    return pd.DataFrame(rows)


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["category", "representation_type"], as_index=False)
        .agg(
            n_words=("nonce_word", "count"),
            mean_max_cosine=("max_cosine", "mean"),
            max_max_cosine=("max_cosine", "max"),
            mean_topk_cosine=("mean_topk_cosine", "mean"),
        )
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    embedding_weight = model.get_input_embeddings().weight.detach().cpu()

    nonce = load_nonce_vocabulary(args.vocab_path)
    ref_nouns = load_reference_nouns()
    ref_verbs = load_reference_verbs()

    noun_lex_nonce = lexical_representations(tokenizer, embedding_weight, nonce["nouns"])
    noun_lex_ref = lexical_representations(tokenizer, embedding_weight, ref_nouns)
    noun_ctx_nonce = batch_contextual_representations(
        tokenizer, model, device, nonce["nouns"], prefix="the", suffix=" is here ."
    )
    noun_ctx_ref = batch_contextual_representations(
        tokenizer, model, device, ref_nouns, prefix="the", suffix=" is here ."
    )

    nonce_verb_present = nonce["verb_present"]
    nonce_verb_past = nonce["verb_past"]

    verb_pres_lex_nonce = lexical_representations(tokenizer, embedding_weight, nonce_verb_present)
    verb_pres_lex_ref = lexical_representations(tokenizer, embedding_weight, ref_verbs["present"])
    verb_pres_ctx_nonce = batch_contextual_representations(
        tokenizer, model, device, nonce_verb_present, prefix="they", suffix=" it ."
    )
    verb_pres_ctx_ref = batch_contextual_representations(
        tokenizer, model, device, ref_verbs["present"], prefix="they", suffix=" it ."
    )

    verb_past_lex_nonce = lexical_representations(tokenizer, embedding_weight, nonce_verb_past)
    verb_past_lex_ref = lexical_representations(tokenizer, embedding_weight, ref_verbs["past"])
    verb_past_ctx_nonce = batch_contextual_representations(
        tokenizer, model, device, nonce_verb_past, prefix="they", suffix=" it ."
    )
    verb_past_ctx_ref = batch_contextual_representations(
        tokenizer, model, device, ref_verbs["past"], prefix="they", suffix=" it ."
    )

    tables = [
        similarity_table(noun_lex_nonce, noun_lex_ref, args.top_k, "noun", "lexical"),
        similarity_table(noun_ctx_nonce, noun_ctx_ref, args.top_k, "noun", "contextual"),
        similarity_table(verb_pres_lex_nonce, verb_pres_lex_ref, args.top_k, "verb_present", "lexical"),
        similarity_table(verb_pres_ctx_nonce, verb_pres_ctx_ref, args.top_k, "verb_present", "contextual"),
        similarity_table(verb_past_lex_nonce, verb_past_lex_ref, args.top_k, "verb_past", "lexical"),
        similarity_table(verb_past_ctx_nonce, verb_past_ctx_ref, args.top_k, "verb_past", "contextual"),
    ]

    detail = pd.concat(tables, ignore_index=True)
    summary = summary_table(detail)

    detail.to_csv(args.output_dir / "semantic_leakage_detail_old_jabber.csv", index=False)
    summary.to_csv(args.output_dir / "semantic_leakage_summary_old_jabber.csv", index=False)


if __name__ == "__main__":
    main()
