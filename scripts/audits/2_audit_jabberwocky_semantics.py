import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
VOCAB_PATH = (
    REPO_ROOT
    / "corpora/transitive/vocabulary_lists/jabberwocky_lexicon_monosyllabic.json"
)
REFERENCE_VOCAB_DIR = REPO_ROOT / "corpora" / "transitive" / "vocabulary_lists"
NOUN_REF_PATH = REFERENCE_VOCAB_DIR / "nounlist_usf_freq.csv"
OUTPUT_DIR = (
    REPO_ROOT / "corpora" / "transitive" / "validation" / "jabberwocky_semantics"
)


def portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit semantic leakage of canonical Jabberwocky nouns with model representations."
    )
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--reference-mode",
        choices=("bundled", "wordfreq", "wordlist"),
        default="bundled",
        help="Reference vocabulary to compare against.",
    )
    parser.add_argument(
        "--wordfreq-top",
        type=int,
        default=50000,
        help="Top-N English words to use when --reference-mode wordfreq.",
    )
    parser.add_argument(
        "--wordlist-path",
        type=Path,
        default=None,
        help="Path to newline-delimited word list when --reference-mode wordlist.",
    )
    parser.add_argument(
        "--contextual",
        choices=("auto", "on", "off"),
        default="auto",
        help="Whether to compute contextual embeddings (auto=on for the bundled noun list).",
    )
    parser.add_argument(
        "--contextual-ref-limit",
        type=int,
        default=10000,
        help="Skip contextual embeddings when reference list exceeds this size.",
    )
    parser.add_argument("--vocab-path", type=Path, default=VOCAB_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the tokenizer/model only from the local Hugging Face cache.",
    )
    return parser.parse_args()


def get_device(user_device: Optional[str]) -> str:
    if user_device:
        return user_device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_nonce_nouns(path: Path) -> List[str]:
    payload = json.loads(path.read_text())
    nouns = [str(noun).strip().lower() for noun in payload.get("nouns", [])]
    if not nouns or any(not noun for noun in nouns) or len(nouns) != len(set(nouns)):
        raise ValueError("Vocabulary JSON must contain unique, non-empty nouns.")
    return nouns


def load_reference_nouns() -> List[str]:
    frame = pd.read_csv(NOUN_REF_PATH, sep=";")
    return sorted(frame["nouns"].str.strip().str.lower().unique().tolist())


def load_wordfreq_list(top_n: int) -> List[str]:
    try:
        import wordfreq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wordfreq is required for --reference-mode wordfreq. Install with `pip install wordfreq`."
        ) from exc
    words = wordfreq.top_n_list("en", top_n, wordlist="best", ascii_only=True)
    return sorted({word.lower() for word in words if word.isalpha()})


def load_wordlist(path: Path) -> List[str]:
    if path is None:
        raise ValueError("--wordlist-path is required when --reference-mode wordlist.")
    if not path.exists():
        raise FileNotFoundError(f"Wordlist not found: {path}")
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if word and word.isalpha():
            words.append(word)
    return sorted(set(words))


def load_reference_set(args: argparse.Namespace) -> Tuple[List[str], str]:
    if args.reference_mode == "bundled":
        return load_reference_nouns(), "bundled"
    if args.reference_mode == "wordfreq":
        words = load_wordfreq_list(args.wordfreq_top)
    else:
        words = load_wordlist(args.wordlist_path)
    return words, args.reference_mode


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
    query_type: str = "nonce",
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
                "query_type": query_type,
                "query_word": nonce_word,
                "max_cosine": float(values[0]),
                "mean_topk_cosine": float(values.mean()),
                "nearest_neighbors": ";".join(neighbors),
            }
        )
    return pd.DataFrame(rows)


def reference_baseline_table(
    ref_reps: Dict[str, torch.Tensor],
    top_k: int,
    category: str,
    representation_type: str,
) -> pd.DataFrame:
    words = list(ref_reps)
    matrix = F.normalize(torch.stack([ref_reps[word] for word in words]), dim=1)
    similarities = torch.matmul(matrix, matrix.T)
    similarities.fill_diagonal_(-torch.inf)
    rows = []
    for row_index, word in enumerate(words):
        values, indices = torch.topk(
            similarities[row_index],
            k=min(top_k, len(words) - 1),
        )
        neighbors = [
            f"{words[index]}:{float(value):.4f}"
            for value, index in zip(values.tolist(), indices.tolist())
        ]
        rows.append(
            {
                "category": category,
                "representation_type": representation_type,
                "query_type": "real_baseline",
                "query_word": word,
                "max_cosine": float(values[0]),
                "mean_topk_cosine": float(values.mean()),
                "nearest_neighbors": ";".join(neighbors),
            }
        )
    return pd.DataFrame(rows)


def centered_representations(
    nonce_reps: Dict[str, torch.Tensor],
    ref_reps: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    center = torch.stack([*nonce_reps.values(), *ref_reps.values()]).mean(dim=0)
    return (
        {word: vector - center for word, vector in nonce_reps.items()},
        {word: vector - center for word, vector in ref_reps.items()},
    )


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["category", "representation_type", "query_type"], as_index=False)
        .agg(
            n_words=("query_word", "count"),
            mean_max_cosine=("max_cosine", "mean"),
            max_max_cosine=("max_cosine", "max"),
            mean_topk_cosine=("mean_topk_cosine", "mean"),
        )
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.eval()
    embedding_weight = model.get_input_embeddings().weight.detach().cpu()

    nonce_nouns = load_nonce_nouns(args.vocab_path)
    ref_nouns, reference_label = load_reference_set(args)

    contextual_setting = args.contextual
    include_contextual = (
        contextual_setting == "on"
        or (contextual_setting == "auto" and args.reference_mode == "bundled")
    )
    ref_size = len(ref_nouns)
    if include_contextual and ref_size > args.contextual_ref_limit:
        print(
            f"Reference list size {ref_size} exceeds contextual limit "
            f"{args.contextual_ref_limit}; skipping contextual embeddings."
        )
        include_contextual = False

    noun_lex_nonce = lexical_representations(tokenizer, embedding_weight, nonce_nouns)
    noun_lex_ref = lexical_representations(tokenizer, embedding_weight, ref_nouns)
    if include_contextual:
        noun_ctx_nonce = batch_contextual_representations(
            tokenizer, model, device, nonce_nouns, prefix="the", suffix=" is here ."
        )
        noun_ctx_ref = batch_contextual_representations(
            tokenizer, model, device, ref_nouns, prefix="the", suffix=" is here ."
        )

    tables = [
        similarity_table(noun_lex_nonce, noun_lex_ref, args.top_k, "noun", "lexical"),
        reference_baseline_table(noun_lex_ref, args.top_k, "noun", "lexical"),
    ]
    if include_contextual:
        noun_ctx_nonce, noun_ctx_ref = centered_representations(noun_ctx_nonce, noun_ctx_ref)
        tables.extend(
            [
                similarity_table(
                    noun_ctx_nonce,
                    noun_ctx_ref,
                    args.top_k,
                    "noun",
                    "contextual_centered",
                ),
                reference_baseline_table(
                    noun_ctx_ref,
                    args.top_k,
                    "noun",
                    "contextual_centered",
                ),
            ]
        )

    detail = pd.concat(tables, ignore_index=True)
    summary = summary_table(detail)

    suffix = reference_label
    detail.to_csv(args.output_dir / f"semantic_leakage_detail_{suffix}.csv", index=False)
    summary.to_csv(args.output_dir / f"semantic_leakage_summary_{suffix}.csv", index=False)
    metadata = {
        "model_name": args.model_name,
        "device": device,
        "reference_mode": args.reference_mode,
        "reference_noun_count": len(ref_nouns),
        "nonce_noun_count": len(nonce_nouns),
        "contextual_embeddings": include_contextual,
        "contextual_centering": "joint mean of nonce and reference noun vectors"
        if include_contextual
        else None,
        "reference_baseline": "leave-one-out nearest neighbors among real nouns",
        "top_k": args.top_k,
        "vocab_path": portable_path(args.vocab_path),
    }
    (args.output_dir / f"semantic_leakage_metadata_{suffix}.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
