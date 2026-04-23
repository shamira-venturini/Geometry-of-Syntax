import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOUNS = REPO_ROOT / "PrimeLM" / "vocabulary_lists" / "nounlist_usf_freq.csv"
DEFAULT_VERBS = REPO_ROOT / "PrimeLM" / "vocabulary_lists" / "verblist_T_usf_freq.csv"
DEFAULT_OUTPUT = REPO_ROOT / "corpora" / "transitive" / "usf_association_edges_core_vocab.csv"
DEFAULT_SUMMARY = REPO_ROOT / "corpora" / "transitive" / "usf_association_edges_core_vocab_summary.json"
USF_APPENDIX_A_INDEX = "http://w3.usf.edu/FreeAssociation/AppendixA/index.html"
USF_APPENDIX_A_BASE = "http://w3.usf.edu/FreeAssociation/AppendixA/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact USF association edge list for the PrimeLM core vocab. "
            "Edges are kept when FSG > 0 and both cue/target are in the selected vocab."
        )
    )
    parser.add_argument("--noun-list", type=Path, default=DEFAULT_NOUNS)
    parser.add_argument("--verb-list", type=Path, default=DEFAULT_VERBS)
    parser.add_argument("--max-rank", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        rows = []
        for row in reader:
            clean = {(k or "").strip().lstrip("\ufeff"): (v or "").strip() for k, v in row.items()}
            rows.append(clean)
    return rows


def rank_value(row: Dict[str, str], key: str = "F_rank") -> int:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return 10**9
    try:
        return int(float(raw))
    except ValueError:
        return 10**9


def load_vocab_words(noun_rows: Iterable[Dict[str, str]], verb_rows: Iterable[Dict[str, str]], max_rank: int) -> Tuple[Set[str], Dict[str, int]]:
    nouns = [row for row in noun_rows if rank_value(row) <= max_rank]
    verbs = [row for row in verb_rows if rank_value(row) <= max_rank]

    vocab: Set[str] = set()
    for row in nouns:
        token = str(row.get("nouns", "")).strip().lower()
        if token.isalpha():
            vocab.add(token)
    for row in verbs:
        lemma = str(row.get("V", "")).strip().lower()
        if lemma.isalpha():
            vocab.add(lemma)

    stats = {
        "noun_rows_kept": len(nouns),
        "verb_rows_kept": len(verbs),
        "vocab_words": len(vocab),
    }
    return vocab, stats


def usf_part_files(timeout: int) -> List[str]:
    response = requests.get(USF_APPENDIX_A_INDEX, timeout=timeout)
    response.raise_for_status()
    links = re.findall(r'HREF="([^"]+)"', response.text, flags=re.IGNORECASE)
    files = sorted({link for link in links if link.startswith("Cue_Target_Pairs")})
    if not files:
        raise RuntimeError("Could not find USF Appendix A Cue_Target_Pairs files.")
    return files


def parse_usf_edges(vocab: Set[str], timeout: int) -> Tuple[List[Tuple[str, str, float]], Dict[str, int]]:
    files = usf_part_files(timeout=timeout)
    parsed_lines = 0
    kept_directed = 0
    directed: Dict[Tuple[str, str], float] = {}

    for filename in files:
        url = USF_APPENDIX_A_BASE + filename
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        lines = response.text.splitlines()

        start_index = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("CUE, TARGET"):
                start_index = i + 1
                break

        for raw in lines[start_index:]:
            line = raw.strip()
            if not line or line.startswith("</"):
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 6:
                continue

            cue = fields[0].lower()
            target = fields[1].lower()
            fsg_raw = fields[5]
            parsed_lines += 1

            if cue not in vocab or target not in vocab:
                continue
            if not cue.isalpha() or not target.isalpha() or cue == target:
                continue

            try:
                fsg = float(fsg_raw.replace("¥", "nan"))
            except ValueError:
                continue
            if not (fsg > 0.0):
                continue

            key = (cue, target)
            if key not in directed or fsg > directed[key]:
                directed[key] = fsg

    kept_directed = len(directed)
    edges = sorted([(cue, target, float(fsg)) for (cue, target), fsg in directed.items()])

    stats = {
        "usf_files": len(files),
        "usf_rows_parsed": parsed_lines,
        "directed_edges_kept": kept_directed,
    }
    return edges, stats


def main() -> None:
    args = parse_args()

    noun_rows = load_rows(args.noun_list)
    verb_rows = load_rows(args.verb_list)
    vocab, vocab_stats = load_vocab_words(noun_rows=noun_rows, verb_rows=verb_rows, max_rank=int(args.max_rank))
    edges, edge_stats = parse_usf_edges(vocab=vocab, timeout=int(args.timeout))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cue", "target", "fsg"])
        for cue, target, fsg in edges:
            writer.writerow([cue, target, f"{fsg:.6f}"])

    covered_words = set()
    for cue, target, _ in edges:
        covered_words.add(cue)
        covered_words.add(target)

    summary = {
        "noun_source": str(args.noun_list),
        "verb_source": str(args.verb_list),
        "output_csv": str(args.output),
        "max_rank": int(args.max_rank),
        "vocab": vocab_stats,
        "usf": edge_stats,
        "covered_vocab_words": len(covered_words),
        "uncovered_vocab_words": len(vocab.difference(covered_words)),
        "notes": [
            "Edges are directed cue->target links from USF Appendix A where FSG > 0.",
            "Generation/matching code should treat these edges as undirected semantic associations by checking both directions.",
        ],
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {len(edges)} directed edges to {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
