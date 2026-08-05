#!/usr/bin/env python3
"""Reproduce a text BM25 baseline on the local SLUE-SQA-5 metadata export.

The script intentionally uses only the normalized question text, the corpus
document text, and document identifiers. Audio is neither loaded nor required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"[a-z0-9]+")
DEFAULT_SPLITS = ("validation", "test", "verified_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/slue_sqa5_metadata"),
        help="Directory containing slue_sqa5_corpus.csv and split CSVs.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split CSV stems to evaluate.",
    )
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path for the complete deterministic result record.",
    )
    return parser.parse_args()


def tokenize(text: object) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def require_columns(frame: pd.DataFrame, names: Iterable[str], path: Path) -> None:
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def evaluate_split(
    split_path: Path,
    bm25: BM25Okapi,
    document_ids: list[str],
) -> dict[str, object]:
    frame = pd.read_csv(split_path)
    require_columns(
        frame,
        ("question_id", "normalized_question_text", "document_id"),
        split_path,
    )
    document_index = {
        document_id: index for index, document_id in enumerate(document_ids)
    }
    gold_ids = frame["document_id"].astype(str)
    missing_gold = sorted(set(gold_ids) - set(document_index))
    if missing_gold:
        raise ValueError(
            f"{split_path} contains {len(missing_gold)} gold document IDs "
            "that are absent from the corpus"
        )

    hits_at_1 = 0
    hits_at_20 = 0
    reciprocal_rank_sum = 0.0
    zero_score_queries = 0
    gold_with_lexical_overlap = 0

    for row in frame.itertuples(index=False):
        scores = bm25.get_scores(tokenize(row.normalized_question_text))

        gold_index = document_index[str(row.document_id)]
        if float(scores.max()) <= 0.0:
            zero_score_queries += 1
            continue

        # Stable sorting makes corpus row order the deterministic tie-breaker.
        ranking = np.argsort(-scores, kind="stable")
        hits_at_1 += int(ranking[0] == gold_index)
        hits_at_20 += int(gold_index in ranking[:20])
        if scores[gold_index] > 0:
            gold_with_lexical_overlap += 1
            gold_rank = int(np.flatnonzero(ranking == gold_index)[0]) + 1
            reciprocal_rank_sum += 1.0 / gold_rank

    query_count = len(frame)
    return {
        "queries": query_count,
        "unique_gold_documents": int(gold_ids.nunique()),
        "gold_document_coverage": 1.0,
        "gold_with_lexical_overlap": gold_with_lexical_overlap,
        "zero_score_queries": zero_score_queries,
        "hit_at_1_percent": 100.0 * hits_at_1 / query_count,
        "hit_at_20_percent": 100.0 * hits_at_20 / query_count,
        "mrr": reciprocal_rank_sum / query_count,
        "input_sha256": sha256(split_path),
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    corpus_path = data_dir / "slue_sqa5_corpus.csv"
    corpus = pd.read_csv(corpus_path)
    require_columns(corpus, ("document_id", "document_text"), corpus_path)

    document_ids = corpus["document_id"].astype(str).tolist()
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Corpus document IDs must be unique")
    if corpus["document_text"].isna().any():
        raise ValueError("Corpus contains missing document text")

    tokenized_corpus = [
        tokenize(document_text) for document_text in corpus["document_text"]
    ]
    bm25 = BM25Okapi(
        tokenized_corpus,
        k1=args.k1,
        b=args.b,
        epsilon=args.epsilon,
    )
    split_results = {}
    for split in args.splits:
        split_path = data_dir / f"{split}.csv"
        split_results[split] = evaluate_split(
            split_path, bm25, document_ids
        )

    result = {
        "experiment": "SLUE-SQA-5 ground-truth-text BM25 rerun",
        "implementation": "scripts/evaluate_bm25.py",
        "corpus": {
            "path": display_path(corpus_path),
            "documents": len(corpus),
            "unique_documents": len(set(document_ids)),
            "average_token_length": bm25.avgdl,
            "vocabulary_size": len(bm25.idf),
            "nonzero_term_document_weights": sum(
                len(document_frequencies)
                for document_frequencies in bm25.doc_freqs
            ),
            "input_sha256": sha256(corpus_path),
        },
        "query_field": "normalized_question_text",
        "document_field": "document_text",
        "tokenization": "lowercase ASCII alphanumeric tokens: [a-z0-9]+",
        "implementation_detail": "rank_bm25.BM25Okapi",
        "parameters": {
            "k1": args.k1,
            "b": args.b,
            "epsilon": args.epsilon,
        },
        "tie_break": "stable corpus CSV row order",
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rank-bm25": version("rank-bm25"),
        },
        "splits": split_results,
    }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
