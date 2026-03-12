"""Shared helpers for hierarchical DocID text handling."""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable


DOCID_TOKEN_PATTERN = re.compile(r"<[^\s<>]+>")


def normalize_docid_text(docid: str) -> str:
    text = str(docid).strip()
    tokens = DOCID_TOKEN_PATTERN.findall(text)
    if tokens:
        return " ".join(tokens)
    return " ".join(text.split())


def extract_docid_tokens(docid: str) -> list[str]:
    text = normalize_docid_text(docid)
    tokens = DOCID_TOKEN_PATTERN.findall(text)
    if tokens:
        return tokens
    return text.split()


def extract_cluster_token(docid: str) -> str:
    tokens = extract_docid_tokens(docid)
    if not tokens:
        return ""
    return tokens[0]


def collect_docid_tokens(valid_ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    for docid in valid_ids:
        for token in DOCID_TOKEN_PATTERN.findall(str(docid)):
            seen.add(token)
    return sorted(seen)


def summarize_cluster_balance(valid_ids: Iterable[str]) -> dict[str, object]:
    cluster_counts = Counter()
    total = 0
    for docid in valid_ids:
        cluster = extract_cluster_token(docid)
        if not cluster:
            continue
        cluster_counts[cluster] += 1
        total += 1

    if total == 0:
        return {
            "num_clusters": 0,
            "num_docids": 0,
            "max_cluster_size": 0,
            "min_cluster_size": 0,
            "mean_cluster_size": 0.0,
            "top_clusters": [],
        }

    sizes = list(cluster_counts.values())
    top_clusters = [
        {"cluster": cluster, "count": count}
        for cluster, count in cluster_counts.most_common(10)
    ]
    return {
        "num_clusters": len(cluster_counts),
        "num_docids": total,
        "max_cluster_size": max(sizes),
        "min_cluster_size": min(sizes),
        "mean_cluster_size": float(sum(sizes) / len(sizes)),
        "top_clusters": top_clusters,
    }


__all__ = [
    "DOCID_TOKEN_PATTERN",
    "collect_docid_tokens",
    "extract_cluster_token",
    "extract_docid_tokens",
    "normalize_docid_text",
    "summarize_cluster_balance",
]
