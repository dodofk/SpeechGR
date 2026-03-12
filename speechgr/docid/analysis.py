"""Analysis helpers for hierarchical DocID artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


@dataclass
class DocIdDistributionReport:
    num_documents: int
    num_unique_docids: int
    duplicate_docid_count: int
    num_clusters: int
    min_cluster_size: int
    max_cluster_size: int
    mean_cluster_size: float
    median_cluster_size: float
    singleton_cluster_count: int
    singleton_cluster_rate: float
    largest_cluster_share: float
    cluster_entropy: float
    top_clusters: List[Dict[str, int]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_documents": self.num_documents,
            "num_unique_docids": self.num_unique_docids,
            "duplicate_docid_count": self.duplicate_docid_count,
            "num_clusters": self.num_clusters,
            "min_cluster_size": self.min_cluster_size,
            "max_cluster_size": self.max_cluster_size,
            "mean_cluster_size": self.mean_cluster_size,
            "median_cluster_size": self.median_cluster_size,
            "singleton_cluster_count": self.singleton_cluster_count,
            "singleton_cluster_rate": self.singleton_cluster_rate,
            "largest_cluster_share": self.largest_cluster_share,
            "cluster_entropy": self.cluster_entropy,
            "top_clusters": self.top_clusters,
        }


def _load_json(path: str | Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at '{path}'")
    return payload


def _extract_cluster(entry: object) -> str:
    if isinstance(entry, dict):
        cluster = entry.get("cluster")
        if cluster is not None:
            return str(cluster)
        tokens = entry.get("tokens")
        if isinstance(tokens, list) and tokens:
            return str(tokens[0])
    if isinstance(entry, str):
        return str(entry).split()[0]
    raise TypeError("Unsupported DocID entry format")


def _extract_docid(entry: object) -> str:
    if isinstance(entry, dict):
        if "docid" in entry:
            return str(entry["docid"])
        tokens = entry.get("tokens")
        if isinstance(tokens, list):
            return " ".join(str(token) for token in tokens)
    if isinstance(entry, str):
        return str(entry)
    raise TypeError("Unsupported DocID entry format")


def analyze_docid_map(docid_map: Mapping[str, object]) -> DocIdDistributionReport:
    if not docid_map:
        raise ValueError("DocID map is empty")

    cluster_counts = Counter()
    docid_counts = Counter()

    for _, entry in docid_map.items():
        cluster_counts[_extract_cluster(entry)] += 1
        docid_counts[_extract_docid(entry)] += 1

    cluster_sizes = sorted(cluster_counts.values())
    num_documents = len(docid_map)
    num_clusters = len(cluster_counts)
    singleton_cluster_count = sum(1 for size in cluster_sizes if size == 1)
    max_cluster_size = max(cluster_sizes)
    min_cluster_size = min(cluster_sizes)
    mean_cluster_size = float(sum(cluster_sizes) / num_clusters)
    median_cluster_size = float(statistics.median(cluster_sizes))
    largest_cluster_share = max_cluster_size / num_documents

    cluster_entropy = 0.0
    for size in cluster_sizes:
        prob = size / num_documents
        cluster_entropy -= prob * math.log(prob + 1e-12)

    duplicate_docid_count = sum(count - 1 for count in docid_counts.values() if count > 1)
    top_clusters = [
        {"cluster": cluster, "count": count}
        for cluster, count in cluster_counts.most_common(20)
    ]

    return DocIdDistributionReport(
        num_documents=num_documents,
        num_unique_docids=len(docid_counts),
        duplicate_docid_count=duplicate_docid_count,
        num_clusters=num_clusters,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        mean_cluster_size=mean_cluster_size,
        median_cluster_size=median_cluster_size,
        singleton_cluster_count=singleton_cluster_count,
        singleton_cluster_rate=singleton_cluster_count / num_clusters,
        largest_cluster_share=largest_cluster_share,
        cluster_entropy=cluster_entropy,
        top_clusters=top_clusters,
    )


def analyze_docid_map_path(path: str | Path) -> DocIdDistributionReport:
    return analyze_docid_map(_load_json(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docid-map",
        required=True,
        help="Path to docid_map.json",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save the distribution report as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_docid_map_path(args.docid_map)
    payload = report.to_dict()
    print(json.dumps(payload, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved distribution report to {output_path.resolve()}")


__all__ = [
    "DocIdDistributionReport",
    "analyze_docid_map",
    "analyze_docid_map_path",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    main()
