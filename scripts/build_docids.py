"""Build deterministic hierarchical DocIDs from offline passage embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from speechgr.docid import HierarchicalDocIdBuilder, HierarchicalDocIdBuilderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-ids-json", required=True, help="JSON list of document ids")
    parser.add_argument(
        "--embeddings-npy",
        required=True,
        help="NumPy .npy file with shape [num_docs, dim] passage embeddings",
    )
    parser.add_argument("--output-dir", required=True, help="Artifact output directory")
    parser.add_argument("--num-coarse-clusters", type=int, default=8)
    parser.add_argument("--leaf1-size", type=int, default=128)
    parser.add_argument("--leaf2-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    doc_ids = json.loads(Path(args.doc_ids_json).read_text())
    embeddings = np.load(args.embeddings_npy)

    config = HierarchicalDocIdBuilderConfig(
        num_coarse_clusters=args.num_coarse_clusters,
        leaf1_size=args.leaf1_size,
        leaf2_size=args.leaf2_size,
        seed=args.seed,
    )
    result = HierarchicalDocIdBuilder(config).build(doc_ids, embeddings)
    result.write_artifacts(args.output_dir)

    print(
        "Wrote DocID artifacts to "
        f"{Path(args.output_dir).resolve()} (docs={result.diagnostics['num_documents']}, "
        f"clusters={result.diagnostics['num_clusters']})"
    )


if __name__ == "__main__":
    main()
