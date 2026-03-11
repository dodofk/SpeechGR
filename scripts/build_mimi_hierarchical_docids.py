"""Build hierarchical DocIDs directly from a Mimi corpus cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from speechgr.docid import (
    HierarchicalDocIdBuilder,
    HierarchicalDocIdBuilderConfig,
    TfidfPassageEmbeddingConfig,
    build_tfidf_passage_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-cache",
        required=True,
        help="Path to corpus Mimi cache (e.g. outputs/.../corpus/corpus_mimi.pt)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where embedding and DocID artifacts will be written",
    )
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--use-bigrams", action="store_true")
    parser.add_argument("--bigram-buckets", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--target-cluster-size", type=int, default=64)
    parser.add_argument("--num-coarse-clusters", type=int, default=0)
    parser.add_argument("--leaf1-size", type=int, default=128)
    parser.add_argument("--leaf2-size", type=int, default=64)
    return parser.parse_args()


def _resolve_cluster_count(num_docs: int, requested: int, target_cluster_size: int) -> int:
    if requested > 0:
        return requested
    target = max(1, int(target_cluster_size))
    return max(1, (int(num_docs) + target - 1) // target)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_cfg = TfidfPassageEmbeddingConfig(
        vocab_size=args.vocab_size,
        use_bigrams=bool(args.use_bigrams),
        bigram_buckets=int(args.bigram_buckets),
        projection_dim=int(args.projection_dim),
        seed=int(args.seed),
    )
    embedding_result = build_tfidf_passage_embeddings(
        args.corpus_cache,
        config=embedding_cfg,
    )
    embedding_result.write_artifacts(output_dir)

    num_clusters = _resolve_cluster_count(
        num_docs=len(embedding_result.doc_ids),
        requested=int(args.num_coarse_clusters),
        target_cluster_size=int(args.target_cluster_size),
    )
    builder_cfg = HierarchicalDocIdBuilderConfig(
        num_coarse_clusters=num_clusters,
        leaf1_size=int(args.leaf1_size),
        leaf2_size=int(args.leaf2_size),
        seed=int(args.seed),
    )
    docid_result = HierarchicalDocIdBuilder(builder_cfg).build(
        embedding_result.doc_ids,
        embedding_result.embeddings,
    )
    docid_result.write_artifacts(output_dir)

    print(
        "Built hierarchical DocIDs at {} (docs={}, clusters={}, collision_rate={:.6f})".format(
            output_dir,
            docid_result.diagnostics["num_documents"],
            docid_result.diagnostics["num_clusters"],
            docid_result.diagnostics["collision_rate"],
        )
    )


if __name__ == "__main__":
    main()
