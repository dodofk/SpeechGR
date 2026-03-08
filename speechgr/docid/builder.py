"""Minimal hierarchical DocID builder scaffold for Stage 2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


@dataclass
class HierarchicalDocIdBuilderConfig:
    num_coarse_clusters: int = 8
    leaf1_size: int = 128
    leaf2_size: int = 64
    cluster_token_prefix: str = "cl"
    leaf1_token_prefix: str = "lf1"
    leaf2_token_prefix: str = "lf2"
    token_width: int = 3
    seed: int = 13


@dataclass
class HierarchicalDocIdBuildResult:
    config: Dict[str, object]
    docid_map: Dict[str, Dict[str, object]]
    cluster_members: Dict[str, List[str]]
    valid_paths: List[List[str]]
    diagnostics: Dict[str, object]

    def write_artifacts(self, output_dir: str | Path) -> None:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        (target_dir / "docid_map.json").write_text(
            json.dumps(self.docid_map, indent=2, sort_keys=True)
        )
        (target_dir / "cluster_members.json").write_text(
            json.dumps(self.cluster_members, indent=2, sort_keys=True)
        )
        (target_dir / "valid_paths.json").write_text(
            json.dumps(self.valid_paths, indent=2)
        )
        (target_dir / "docid_diagnostics.json").write_text(
            json.dumps(
                {
                    "config": self.config,
                    "diagnostics": self.diagnostics,
                },
                indent=2,
                sort_keys=True,
            )
        )


class HierarchicalDocIdBuilder:
    """Build deterministic `cluster -> leaf1 -> leaf2` DocIDs from embeddings.

    This Stage 2 kickoff implementation intentionally favors stability and
    inspectability over sophistication: it partitions passages into balanced
    coarse buckets via a deterministic random projection, then assigns local leaf
    tokens by rank within each cluster. The API and output artifacts are the part
    we need now; higher-quality clustering can replace the internal assignment
    logic later without changing the file contract.
    """

    def __init__(self, config: HierarchicalDocIdBuilderConfig | None = None) -> None:
        self.config = config or HierarchicalDocIdBuilderConfig()

    def build(
        self,
        doc_ids: Sequence[str],
        embeddings: np.ndarray | Sequence[Sequence[float]],
    ) -> HierarchicalDocIdBuildResult:
        doc_id_list = [str(doc_id) for doc_id in doc_ids]
        if not doc_id_list:
            raise ValueError("HierarchicalDocIdBuilder requires at least one document")

        seen_doc_ids = set()
        duplicate_doc_ids = []
        for doc_id in doc_id_list:
            if doc_id in seen_doc_ids and doc_id not in duplicate_doc_ids:
                duplicate_doc_ids.append(doc_id)
            seen_doc_ids.add(doc_id)
        if duplicate_doc_ids:
            duplicates = ", ".join(sorted(duplicate_doc_ids))
            raise ValueError(
                f"HierarchicalDocIdBuilder requires unique doc_ids; duplicates: {duplicates}"
            )

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected a 2D embedding matrix, got shape {matrix.shape}")
        if matrix.shape[0] != len(doc_id_list):
            raise ValueError(
                "doc_ids and embeddings must have the same first dimension, got "
                f"{len(doc_id_list)} ids vs {matrix.shape[0]} embeddings"
            )

        cluster_assignments = self._assign_clusters(matrix)
        cluster_members: Dict[str, List[str]] = {}
        docid_map: Dict[str, Dict[str, object]] = {}
        valid_paths: List[List[str]] = []
        collisions = 0

        for cluster_idx in sorted(set(cluster_assignments.tolist())):
            member_positions = np.flatnonzero(cluster_assignments == cluster_idx)
            ordered_positions = self._order_cluster_members(matrix, member_positions)
            cluster_token = self._format_token(
                self.config.cluster_token_prefix,
                int(cluster_idx),
            )
            cluster_members[cluster_token] = []

            seen_paths = set()
            for local_rank, position in enumerate(ordered_positions.tolist()):
                leaf1_value = local_rank // self.config.leaf2_size
                leaf2_value = local_rank % self.config.leaf2_size
                if leaf1_value >= self.config.leaf1_size:
                    raise ValueError(
                        "Cluster capacity exceeded for the current scaffold. Increase "
                        "leaf1_size/leaf2_size before building DocIDs."
                    )

                leaf1_token = self._format_token(
                    self.config.leaf1_token_prefix,
                    int(leaf1_value),
                )
                leaf2_token = self._format_token(
                    self.config.leaf2_token_prefix,
                    int(leaf2_value),
                )
                path = [cluster_token, leaf1_token, leaf2_token]
                path_key = tuple(path)
                if path_key in seen_paths:
                    collisions += 1
                seen_paths.add(path_key)

                doc_id = doc_id_list[position]
                cluster_members[cluster_token].append(doc_id)
                docid_map[doc_id] = {
                    "docid": " ".join(path),
                    "tokens": path,
                    "cluster": cluster_token,
                    "leaf1": leaf1_token,
                    "leaf2": leaf2_token,
                    "cluster_index": int(cluster_idx),
                    "local_rank": int(local_rank),
                }
                valid_paths.append(path)

        diagnostics = {
            "num_documents": len(doc_id_list),
            "num_clusters": len(cluster_members),
            "max_cluster_size": max(len(members) for members in cluster_members.values()),
            "min_cluster_size": min(len(members) for members in cluster_members.values()),
            "collision_count": collisions,
            "collision_rate": collisions / max(1, len(doc_id_list)),
            "leaf_capacity_per_cluster": self.config.leaf1_size * self.config.leaf2_size,
        }

        return HierarchicalDocIdBuildResult(
            config=asdict(self.config),
            docid_map=docid_map,
            cluster_members=cluster_members,
            valid_paths=valid_paths,
            diagnostics=diagnostics,
        )

    def _assign_clusters(self, matrix: np.ndarray) -> np.ndarray:
        num_docs = matrix.shape[0]
        effective_clusters = max(1, min(self.config.num_coarse_clusters, num_docs))

        if matrix.shape[1] == 0:
            raise ValueError("Embeddings must have at least one feature dimension")

        rng = np.random.default_rng(self.config.seed)
        projection = matrix @ rng.normal(size=(matrix.shape[1],), loc=0.0, scale=1.0)
        order = np.argsort(projection, kind="stable")

        assignments = np.zeros(num_docs, dtype=np.int64)
        for cluster_idx, chunk in enumerate(np.array_split(order, effective_clusters)):
            assignments[chunk] = cluster_idx
        return assignments

    def _order_cluster_members(
        self,
        matrix: np.ndarray,
        member_positions: np.ndarray,
    ) -> np.ndarray:
        if member_positions.size <= 1:
            return member_positions

        cluster_matrix = matrix[member_positions]
        centroid = cluster_matrix.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(cluster_matrix - centroid, axis=1)
        order = np.argsort(distances, kind="stable")
        return member_positions[order]

    def _format_token(self, prefix: str, value: int) -> str:
        width = max(self.config.token_width, len(str(value)))
        return f"<{prefix}_{value:0{width}d}>"


def build_and_write_docids(
    doc_ids: Sequence[str],
    embeddings: np.ndarray | Sequence[Sequence[float]],
    output_dir: str | Path,
    config: HierarchicalDocIdBuilderConfig | None = None,
) -> HierarchicalDocIdBuildResult:
    result = HierarchicalDocIdBuilder(config=config).build(doc_ids, embeddings)
    result.write_artifacts(output_dir)
    return result


__all__ = [
    "HierarchicalDocIdBuildResult",
    "HierarchicalDocIdBuilder",
    "HierarchicalDocIdBuilderConfig",
    "build_and_write_docids",
]
