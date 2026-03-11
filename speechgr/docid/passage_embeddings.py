"""Deterministic passage embedding helpers for hierarchical DocID generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch


@dataclass
class TfidfPassageEmbeddingConfig:
    vocab_size: int = 2048
    use_bigrams: bool = False
    bigram_buckets: int = 0
    projection_dim: int = 256
    seed: int = 13
    l2_normalize: bool = True


@dataclass
class PassageEmbeddingBuildResult:
    config: Dict[str, object]
    doc_ids: list[str]
    embeddings: np.ndarray
    diagnostics: Dict[str, object]

    def write_artifacts(self, output_dir: str | Path) -> None:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        (target_dir / "doc_ids.json").write_text(json.dumps(self.doc_ids, indent=2))
        np.save(target_dir / "embeddings.npy", self.embeddings)
        (target_dir / "embedding_diagnostics.json").write_text(
            json.dumps(
                {
                    "config": self.config,
                    "diagnostics": self.diagnostics,
                },
                indent=2,
                sort_keys=True,
            )
        )


def _load_cache(path: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, dict):
        raise TypeError(f"Cache at '{path}' must be a mapping")
    return cache


def _compute_df(features: np.ndarray) -> np.ndarray:
    return (features > 0).sum(axis=0).astype(np.float32)


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return features / norms


def _maybe_project(features: np.ndarray, *, projection_dim: int, seed: int) -> np.ndarray:
    if projection_dim <= 0 or features.shape[1] <= projection_dim:
        return features
    rng = np.random.default_rng(seed)
    projection = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(float(projection_dim)),
        size=(features.shape[1], projection_dim),
    ).astype(np.float32)
    return features @ projection


def build_tfidf_passage_embeddings(
    cache_path: str | Path,
    *,
    config: TfidfPassageEmbeddingConfig | None = None,
) -> PassageEmbeddingBuildResult:
    cfg = config or TfidfPassageEmbeddingConfig()
    cache = _load_cache(Path(cache_path))

    doc_ids = sorted(str(doc_id) for doc_id in cache.keys())
    if not doc_ids:
        raise ValueError(f"No passages found in cache '{cache_path}'")

    feature_dim = cfg.vocab_size + (cfg.bigram_buckets if cfg.use_bigrams else 0)
    features = np.zeros((len(doc_ids), feature_dim), dtype=np.float32)

    for row_idx, doc_id in enumerate(doc_ids):
        entry = cache[doc_id]
        codes = entry.get("codes")
        if codes is None:
            raise KeyError(f"Missing 'codes' in cache entry for '{doc_id}'")

        tensor = codes if isinstance(codes, torch.Tensor) else torch.as_tensor(codes)
        sequence = tensor.long().reshape(-1)
        if sequence.numel() == 0:
            continue

        max_code = int(sequence.max().item())
        if max_code >= cfg.vocab_size:
            raise ValueError(
                f"Found token id {max_code} in '{doc_id}', which exceeds vocab_size={cfg.vocab_size}"
            )

        counts = torch.bincount(sequence, minlength=cfg.vocab_size).to(torch.float32).numpy()
        features[row_idx, : cfg.vocab_size] = counts

        if cfg.use_bigrams and cfg.bigram_buckets > 0 and sequence.numel() > 1:
            head = sequence[:-1]
            tail = sequence[1:]
            bigrams = (head * cfg.vocab_size + tail) % cfg.bigram_buckets
            bigram_counts = torch.bincount(
                bigrams,
                minlength=cfg.bigram_buckets,
            ).to(torch.float32).numpy()
            features[row_idx, cfg.vocab_size :] = bigram_counts

    df = _compute_df(features)
    num_docs = float(len(doc_ids))
    idf = np.log((num_docs + 1.0) / (df + 1.0)) + 1.0
    tfidf = features * idf
    if cfg.l2_normalize:
        tfidf = _l2_normalize(tfidf)

    embeddings = _maybe_project(
        tfidf,
        projection_dim=int(cfg.projection_dim),
        seed=int(cfg.seed),
    ).astype(np.float32)
    if cfg.l2_normalize:
        embeddings = _l2_normalize(embeddings).astype(np.float32)

    diagnostics = {
        "num_documents": len(doc_ids),
        "input_feature_dim": int(feature_dim),
        "output_embedding_dim": int(embeddings.shape[1]),
        "nonzero_df": int((df > 0).sum()),
        "mean_document_length": float(features[:, : cfg.vocab_size].sum(axis=1).mean()),
        "max_document_length": int(features[:, : cfg.vocab_size].sum(axis=1).max()),
    }

    return PassageEmbeddingBuildResult(
        config=asdict(cfg),
        doc_ids=doc_ids,
        embeddings=embeddings,
        diagnostics=diagnostics,
    )


__all__ = [
    "PassageEmbeddingBuildResult",
    "TfidfPassageEmbeddingConfig",
    "build_tfidf_passage_embeddings",
]
