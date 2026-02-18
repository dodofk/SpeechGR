"""Base dataset and utility logic for SLUE SQA5."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_SPLITS = {"train", "validation", "test", "verified_test"}

logger = logging.getLogger(__name__)


def _load_cache(cache_path: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Precompute cache not found at '{cache_path}'")
    cache = torch.load(cache_path, map_location="cpu")
    if not isinstance(cache, dict):  # pragma: no cover - defensive
        raise TypeError(f"Cache at '{cache_path}' must be a mapping")
    return cache


def _ensure_tensor(value, dtype=torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype)
    return torch.tensor(value, dtype=dtype)


def _truncate(sequence: torch.Tensor, max_length: Optional[int]) -> torch.Tensor:
    if max_length is None or sequence.numel() <= max_length:
        return sequence
    return sequence[:max_length]


def _normalize_field_list(field: Optional[Any], default: Iterable[str]) -> List[str]:
    if field is None:
        return list(default)
    if isinstance(field, (list, tuple)):
        return [str(f) for f in field]
    return [str(field)]


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    text = str(value).strip()
    return text or None


class SLUESQA5Dataset(Dataset, ABC):
    """Base dataset that loads split manifests and corpus metadata."""

    def __init__(
        self,
        split: str,
        *,
        dataset_path: Optional[str] = None,
        csv_root: Optional[str] = None,
        include_corpus: bool = True,
        corpus_splits: Optional[Iterable[str]] = None,
        train_atomic: bool = False,
        atomic_offset: Optional[int] = None,
    ) -> None:
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got '{split}'")

        self.split = split
        root = dataset_path or csv_root
        if root is None:
            raise ValueError("'dataset_path' must be provided for SLUE SQA5 datasets")
        self.dataset_path = str(root)
        allowed_corpus_splits = (
            {"train"}
            if corpus_splits is None
            else {str(s).lower() for s in corpus_splits}
        )
        if not allowed_corpus_splits:
            allowed_corpus_splits = {"train"}
        self.allowed_corpus_splits = frozenset(allowed_corpus_splits)
        self.include_corpus = include_corpus and split in self.allowed_corpus_splits
        self.train_atomic = train_atomic
        self._atomic_offset_seed: Optional[int] = atomic_offset
        self._atomic_offset_resolved: Optional[int] = atomic_offset

        csv_dir = Path(self.dataset_path)
        self.query_frame = pd.read_csv(csv_dir / f"{split}.csv")
        corpus_path = csv_dir / "corpus.csv"
        if not corpus_path.exists():
            legacy_path = csv_dir / "slue_sqa5_corpus.csv"
            if legacy_path.exists():
                corpus_path = legacy_path
            else:
                raise FileNotFoundError(
                    f"Corpus manifest not found at '{corpus_path}' or '{legacy_path}'"
                )
        self.corpus_frame = pd.read_csv(corpus_path)

        self.query_len = len(self.query_frame)
        self.doc_ids = self.corpus_frame["document_id"].tolist()
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}

        if self.train_atomic:
            self._update_valid_ids()
        else:
            self.valid_ids = self.doc_ids

    def __len__(self) -> int:
        if self.include_corpus:
            return self.query_len + self._corpus_length()
        return self.query_len

    def __getitem__(self, index: int):
        if index < self.query_len:
            row = self.query_frame.iloc[index]
            question_id = str(row["question_id"])
            document_id = str(row["document_id"])
            doc_idx = self.doc_id_to_idx.get(document_id, -1)
            features = self._get_query_features(question_id)
            label = self._label(document_id, doc_idx)
            return features, label, doc_idx

        if not self.include_corpus:
            raise IndexError("Index out of range for query-only dataset")

        corpus_idx = index - self.query_len
        return self._get_corpus_entry(corpus_idx)

    def _label(self, document_id: str, doc_idx: int) -> str | int:
        if self.train_atomic:
            return doc_idx + self._resolve_atomic_offset()
        return document_id

    def _resolve_atomic_offset(self) -> int:
        if self._atomic_offset_resolved is None:
            self._atomic_offset_resolved = self._default_atomic_offset()
        return self._atomic_offset_resolved

    def _default_atomic_offset(self) -> int:
        return self._atomic_offset_seed or 0

    def _set_atomic_offset(self, offset: int) -> None:
        self._atomic_offset_seed = offset
        self._atomic_offset_resolved = offset
        self._update_valid_ids()

    def _update_valid_ids(self) -> None:
        if not self.train_atomic:
            self.valid_ids = self.doc_ids
            return
        offset = self._resolve_atomic_offset()
        self.valid_ids = [str(idx + offset) for idx in range(len(self.doc_ids))]

    @abstractmethod
    def _get_query_features(self, question_id: str):
        raise NotImplementedError

    @abstractmethod
    def _get_corpus_features(self, document_id: str):
        raise NotImplementedError

    def _corpus_length(self) -> int:
        return len(self.doc_ids)

    def _get_corpus_entry(self, corpus_idx: int):
        document_id = self.doc_ids[corpus_idx]
        doc_idx = self.doc_id_to_idx[document_id]
        features = self._get_corpus_features(document_id)
        label = self._label(document_id, doc_idx)
        return features, label, doc_idx
