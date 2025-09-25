"""SLUE SQA5 dataset wrappers for precomputed features."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

_SPLITS = {"train", "validation", "test", "verified_test"}


def _load_cache(cache_path: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Precompute cache not found at '{cache_path}'")
    cache = torch.load(cache_path, map_location="cpu")
    if not isinstance(cache, dict):
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


class SLUESQA5Dataset(Dataset, ABC):
    """Base dataset that loads split manifests and corpus metadata."""

    def __init__(
        self,
        split: str,
        *,
        csv_root: str,
        include_corpus: bool = True,
        train_atomic: bool = False,
        atomic_offset: int = 0,
    ) -> None:
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got '{split}'")

        self.split = split
        self.include_corpus = include_corpus and split == "train"
        self.train_atomic = train_atomic
        self.atomic_offset = atomic_offset

        csv_dir = Path(csv_root)
        self.query_frame = pd.read_csv(csv_dir / f"{split}.csv")
        self.query_len = len(self.query_frame)

        corpus_frame = pd.read_csv(csv_dir / "corpus.csv")
        self.doc_ids = corpus_frame["document_id"].tolist()
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}

        if self.train_atomic:
            self.valid_ids = [str(idx + self.atomic_offset) for idx in range(len(self.doc_ids))]
        else:
            self.valid_ids = self.doc_ids

    def __len__(self) -> int:
        if self.include_corpus:
            return self.query_len + len(self.doc_ids)
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
        document_id = self.doc_ids[corpus_idx]
        doc_idx = self.doc_id_to_idx[document_id]
        features = self._get_corpus_features(document_id)
        label = self._label(document_id, doc_idx)
        return features, label, doc_idx

    def _label(self, document_id: str, doc_idx: int) -> str | int:
        if self.train_atomic:
            return doc_idx + self.atomic_offset
        return document_id

    @abstractmethod
    def _get_query_features(self, question_id: str) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _get_corpus_features(self, document_id: str) -> torch.Tensor:
        raise NotImplementedError


class DiscreteUnitDataset(SLUESQA5Dataset):
    """Dataset that returns precomputed discrete-unit tensors."""

    def __init__(
        self,
        split: str,
        *,
        csv_root: str,
        cache_root: str,
        encoder_name: str,
        include_corpus: bool = True,
        max_length: Optional[int] = None,
        codes_key: str = "codes",
        train_atomic: bool = False,
        atomic_offset: int = 0,
    ) -> None:
        super().__init__(
            split,
            csv_root=csv_root,
            include_corpus=include_corpus,
            train_atomic=train_atomic,
            atomic_offset=atomic_offset,
        )

        cache_dir = Path(cache_root)
        self.encoder_name = encoder_name
        self.max_length = max_length
        self.codes_key = codes_key

        self.query_cache = _load_cache(
            cache_dir / split / f"{split}_{encoder_name}.pt"
        )
        self.corpus_cache = _load_cache(
            cache_dir / "corpus" / f"corpus_{encoder_name}.pt"
        )

    def _get_query_features(self, question_id: str) -> torch.Tensor:
        cache_entry = self.query_cache.get(question_id)
        if cache_entry is None:
            raise KeyError(
                f"Missing {self.encoder_name} cache for question '{question_id}'"
            )
        codes = cache_entry.get(self.codes_key)
        if codes is None:
            raise KeyError(
                f"Cache entry for '{question_id}' missing key '{self.codes_key}'"
            )
        tensor = _truncate(_ensure_tensor(codes, dtype=torch.long), self.max_length)
        return tensor

    def _get_corpus_features(self, document_id: str) -> torch.Tensor:
        cache_entry = self.corpus_cache.get(document_id)
        if cache_entry is None:
            raise KeyError(
                f"Missing {self.encoder_name} cache for document '{document_id}'"
            )
        codes = cache_entry.get(self.codes_key)
        if codes is None:
            raise KeyError(
                f"Cache entry for '{document_id}' missing key '{self.codes_key}'"
            )
        tensor = _truncate(_ensure_tensor(codes, dtype=torch.long), self.max_length)
        return tensor


class ContinuousDataset(SLUESQA5Dataset):
    """Dataset that returns precomputed continuous feature tensors."""

    def __init__(
        self,
        split: str,
        *,
        csv_root: str,
        cache_root: str,
        encoder_name: str,
        include_corpus: bool = True,
        feature_key: str = "features",
        dtype: torch.dtype = torch.float32,
        train_atomic: bool = False,
        atomic_offset: int = 0,
    ) -> None:
        super().__init__(
            split,
            csv_root=csv_root,
            include_corpus=include_corpus,
            train_atomic=train_atomic,
            atomic_offset=atomic_offset,
        )

        cache_dir = Path(cache_root)
        self.encoder_name = encoder_name
        self.feature_key = feature_key
        self.dtype = dtype

        self.query_cache = _load_cache(
            cache_dir / split / f"{split}_{encoder_name}.pt"
        )
        self.corpus_cache = _load_cache(
            cache_dir / "corpus" / f"corpus_{encoder_name}.pt"
        )

    def _get_query_features(self, question_id: str) -> torch.Tensor:
        cache_entry = self.query_cache.get(question_id)
        if cache_entry is None:
            raise KeyError(
                f"Missing {self.encoder_name} cache for question '{question_id}'"
            )
        features = cache_entry.get(self.feature_key)
        if features is None:
            raise KeyError(
                f"Cache entry for '{question_id}' missing key '{self.feature_key}'"
            )
        tensor = _ensure_tensor(features, dtype=self.dtype)
        return tensor

    def _get_corpus_features(self, document_id: str) -> torch.Tensor:
        cache_entry = self.corpus_cache.get(document_id)
        if cache_entry is None:
            raise KeyError(
                f"Missing {self.encoder_name} cache for document '{document_id}'"
            )
        features = cache_entry.get(self.feature_key)
        if features is None:
            raise KeyError(
                f"Cache entry for '{document_id}' missing key '{self.feature_key}'"
            )
        tensor = _ensure_tensor(features, dtype=self.dtype)
        return tensor


class SlueSQA5DatasetV2(DiscreteUnitDataset):
    """Backwards-compatible wrapper around :class:`DiscreteUnitDataset`."""

    def __init__(
        self,
        split: str = "train",
        *,
        max_length: int = 512,
        dataset_path: str,
        code_path: str,
        encoder_name: str = "wavtokenizer",
        include_corpus: bool = True,
        train_atomic: bool = False,
        atomic_offset: int = 0,
        **_: object,
    ) -> None:
        super().__init__(
            split=split,
            csv_root=dataset_path,
            cache_root=code_path,
            encoder_name=encoder_name,
            include_corpus=include_corpus,
            max_length=max_length,
            train_atomic=train_atomic,
            atomic_offset=atomic_offset,
        )


__all__ = [
    "SLUESQA5Dataset",
    "DiscreteUnitDataset",
    "ContinuousDataset",
    "SlueSQA5DatasetV2",
]
