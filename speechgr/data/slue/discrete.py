"""Discrete unit datasets for SLUE SQA5."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from .base import (
    SLUESQA5Dataset,
    _ensure_tensor,
    _load_cache,
    _truncate,
)

logger = logging.getLogger(__name__)


class DiscreteUnitDataset(SLUESQA5Dataset):
    """Dataset that returns precomputed discrete-unit tensors."""

    def __init__(
        self,
        split: str,
        *,
        dataset_path: Optional[str] = None,
        csv_root: Optional[str] = None,
        precompute_root: Optional[str] = None,
        cache_root: Optional[str] = None,
        encoder_name: str,
        include_corpus: bool = True,
        max_length: Optional[int] = None,
        query_max_length: Optional[int] = None,
        corpus_max_length: Optional[int] = None,
        codes_key: str = "codes",
        train_atomic: bool = False,
        atomic_offset: Optional[int] = None,
        corpus_splits: Optional[Iterable[str]] = None,
        corpus_chunk_size: Optional[int] = None,
        corpus_chunk_stride: Optional[int] = None,
        corpus_min_tokens: int = 1,
        special_token: Optional[int] = None,
    ) -> None:
        root = dataset_path or csv_root
        if root is None:
            raise ValueError("'dataset_path' must be provided for discrete datasets")
        storage_root = precompute_root or cache_root
        if storage_root is None:
            raise ValueError("'precompute_root' must be provided for discrete datasets")

        super().__init__(
            split,
            dataset_path=root,
            include_corpus=include_corpus,
            corpus_splits=corpus_splits,
            train_atomic=train_atomic,
            atomic_offset=atomic_offset,
        )

        cache_dir = Path(storage_root)
        self.encoder_name = encoder_name
        default_limit = max_length if max_length is not None else 512
        if query_max_length is None and max_length is None:
            logger.warning(
                "DiscreteUnitDataset[%s]: no query_max_length provided; defaulting to %d",
                self.split,
                default_limit,
            )
        if corpus_max_length is None and max_length is None:
            logger.warning(
                "DiscreteUnitDataset[%s]: no corpus_max_length provided; defaulting to %d",
                self.split,
                default_limit,
            )
        self.max_length = default_limit
        self.query_max_length = query_max_length or default_limit
        self.corpus_max_length = corpus_max_length or default_limit
        self.codes_key = codes_key
        self.corpus_chunk_size = corpus_chunk_size
        self.corpus_chunk_stride = corpus_chunk_stride
        self.corpus_min_tokens = max(1, int(corpus_min_tokens))
        self.special_token = special_token

        if self.corpus_chunk_size is None:
            self.corpus_chunk_size = self.corpus_max_length
        if self.corpus_chunk_stride is None and self.corpus_chunk_size is not None:
            self.corpus_chunk_stride = self.corpus_chunk_size

        self.query_cache = _load_cache(cache_dir / split / f"{split}_{encoder_name}.pt")
        self.corpus_cache = _load_cache(cache_dir / "corpus" / f"corpus_{encoder_name}.pt")
        self._corpus_tensor_cache: Dict[str, torch.Tensor] = {}
        self.corpus_segments_per_doc: Dict[str, int] = {
            doc_id: 0 for doc_id in self.doc_ids
        }
        if self.include_corpus:
            self._corpus_segments = self._build_corpus_segments()
        else:
            self._corpus_segments = []

        self._raw_query_max_tokens = self._max_cache_length(self.query_cache)
        self._raw_corpus_max_tokens = self._max_cache_length(self.corpus_cache)

        dataset_total = len(self)
        logger.info(
            "DiscreteUnitDataset[%s]: queries=%d corpus_docs=%d corpus_segments=%d "
            "(raw_query_max=%d, raw_corpus_max=%d, query_max_length=%d, corpus_max_length=%d, chunk_size=%s, stride=%s)",
            self.split,
            self.query_len,
            len(self.doc_ids),
            len(self._corpus_segments),
            self._raw_query_max_tokens,
            self._raw_corpus_max_tokens,
            self.query_max_length,
            self.corpus_max_length,
            self.corpus_chunk_size if self.corpus_chunk_size is not None else "None",
            self.corpus_chunk_stride if self.corpus_chunk_stride is not None else "None",
        )
        logger.info(
            "DiscreteUnitDataset[%s]: effective dataset rows=%d (queries + corpus segments)",
            self.split,
            dataset_total,
        )

        if self.train_atomic:
            if self._atomic_offset_seed is None:
                offset = self._auto_atomic_offset()
            else:
                offset = self._atomic_offset_seed
            self._set_atomic_offset(offset)

    def _max_cache_length(self, cache: Dict[str, Dict[str, torch.Tensor]]) -> int:
        max_len = 0
        for entry in cache.values():
            codes = entry.get(self.codes_key)
            if codes is None:
                continue
            tensor = _ensure_tensor(codes, dtype=torch.long)
            length = tensor.numel()
            if length > max_len:
                max_len = length
        return max_len

    @staticmethod
    def _normalize_feature_tensor(tensor: torch.Tensor, *, context: str) -> torch.Tensor:
        squeezed = tensor.squeeze()
        if squeezed.ndim != 1:
            raise ValueError(
                f"Expected 1D codes for {context}; got shape {tuple(tensor.shape)} "
                f"which becomes {tuple(squeezed.shape)} after squeeze."
            )
        return squeezed

    def _get_query_features(self, question_id: str) -> torch.Tensor:
        cache_entry = self.query_cache.get(question_id)
        if cache_entry is None:
            raise KeyError(f"Missing {self.encoder_name} cache for question '{question_id}'")
        codes = cache_entry.get(self.codes_key)
        if codes is None:
            raise KeyError(f"Cache entry for '{question_id}' missing key '{self.codes_key}'")
        tensor = _ensure_tensor(codes, dtype=torch.long)
        tensor = self._normalize_feature_tensor(
            tensor, context=f"query '{question_id}'"
        )
        tensor = _truncate(tensor, self.query_max_length)
        return tensor.reshape(-1)

    def _get_corpus_tensor(self, document_id: str) -> torch.Tensor:
        tensor = self._corpus_tensor_cache.get(document_id)
        if tensor is None:
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
            tensor = _ensure_tensor(codes, dtype=torch.long)
            tensor = self._normalize_feature_tensor(
                tensor, context=f"document '{document_id}'"
            )
            self._corpus_tensor_cache[document_id] = tensor
        return tensor

    def _compute_segment_offsets(self, length: int) -> List[Tuple[int, int]]:
        if self.corpus_chunk_size is None or self.corpus_chunk_size <= 0:
            return [(0, length)]

        stride = self.corpus_chunk_stride or self.corpus_chunk_size
        if stride <= 0:
            raise ValueError("corpus_chunk_stride must be positive when chunking is enabled")

        offsets: List[Tuple[int, int]] = []
        start = 0
        chunk = self.corpus_chunk_size
        while start < length:
            end = min(start + chunk, length)
            seg_len = end - start
            if seg_len >= self.corpus_min_tokens:
                offsets.append((start, end))
            if end == length:
                break
            start += stride

        if not offsets:
            end = length if length > 0 else chunk
            offsets.append((max(0, end - chunk), end))
        else:
            last_start, last_end = offsets[-1]
            if last_end < length:
                offsets[-1] = (last_start, length)

        return offsets

    def _build_corpus_segments(self) -> List[Tuple[str, int, int, int]]:
        segments: List[Tuple[str, int, int, int]] = []
        for doc_idx, document_id in enumerate(self.doc_ids):
            tensor = self._get_corpus_tensor(document_id)
            offsets = self._compute_segment_offsets(tensor.numel())
            self.corpus_segments_per_doc[document_id] = len(offsets)
            for start, end in offsets:
                segments.append((document_id, doc_idx, start, end))
        return segments

    def _corpus_length(self) -> int:
        return len(self._corpus_segments)

    def _get_corpus_entry(self, corpus_idx: int):
        document_id, doc_idx, start, end = self._corpus_segments[corpus_idx]
        features = self._get_corpus_features(
            document_id, start=start, end=end
        )
        label = self._label(document_id, doc_idx)
        return features, label, doc_idx

    def _get_corpus_features(
        self,
        document_id: str,
        *,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> torch.Tensor:
        tensor = self._get_corpus_tensor(document_id)
        if start is not None or end is not None:
            tensor = tensor[(start or 0) : end]
        corpus_limit = self.corpus_chunk_size or self.corpus_max_length
        tensor = _truncate(tensor.clone(), corpus_limit)
        return tensor.reshape(-1)

    def _auto_atomic_offset(self) -> int:
        max_token = -1

        def _scan_cache(cache: Dict[str, Dict[str, torch.Tensor]]) -> None:
            nonlocal max_token
            for entry in cache.values():
                codes = entry.get(self.codes_key)
                if codes is None:
                    continue
                tensor = _ensure_tensor(codes, dtype=torch.long)
                if tensor.numel() == 0:
                    continue
                value = int(tensor.max().item())
                if value > max_token:
                    max_token = value

        _scan_cache(self.query_cache)
        _scan_cache(self.corpus_cache)

        reserved = {0, 1}
        if self.special_token is not None:
            reserved.add(int(self.special_token))
        baseline = max(reserved)

        candidate = max(max_token + 1, baseline + 1)
        return candidate


class SlueSQA5DatasetV2(DiscreteUnitDataset):
    """Backwards-compatible wrapper around :class:`DiscreteUnitDataset`."""

    def __init__(
        self,
        split: str = "train",
        *,
        max_length: int = 512,
        dataset_path: str,
        precompute_root: Optional[str] = None,
        code_path: Optional[str] = None,
        encoder_name: str = "wavtokenizer",
        include_corpus: bool = True,
        train_atomic: bool = False,
        atomic_offset: Optional[int] = None,
        corpus_splits: Optional[Iterable[str]] = None,
        corpus_chunk_size: Optional[int] = None,
        corpus_chunk_stride: Optional[int] = None,
        corpus_min_tokens: int = 1,
        query_max_length: Optional[int] = None,
        corpus_max_length: Optional[int] = None,
        **_: object,
    ) -> None:
        cache_root = precompute_root or code_path
        if cache_root is None:
            raise ValueError("'precompute_root' must be provided for discrete datasets")

        super().__init__(
            split=split,
            dataset_path=dataset_path,
            precompute_root=cache_root,
            encoder_name=encoder_name,
            include_corpus=include_corpus,
            max_length=max_length,
            query_max_length=query_max_length,
            corpus_max_length=corpus_max_length,
            train_atomic=train_atomic,
            atomic_offset=atomic_offset,
            corpus_splits=corpus_splits,
            corpus_chunk_size=corpus_chunk_size,
            corpus_chunk_stride=corpus_chunk_stride,
            corpus_min_tokens=corpus_min_tokens,
        )
