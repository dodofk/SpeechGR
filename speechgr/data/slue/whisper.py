"""Continuous and Whisper-based datasets for SLUE SQA5."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from speechgr.encoders.whisper.encoder import WhisperEncoder

from .base import (
    SLUESQA5Dataset,
    _ensure_tensor,
    _load_cache,
)

logger = logging.getLogger(__name__)


class ContinuousDataset(SLUESQA5Dataset):
    """Dataset that returns precomputed continuous feature tensors."""

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
        feature_key: str = "features",
        dtype: torch.dtype = torch.float32,
        train_atomic: bool = False,
        atomic_offset: Optional[int] = None,
        corpus_splits: Optional[Iterable[str]] = None,
    ) -> None:
        root = dataset_path or csv_root
        if root is None:
            raise ValueError("'dataset_path' must be provided for continuous datasets")
        storage_root = precompute_root or cache_root
        if storage_root is None:
            raise ValueError("'precompute_root' must be provided for continuous datasets")

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
        self.feature_key = feature_key
        self.dtype = dtype

        self.query_cache = _load_cache(cache_dir / split / f"{split}_{encoder_name}.pt")
        self.corpus_cache = _load_cache(cache_dir / "corpus" / f"corpus_{encoder_name}.pt")

    def _get_query_features(self, question_id: str) -> torch.Tensor:
        cache_entry = self.query_cache.get(question_id)
        if cache_entry is None:
            raise KeyError(f"Missing {self.encoder_name} cache for question '{question_id}'")
        features = cache_entry.get(self.feature_key)
        if features is None:
            raise KeyError(f"Cache entry for '{question_id}' missing key '{self.feature_key}'")
        tensor = _ensure_tensor(features, dtype=self.dtype)
        return tensor

    def _get_corpus_features(self, document_id: str) -> torch.Tensor:
        cache_entry = self.corpus_cache.get(document_id)
        if cache_entry is None:
            raise KeyError(f"Missing {self.encoder_name} cache for document '{document_id}'")
        features = cache_entry.get(self.feature_key)
        if features is None:
            raise KeyError(f"Cache entry for '{document_id}' missing key '{self.feature_key}'")
        tensor = _ensure_tensor(features, dtype=self.dtype)
        return tensor


class SlueSQA5WhisperCachedDataset(ContinuousDataset):
    """Cache-backed dataset that serves precomputed Whisper feature tensors."""

    def __init__(
        self,
        split: str,
        *,
        dataset_path: Optional[str] = None,
        csv_root: Optional[str] = None,
        precompute_root: Optional[str] = None,
        cache_root: Optional[str] = None,
        encoder_name: str = "whisper",
        include_corpus: bool = True,
        feature_key: str = "features",
        length_key: str = "length",
        dtype: torch.dtype = torch.float32,
        train_atomic: bool = False,
        atomic_offset: Optional[int] = None,
        corpus_splits: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(
            split,
            dataset_path=dataset_path,
            csv_root=csv_root,
            precompute_root=precompute_root,
            cache_root=cache_root,
            encoder_name=encoder_name,
            include_corpus=include_corpus,
            feature_key=feature_key,
            dtype=dtype,
            train_atomic=train_atomic,
            atomic_offset=atomic_offset,
            corpus_splits=corpus_splits,
        )
        self.length_key = length_key

    def _extract_tensor_with_length(
        self,
        cache: Dict[str, Any],
        key: str,
        *,
        context: str,
    ) -> tuple[torch.Tensor, int]:
        entry = cache.get(key)
        if entry is None:
            raise KeyError(f"Missing {self.encoder_name} cache for {context}")
        if not isinstance(entry, dict):
            raise TypeError(
                f"Expected mapping for cache entry '{context}', got {type(entry)!r}"
            )

        features = entry.get(self.feature_key)
        if features is None:
            raise KeyError(
                f"Cache entry for {context} missing key '{self.feature_key}'"
            )
        tensor = _ensure_tensor(features, dtype=self.dtype)
        if tensor.ndim != 2:
            raise ValueError(
                f"Expected 2D Whisper features for {context}; got shape {tuple(tensor.shape)}"
            )

        length_val = entry.get(self.length_key)
        if length_val is None:
            seq_len = int(tensor.shape[0])
        else:
            seq_len = int(length_val)
        return tensor, seq_len

    def _get_query_tensor_with_length(
        self, question_id: str
    ) -> tuple[torch.Tensor, int]:
        return self._extract_tensor_with_length(
            self.query_cache, question_id, context=f"query '{question_id}'"
        )

    def _get_corpus_tensor_with_length(
        self, document_id: str
    ) -> tuple[torch.Tensor, int]:
        return self._extract_tensor_with_length(
            self.corpus_cache, document_id, context=f"document '{document_id}'"
        )

    def __getitem__(self, index: int):
        if index < self.query_len:
            row = self.query_frame.iloc[index]
            question_id = str(row["question_id"])
            document_id = str(row["document_id"])
            doc_idx = self.doc_id_to_idx.get(document_id, -1)
            features, length = self._get_query_tensor_with_length(question_id)
            label = self._label(document_id, doc_idx)
            return features, label, doc_idx, length

        if not self.include_corpus:
            raise IndexError("Index out of range for query-only dataset")

        corpus_idx = index - self.query_len
        document_id = self.doc_ids[corpus_idx]
        doc_idx = self.doc_id_to_idx[document_id]
        features, length = self._get_corpus_tensor_with_length(document_id)
        label = self._label(document_id, doc_idx)
        return features, label, doc_idx, length


class SlueSQA5WhisperDataset(torch.utils.data.Dataset):
    """Dataset that extracts Whisper features on the fly from HF SLUE data."""

    def __init__(
        self,
        split: str = "train",
        *,
        whisper_model_name: str = "openai/whisper-base",
        device: str = "cuda",
        model_name_or_path: str = "google/flan-t5-base",
        include_corpus: bool = True,
        debug_max_samples: Optional[int] = None,
        apply_spec_augment: bool = False,
        time_warp_param: int = 80,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
    ) -> None:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.split = split
        self.include_corpus = include_corpus

        self.dataset = load_dataset("asapp/slue-phase-2", "sqa5")
        self.data = self.dataset[split]
        if debug_max_samples is not None:
            self.data = self.data.select(range(min(debug_max_samples, len(self.data))))

        self.query_len = len(self.data)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.encoder = WhisperEncoder(
            whisper_model_name=whisper_model_name,
            device=device,
            apply_spec_augment=apply_spec_augment,
            time_warp_param=time_warp_param,
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
        )

        self.doc_id_to_idx: Dict[str, int] = {}
        self.valid_ids: List[str] = []
        self.corpus_records: List[Dict[str, torch.Tensor]] = []

        if include_corpus and split == "train":
            self._build_corpus(debug_max_samples)
        else:
            # For val/test, we usually only need queries OR the corpus is handled elsewhere.
            # Minimal metadata setup:
            full_corpus = self.dataset["train"]  # assuming train has all docs
            self.doc_ids = list(set(full_corpus["document_id"]))
            self.doc_id_to_idx = {did: i for i, did in enumerate(self.doc_ids)}
            self.valid_ids = self.doc_ids

    def _build_corpus(self, debug_max_samples: Optional[int] = None) -> None:
        """Collect unique documents from the training split."""
        seen = set()
        for row in self.data:
            doc_id = row["document_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                self.corpus_records.append(row)
        
        self.doc_ids = [str(r["document_id"]) for r in self.corpus_records]
        self.doc_id_to_idx = {did: i for i, did in enumerate(self.doc_ids)}
        self.valid_ids = self.doc_ids

    def __len__(self) -> int:
        if self.include_corpus:
            return self.query_len + len(self.corpus_records)
        return self.query_len

    def __getitem__(self, index: int):
        if index < self.query_len:
            row = self.data[index]
            audio = row["audio"]["array"]
            sr = row["audio"]["sampling_rate"]
            document_id = str(row["document_id"])
            doc_idx = self.doc_id_to_idx.get(document_id, -1)
            
            features = self.encoder.encode_audio(audio, sr)
            length = features.shape[0]
            
            label_ids = self.tokenizer(
                document_id, add_special_tokens=True, return_tensors="pt"
            ).input_ids.squeeze(0)
            
            return features, document_id, doc_idx, length

        corpus_idx = index - self.query_len
        row = self.corpus_records[corpus_idx]
        audio = row["audio"]["array"]
        sr = row["audio"]["sampling_rate"]
        document_id = str(row["document_id"])
        doc_idx = self.doc_id_to_idx[document_id]

        features = self.encoder.encode_audio(audio, sr)
        length = features.shape[0]
        
        return features, document_id, doc_idx, length
