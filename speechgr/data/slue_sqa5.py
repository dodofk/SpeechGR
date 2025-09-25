"""SLUE SQA5 dataset wrappers for precomputed and on-the-fly encodings."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from speechgr.encoders.text.encoder import TextEncoder
from speechgr.encoders.whisper.encoder import WhisperEncoder

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
        self.corpus_frame = pd.read_csv(csv_dir / "corpus.csv")

        self.query_len = len(self.query_frame)
        self.doc_ids = self.corpus_frame["document_id"].tolist()
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
    def _get_query_features(self, question_id: str):
        raise NotImplementedError

    @abstractmethod
    def _get_corpus_features(self, document_id: str):
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

        self.query_cache = _load_cache(cache_dir / split / f"{split}_{encoder_name}.pt")
        self.corpus_cache = _load_cache(cache_dir / "corpus" / f"corpus_{encoder_name}.pt")

    def _get_query_features(self, question_id: str) -> torch.Tensor:
        cache_entry = self.query_cache.get(question_id)
        if cache_entry is None:
            raise KeyError(f"Missing {self.encoder_name} cache for question '{question_id}'")
        codes = cache_entry.get(self.codes_key)
        if codes is None:
            raise KeyError(f"Cache entry for '{question_id}' missing key '{self.codes_key}'")
        tensor = _truncate(_ensure_tensor(codes, dtype=torch.long), self.max_length)
        return tensor

    def _get_corpus_features(self, document_id: str) -> torch.Tensor:
        cache_entry = self.corpus_cache.get(document_id)
        if cache_entry is None:
            raise KeyError(f"Missing {self.encoder_name} cache for document '{document_id}'")
        codes = cache_entry.get(self.codes_key)
        if codes is None:
            raise KeyError(f"Cache entry for '{document_id}' missing key '{self.codes_key}'")
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


class SlueSQA5TextDataset(SLUESQA5Dataset):
    """Dataset that tokenizes SLUE text fields on the fly."""

    def __init__(
        self,
        split: str = "train",
        *,
        csv_root: str,
        text_encoder: Optional[TextEncoder] = None,
        text_tokenizer_name: str = "google/flan-t5-base",
        query_text_field: Optional[Any] = None,
        corpus_text_field: Optional[Any] = None,
        include_corpus: bool = True,
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

        self.encoder = text_encoder or TextEncoder(
            tokenizer_name=text_tokenizer_name,
            max_length=None,
            padding=False,
            truncation=True,
            add_special_tokens=True,
        )

        self.query_text_fields = _normalize_field_list(
            query_text_field, ["question_text", "normalized_question_text"]
        )
        self.corpus_text_fields = _normalize_field_list(
            corpus_text_field, ["document_text", "normalized_document_text"]
        )

        self.query_texts = self._extract_query_texts()
        self.corpus_texts = self._extract_corpus_texts()

    def _extract_query_texts(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for _, row in self.query_frame.iterrows():
            qid = str(row["question_id"])
            for field in self.query_text_fields:
                if field in row and _normalize_text(row[field]):
                    mapping[qid] = _normalize_text(row[field]) or ""
                    break
            else:
                raise KeyError(f"Missing query text for question_id '{qid}'")
        return mapping

    def _extract_corpus_texts(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for _, row in self.corpus_frame.iterrows():
            doc_id = str(row["document_id"])
            for field in self.corpus_text_fields:
                if field in row and _normalize_text(row[field]):
                    mapping[doc_id] = _normalize_text(row[field]) or ""
                    break
            else:
                raise KeyError(f"Missing corpus text for document_id '{doc_id}'")
        return mapping

    def _encode(self, text: str) -> Dict[str, torch.Tensor]:
        encoded = self.encoder.encode(text)
        return {key: value.clone().detach() for key, value in encoded.items()}

    def _get_query_features(self, question_id: str) -> Dict[str, torch.Tensor]:
        text = self.query_texts[question_id]
        return self._encode(text)

    def _get_corpus_features(self, document_id: str) -> Dict[str, torch.Tensor]:
        text = self.corpus_texts[document_id]
        return self._encode(text)


class SlueSQA5WhisperDataset(Dataset):
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
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got '{split}'")

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
            self._collect_document_ids()

    def _collect_document_ids(self) -> None:
        for split_name in _SPLITS:
            for item in self.dataset[split_name]:
                doc_id = item["document_id"]
                if doc_id not in self.doc_id_to_idx:
                    self.doc_id_to_idx[doc_id] = len(self.doc_id_to_idx)
        self.valid_ids = list(self.doc_id_to_idx.keys())

    def _build_corpus(self, debug_max_samples: Optional[int]) -> None:
        seen = set()
        for split_name in _SPLITS:
            records = self.dataset[split_name]
            if debug_max_samples is not None:
                portion = max(1, debug_max_samples // len(_SPLITS))
                records = records.select(range(min(portion, len(records))))
            for item in records:
                doc_id = item["document_id"]
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                if doc_id not in self.doc_id_to_idx:
                    self.doc_id_to_idx[doc_id] = len(self.doc_id_to_idx)
                features = self.encoder.encode_audio(
                    item["document_audio"]["array"],
                    item["document_audio"]["sampling_rate"],
                )
                self.corpus_records.append({"document_id": doc_id, "features": features})
        self.valid_ids = list(self.doc_id_to_idx.keys())

    def __len__(self) -> int:
        return self.query_len + (len(self.corpus_records) if self.include_corpus and self.split == "train" else 0)

    def __getitem__(self, index: int):
        if index < self.query_len:
            sample = self.data[index]
            features = self.encoder.encode_audio(
                sample["question_audio"]["array"],
                sample["question_audio"]["sampling_rate"],
            )
            document_id = sample["document_id"].replace("_", " ")
            return features, document_id, -1

        if not (self.include_corpus and self.split == "train"):
            raise IndexError("Index out of range for query-only dataset")

        corpus_idx = index - self.query_len
        record = self.corpus_records[corpus_idx]
        document_id = record["document_id"].replace("_", " ")
        return record["features"], document_id, -1


__all__ = [
    "SLUESQA5Dataset",
    "DiscreteUnitDataset",
    "ContinuousDataset",
    "SlueSQA5DatasetV2",
    "SlueSQA5TextDataset",
    "SlueSQA5WhisperDataset",
]
