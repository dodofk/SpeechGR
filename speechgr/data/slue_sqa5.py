"""SLUE SQA5 dataset wrappers for precomputed and on-the-fly encodings."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

    def _get_query_features(self, question_id: str) -> torch.Tensor:
        cache_entry = self.query_cache.get(question_id)
        if cache_entry is None:
            raise KeyError(f"Missing {self.encoder_name} cache for question '{question_id}'")
        codes = cache_entry.get(self.codes_key)
        if codes is None:
            raise KeyError(f"Cache entry for '{question_id}' missing key '{self.codes_key}'")
        tensor = _truncate(
            _ensure_tensor(codes, dtype=torch.long), self.query_max_length
        )
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
            tensor = _ensure_tensor(codes, dtype=torch.long).reshape(-1)
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


class SlueSQA5TextDataset(SLUESQA5Dataset):
    """Dataset that tokenizes SLUE text fields on the fly."""

    def __init__(
        self,
        split: str = "train",
        *,
        csv_root: Optional[str] = None,
        dataset_path: Optional[str] = None,
        text_encoder: Optional[TextEncoder] = None,
        text_tokenizer_name: str = "google/flan-t5-base",
        query_text_field: Optional[Any] = None,
        corpus_text_field: Optional[Any] = None,
        include_corpus: bool = True,
        corpus_splits: Optional[Iterable[str]] = None,
        train_atomic: bool = False,
        atomic_offset: int = 0,
    ) -> None:
        root = csv_root or dataset_path
        if root is None:
            raise ValueError("Either 'csv_root' or 'dataset_path' must be provided")

        super().__init__(
            split,
            csv_root=root,
            include_corpus=include_corpus,
            corpus_splits=corpus_splits,
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

    def _encode(self, text: str) -> Dict[str, List[int]]:
        encoded = self.encoder.encode(text)
        processed: Dict[str, torch.Tensor] = {}
        for key, value in encoded.items():
            if isinstance(value, torch.Tensor):
                tensor_value = value.clone().detach()
            else:
                tensor_value = _ensure_tensor(value, dtype=torch.long)
            processed[key] = tensor_value.tolist()
        return processed

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


def _resolve_default_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return value.replace("${hydra:runtime.cwd}", str(Path.cwd()))


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(
        description="Instantiate the default SLUE SQA5 discrete dataset for debugging",
    )
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[2]
            / "configs"
            / "data"
            / "slue_sqa5_wavtok.yaml"
        ),
        help="Path to a Hydra-style data config (default: slue_sqa5_wavtok.yaml).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to inspect (default: train).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file '{cfg_path}' not found")

    cfg = OmegaConf.load(cfg_path)
    dataset_kwargs = {
        "dataset_path": _resolve_default_path(cfg.get("dataset_path")),
        "precompute_root": _resolve_default_path(
            cfg.get("precompute_root") or cfg.get("code_path")
        ),
        "encoder_name": cfg.get("encoder_name", "wavtokenizer"),
        "include_corpus": cfg.get("include_corpus", True),
        "train_atomic": cfg.get("train_atomic", False),
        "corpus_splits": cfg.get("include_corpus_splits"),
        "corpus_chunk_size": cfg.get("corpus_chunk_size"),
        "corpus_chunk_stride": cfg.get("corpus_chunk_stride"),
        "corpus_min_tokens": cfg.get("corpus_min_tokens", 1),
        "special_token": cfg.get("special_token"),
        "query_max_length": cfg.get("query_max_length"),
        "corpus_max_length": cfg.get("corpus_max_length"),
    }

    missing_paths = [
        key
        for key in ("dataset_path", "precompute_root")
        if not dataset_kwargs.get(key)
    ]
    if missing_paths:
        raise ValueError(
            "Missing required paths in config: " + ", ".join(missing_paths)
        )

    ds = DiscreteUnitDataset(split=args.split, **dataset_kwargs)
    print(
        f"Loaded split='{args.split}' -> queries={ds.query_len}, corpus_docs={len(ds.doc_ids)}, "
        f"corpus_segments={len(ds) - ds.query_len}"
    )
    print(
        f"Raw lengths: query_max={ds._raw_query_max_tokens}, corpus_max={ds._raw_corpus_max_tokens}"
    )
    print(
        f"Effective truncation: query_max_length={ds.query_max_length}, corpus_max_length={ds.corpus_max_length}, "
        f"chunk_size={ds.corpus_chunk_size}, stride={ds.corpus_chunk_stride}"
    )


__all__ = [
    "SLUESQA5Dataset",
    "DiscreteUnitDataset",
    "ContinuousDataset",
    "SlueSQA5DatasetV2",
    "SlueSQA5TextDataset",
    "SlueSQA5WhisperDataset",
]
