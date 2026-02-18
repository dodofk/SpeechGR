"""Text-based datasets for SLUE SQA5."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from speechgr.encoders.text.encoder import TextEncoder

from .base import (
    SLUESQA5Dataset,
    _normalize_field_list,
    _normalize_text,
)

logger = logging.getLogger(__name__)


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

    def _encode(self, text: str) -> Dict[str, Any]:
        from .base import _ensure_tensor
        import torch
        
        encoded = self.encoder.encode(text)
        processed: Dict[str, Any] = {}
        for key, value in encoded.items():
            if isinstance(value, torch.Tensor):
                tensor_value = value.clone().detach()
            else:
                tensor_value = _ensure_tensor(value, dtype=torch.long)
            processed[key] = tensor_value.tolist()
        return processed

    def _get_query_features(self, question_id: str) -> Dict[str, Any]:
        text = self.query_texts[question_id]
        return self._encode(text)

    def _get_corpus_features(self, document_id: str) -> Dict[str, Any]:
        text = self.corpus_texts[document_id]
        return self._encode(text)
