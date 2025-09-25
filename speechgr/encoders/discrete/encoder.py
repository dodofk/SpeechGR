"""Encoders for discrete unit representations."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

import numpy as np
from omegaconf import DictConfig
from transformers import AutoTokenizer

from speechgr.encoders.base import ModalityEncoder


class DiscreteCodeEncoder(ModalityEncoder):
    """Handles loading and mapping of discrete unit sequences."""

    def __init__(
        self,
        *,
        code_path: str,
        tokenizer_name: str,
        discrete_code_num: int,
        special_token: int,
        lookup_file_name: Optional[str],
        train_atomic: bool,
        atomic_offset: int,
        cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(name="discrete", cfg=cfg)
        self.code_path = code_path
        self.discrete_code_num = discrete_code_num
        self.special_token = special_token
        self.lookup_file_name = lookup_file_name
        self.train_atomic = train_atomic
        self.atomic_offset = atomic_offset

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.code_to_idx: dict[int, int] = {}
        self.discrete_code_lookup: List[int] = []
        self.corpus_atomic_offset: int = atomic_offset + discrete_code_num

    def build_lookup(self, pq_data) -> None:
        """Construct mapping from raw tokens to discrete indices."""

        if self.train_atomic:
            self.discrete_code_lookup = list(
                range(self.atomic_offset, self.atomic_offset + self.discrete_code_num)
            )
            self.code_to_idx = {
                idx: idx + self.atomic_offset for idx in range(self.discrete_code_num)
            }
            self.corpus_atomic_offset = self.atomic_offset + self.discrete_code_num
            return

        if self.lookup_file_name:
            lookup = np.loadtxt(self.lookup_file_name).astype(int)
            self.discrete_code_lookup = lookup.tolist()
            self.code_to_idx = {
                idx: code for idx, code in enumerate(self.discrete_code_lookup)
            }
            return

        corpus = pq_data["post_query"].tolist()
        vocab_size = self.tokenizer.vocab_size
        all_tokens = set(range(vocab_size))
        used_tokens = set()
        for text in corpus:
            used_tokens.update(self.tokenizer(text)["input_ids"])
        unused_tokens = [tok for tok in sorted(all_tokens - used_tokens) if tok >= 20]
        if len(unused_tokens) < self.discrete_code_num:
            raise ValueError(
                f"Not enough unused tokens to build lookup (need {self.discrete_code_num}, got {len(unused_tokens)})."
            )
        self.discrete_code_lookup = unused_tokens[: self.discrete_code_num]
        self.code_to_idx = {
            idx: code for idx, code in enumerate(self.discrete_code_lookup)
        }

    def _load_code_file(self, path: str) -> np.ndarray:
        code = np.loadtxt(path).astype(int)
        if code.ndim == 0:
            code = np.array([int(code)])
        return code

    def _map_sequence(self, code: np.ndarray) -> np.ndarray:
        mapped = np.array([self.code_to_idx[int(x)] for x in code], dtype=np.int64)
        return mapped

    def encode_query(
        self,
        *,
        question_id: str,
        split: str,
        max_length: int,
        special_token: Optional[int] = None,
    ) -> np.ndarray:
        code_path = os.path.join(self.code_path, f"{split}_code", f"{question_id}.code")
        code = self._load_code_file(code_path)
        return self.encode_query_sequence(
            code,
            max_length=max_length,
            special_token=special_token,
        )

    def encode_query_sequence(
        self,
        code: np.ndarray,
        *,
        max_length: int,
        special_token: Optional[int] = None,
    ) -> np.ndarray:
        mapped = self._map_sequence(code)
        prefix = special_token if special_token is not None else self.special_token
        sequence = np.concatenate([[prefix], mapped, [1]])
        if len(sequence) > max_length:
            sequence = np.concatenate([sequence[: max_length - 1], [1]])
        return sequence

    def encode_document(
        self,
        *,
        doc_id: str,
        max_length: int,
        truncate_offset: int,
    ) -> List[np.ndarray]:
        doc_path = os.path.join(self.code_path, "document_code", f"{doc_id}.code")
        code = self._load_code_file(doc_path)
        return self.encode_document_sequence(
            code,
            max_length=max_length,
            truncate_offset=truncate_offset,
        )

    def encode_document_sequence(
        self,
        code: np.ndarray,
        *,
        max_length: int,
        truncate_offset: int,
    ) -> List[np.ndarray]:
        mapped = self._map_sequence(code)
        chunks: List[np.ndarray] = []
        start = 0
        reach_end = False
        while start < len(mapped) and not reach_end:
            end = min(start + max_length, len(mapped))
            if end == len(mapped):
                reach_end = True
            chunks.append(mapped[start:end])
            start = max(0, end - truncate_offset)
            if start == end:
                start = end
        return chunks

    def label_for_document(self, doc_id: str, doc_index: int) -> int | str:
        if self.train_atomic:
            return doc_index + self.corpus_atomic_offset
        return doc_id

    def supports_precompute(self) -> bool:
        return False


__all__ = ["DiscreteCodeEncoder"]
