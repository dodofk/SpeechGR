"""
Utilities for storing variable-length discrete unit sequences without creating
one text file per utterance.

The packed format is a single .npz archive with:
  - ids: record ids
  - codes, code_offsets, code_lengths
  - counts, count_offsets, count_lengths
  - optional text and doc_ids metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SLUE_UNIT_REPO_ID = "dodofk/slue-sqa-code-l22-c500"
DEFAULT_SLUE_UNIT_HF_PATH = f"hf://datasets/{DEFAULT_SLUE_UNIT_REPO_ID}"
HF_DATASET_PREFIX = "hf://datasets/"
PACKED_SLUE_PATTERNS = [
    "README.md",
    "documents.npz",
    "train.npz",
    "validation.npz",
    "test.npz",
    "verified_test.npz",
]


def _as_1d_int(array, dtype=np.int32) -> np.ndarray:
    values = np.asarray(array, dtype=dtype)
    if values.ndim == 0:
        return values.reshape(1)
    return values.reshape(-1)


def _pack_sequences(sequences: Iterable[np.ndarray], dtype) -> Dict[str, np.ndarray]:
    offsets = []
    lengths = []
    flat_parts = []
    cursor = 0
    for seq in sequences:
        seq = _as_1d_int(seq, dtype=dtype)
        offsets.append(cursor)
        lengths.append(len(seq))
        flat_parts.append(seq)
        cursor += len(seq)

    flat = (
        np.concatenate(flat_parts).astype(dtype, copy=False)
        if flat_parts
        else np.array([], dtype=dtype)
    )
    return {
        "flat": flat,
        "offsets": np.asarray(offsets, dtype=np.int64),
        "lengths": np.asarray(lengths, dtype=np.int32),
    }


@dataclass
class PackedUnitWriter:
    ids: list[str] = field(default_factory=list)
    codes: list[np.ndarray] = field(default_factory=list)
    counts: list[np.ndarray] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)

    def add(
        self,
        record_id,
        codes,
        counts=None,
        text: str = "",
        doc_id: str = "",
    ) -> None:
        self.ids.append(str(record_id))
        self.codes.append(_as_1d_int(codes, dtype=np.int32))
        if counts is None:
            counts = np.ones(len(self.codes[-1]), dtype=np.int32)
        self.counts.append(_as_1d_int(counts, dtype=np.int32))
        self.text.append("" if text is None else str(text))
        self.doc_ids.append("" if doc_id is None else str(doc_id))

    def save(self, path, compressed: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        code_pack = _pack_sequences(self.codes, dtype=np.int32)
        count_pack = _pack_sequences(self.counts, dtype=np.int32)
        payload = {
            "ids": np.asarray(self.ids, dtype=str),
            "codes": code_pack["flat"],
            "code_offsets": code_pack["offsets"],
            "code_lengths": code_pack["lengths"],
            "counts": count_pack["flat"],
            "count_offsets": count_pack["offsets"],
            "count_lengths": count_pack["lengths"],
            "text": np.asarray(self.text, dtype=str),
            "doc_ids": np.asarray(self.doc_ids, dtype=str),
        }
        if compressed:
            np.savez_compressed(path, **payload)
        else:
            np.savez(path, **payload)


class PackedUnitStore:
    def __init__(self, path):
        self.path = Path(path)
        self.data = np.load(self.path, allow_pickle=False)
        self.ids = [str(x) for x in self.data["ids"].tolist()]
        self.id_to_idx = {record_id: idx for idx, record_id in enumerate(self.ids)}
        self.codes = self.data["codes"]
        self.code_offsets = self.data["code_offsets"]
        self.code_lengths = self.data["code_lengths"]
        self.counts = self.data["counts"]
        self.count_offsets = self.data["count_offsets"]
        self.count_lengths = self.data["count_lengths"]
        self.text = self.data["text"] if "text" in self.data else None
        self.doc_ids = self.data["doc_ids"] if "doc_ids" in self.data else None

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, record_id) -> bool:
        return str(record_id) in self.id_to_idx

    def _slice(self, flat: np.ndarray, offsets: np.ndarray, lengths: np.ndarray, idx: int) -> np.ndarray:
        start = int(offsets[idx])
        end = start + int(lengths[idx])
        return flat[start:end]

    def get_code(self, record_id) -> np.ndarray:
        idx = self.id_to_idx[str(record_id)]
        return self._slice(self.codes, self.code_offsets, self.code_lengths, idx)

    def get_counts(self, record_id) -> np.ndarray:
        idx = self.id_to_idx[str(record_id)]
        return self._slice(self.counts, self.count_offsets, self.count_lengths, idx)

    def get_text(self, record_id) -> str:
        if self.text is None:
            return ""
        return str(self.text[self.id_to_idx[str(record_id)]])

    def get_doc_id(self, record_id) -> str:
        if self.doc_ids is None:
            return ""
        return str(self.doc_ids[self.id_to_idx[str(record_id)]])


def load_packed_store(path) -> Optional[PackedUnitStore]:
    path = Path(path)
    if path.exists():
        return PackedUnitStore(path)
    return None


def repo_id_from_hf_path(code_path: str) -> str:
    path_str = str(code_path)
    if not path_str.startswith(HF_DATASET_PREFIX):
        raise ValueError(f"Expected path starting with {HF_DATASET_PREFIX}, got {path_str}")
    repo_id = path_str[len(HF_DATASET_PREFIX) :]
    if not repo_id:
        raise ValueError(f"Missing dataset repo id in {path_str}")
    return repo_id


def download_packed_unit_dataset(
    repo_id: str = DEFAULT_SLUE_UNIT_REPO_ID,
    allow_patterns: Optional[Sequence[str]] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    force_download: bool = False,
) -> str:
    """
    Download a packed unit dataset from Hugging Face Hub and return its local path.

    By default this uses the Hugging Face cache, so repeated training runs reuse
    the same files. Pass local_dir when you want a normal materialized directory.
    """
    if repo_id.startswith(HF_DATASET_PREFIX):
        repo_id = repo_id_from_hf_path(repo_id)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Hugging Face dataset download requires huggingface_hub. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    cache_dir = cache_dir or os.environ.get("SPEECHGR_UNIT_CACHE_DIR")
    revision = revision or os.environ.get("SPEECHGR_UNIT_REVISION")
    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "allow_patterns": list(allow_patterns or PACKED_SLUE_PATTERNS),
        "force_download": force_download,
    }
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if local_dir:
        kwargs["local_dir"] = local_dir

    logger.info("Downloading packed unit dataset %s", repo_id)
    return snapshot_download(**kwargs)


def resolve_unit_code_path(
    code_path,
    allow_patterns: Optional[list[str]] = None,
) -> str:
    """
    Resolve local or Hugging Face dataset-backed unit-code paths.

    Local paths are returned unchanged. Hub paths should use:
      hf://datasets/<namespace>/<repo>
    and are downloaded through huggingface_hub.snapshot_download.
    """
    path_str = str(code_path)
    if not path_str.startswith(HF_DATASET_PREFIX):
        return path_str

    return download_packed_unit_dataset(
        repo_id=repo_id_from_hf_path(path_str),
        allow_patterns=allow_patterns,
    )
