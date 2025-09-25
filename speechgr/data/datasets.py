"""
Deprecated dataset implementations.

This module is retained for backward compatibility while the new
`speechgr.data.slue_sqa5` module takes over. Expect this file to be
refactored or removed once downstream references migrate.
"""

from dataclasses import dataclass
from tqdm import tqdm
import datasets
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import logging
import librosa
import math
from typing import Iterable

import h5py

from speechgr.encoders import DiscreteCodeEncoder, WhisperEncoder, TextEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_field_list(field: Optional[Any], default: Iterable[str]) -> List[str]:
    if field is None:
        return list(default)
    if isinstance(field, (list, tuple)):
        return [str(f) for f in field]
    return [str(field)]


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(pd, "isna") and pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _tensor_to_list(value: Any) -> List[int]:
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [int(value)] if isinstance(value, (int, np.integer)) else [value]


class BaseSpeechGRDataset(Dataset):
    """Abstract base dataset with shared SLUE-SQA bookkeeping."""

    VALID_SPLITS = {"train", "validation", "test", "verified_test"}

    def __init__(
        self,
        *,
        split: str,
        dataset_path: str,
        corpus_filename: str,
        pq_filename: str,
    ) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"split must be one of {self.VALID_SPLITS}, got '{split}'"
            )

        self.split = split
        self.dataset_path = dataset_path
        split_path = os.path.join(dataset_path, f"{split}.csv")
        self.data = pd.read_csv(split_path)
        self.query_len = len(self.data)

        corpus_path = os.path.join(dataset_path, corpus_filename)
        self.corpus_data = pd.read_csv(corpus_path)

        pq_path = os.path.join(dataset_path, pq_filename)
        pq_frame = pd.read_csv(pq_path)
        self.pq_data = pq_frame.dropna(subset=["post_query", "document_id"])

        self.doc_id_2_id: Dict[str, str] = {}
        for _, row in self.pq_data.iterrows():
            self.doc_id_2_id.setdefault(row["document_id"], str(row["idx"]))

        self.doc_id_2_idx: Dict[str, int] = {}
        for doc_id in self.corpus_data["document_id"]:
            self.doc_id_2_idx.setdefault(doc_id, len(self.doc_id_2_idx))

    def ensure_doc_index(self, doc_id: str) -> int:
        """Return (and lazily allocate) the integer index for a document id."""

        if doc_id not in self.doc_id_2_idx:
            self.doc_id_2_idx[doc_id] = len(self.doc_id_2_idx)
        return self.doc_id_2_idx[doc_id]

        return warped_spec


class SlueSQA5DatasetV2(BaseSpeechGRDataset):
    """
    This dataset class is a new version for SLUE SQA5.
    Instead of mapping a query (via question_id) to discrete tokens, this version uses the passage.
    It converts the passage to a discrete token sequence (either via a precomputed code file
    or by tokenizing the passage text and mapping tokens using a lookup), and pads/truncates the sequence
    to a fixed length. The label is the document id mapped via a discrete code mapping.

    This version use document discrete unit to directly learn the mapping for index task.
    """

    def __init__(
        self,
        split: str = "train",
        max_length: int = 512,
        dataset_path: str = "/home/ricky/dodofk/dataset/slue_sqa5/",
        code_path: str = "/home/ricky/dodofk/dataset/slue_sqa_code_l22_c500",
        pq_filename: str = "slue_sqa5_pq10_llama32_3b_clean.csv",
        corpus_filename: str = "slue_sqa5_corpus.csv",
        model_name_or_path: str = "google/flan-t5-base",
        epoch: Optional[int] = None,
        discrete_code_num: int = 128,
        truncate_offset: int = 50,
        special_token: int = 32000,
        lookup_file_name: Optional[str] = None,
        train_atomic: bool = False,
        atomic_offset: int = 50,
        question_cache_file: Optional[str] = None,
        document_cache_file: Optional[str] = None,
    ):
        super().__init__(
            split=split,
            dataset_path=dataset_path,
            corpus_filename=corpus_filename,
            pq_filename=pq_filename,
        )

        self.max_length = max_length
        self.truncate_offset = truncate_offset
        self.code_path = code_path
        self.special_token = special_token
        self.lookup_file_name = lookup_file_name
        self.train_atomic = train_atomic
        self.atomic_offset = atomic_offset
        self.epoch = epoch
        self.discrete_code_num = discrete_code_num
        self.question_cache = self._load_cache_file(question_cache_file)
        self.document_cache = self._load_cache_file(document_cache_file)

        self.encoder = DiscreteCodeEncoder(
            code_path=code_path,
            tokenizer_name=model_name_or_path,
            discrete_code_num=discrete_code_num,
            special_token=special_token,
            lookup_file_name=lookup_file_name,
            train_atomic=train_atomic,
            atomic_offset=atomic_offset,
        )
        self.encoder.build_lookup(self.pq_data)
        self.corpus_atomic_offset = self.encoder.corpus_atomic_offset
        self.idx_len = None
        if split == "train":
            self.corpus_code_data, self.corpus_code_label, self.corpus_doc_id = (
                self.build_corpus_code()
            )
        else:
            # the indexing task is only used for training
            self.corpus_code_data, self.corpus_code_label, self.corpus_doc_id = (
                self.build_corpus_code()
            )
            self.corpus_code_data = []
            self.corpus_code_label = []

        if self.train_atomic:
            self.valid_ids = list(set(self.corpus_code_label))
        else:
            self.valid_ids = list(set(self.corpus_doc_id))

    def _load_cache_file(
        self, cache_path: Optional[str]
    ) -> Dict[str, List[int]]:
        if not cache_path:
            return {}
        path = Path(cache_path).expanduser()
        if not path.exists():
            logger.warning("Cache file %s not found; ignoring", path)
            return {}
        loaded = torch.load(path, map_location="cpu")
        cache: Dict[str, List[int]] = {}
        for key, value in loaded.items():
            if isinstance(value, dict) and "codes" in value:
                cache[str(key)] = _tensor_to_list(value["codes"])
            else:
                cache[str(key)] = _tensor_to_list(value)
        return cache

    def build_corpus_code(self) -> Tuple[List[torch.LongTensor], List[str]]:
        # load train corpus data

        corpus_code_data: List[np.ndarray] = []
        corpus_code_label: List[str | int] = []
        corpus_doc_id: List[str] = []

        for doc_id in self.corpus_data["document_id"]:
            doc_idx = self.ensure_doc_index(doc_id)
            if doc_id in self.document_cache:
                raw_sequence = np.array(
                    self.document_cache[doc_id], dtype=np.int64
                )
                chunks = self.encoder.encode_document_sequence(
                    raw_sequence,
                    max_length=self.max_length,
                    truncate_offset=self.truncate_offset,
                )
            else:
                chunks = self.encoder.encode_document(
                    doc_id=doc_id,
                    max_length=self.max_length,
                    truncate_offset=self.truncate_offset,
                )
            for chunk in chunks:
                corpus_code_data.append(chunk)
                corpus_code_label.append(
                    self.encoder.label_for_document(doc_id, doc_idx)
                )
                corpus_doc_id.append(doc_id)

        return corpus_code_data, corpus_code_label, corpus_doc_id

    def __len__(self):
        # Return total length including pq data if needed
        return self.query_len + len(self.corpus_code_data)

    def __getitem__(self, idx):
        if idx < self.query_len:
            row = self.data.iloc[idx]
            question_id = row["question_id"]
            document_id = row["document_id"]
            if question_id in self.question_cache:
                raw_sequence = np.array(
                    self.question_cache[question_id], dtype=np.int64
                )
                code = self.encoder.encode_query_sequence(
                    raw_sequence,
                    max_length=self.max_length,
                    special_token=self.special_token,
                )
            else:
                code = self.encoder.encode_query(
                    question_id=question_id,
                    split=self.split,
                    max_length=self.max_length,
                    special_token=self.special_token,
                )
            doc_idx = self.ensure_doc_index(document_id)

            if self.split == "train":
                if self.train_atomic:
                    return (
                        torch.LongTensor(code),
                        doc_idx + self.corpus_atomic_offset,
                        doc_idx,
                    )
                return (
                    torch.LongTensor(code),
                    document_id,
                    doc_idx,
                )

            if self.train_atomic:
                return (
                    torch.LongTensor(code),
                    doc_idx + self.corpus_atomic_offset,
                    doc_idx,
                )
            return (
                torch.LongTensor(code),
                document_id,
                -1,
            )  # fake query id for validation set (will be removed in prediction step)
        else:
            # For index task, only used in train
            idx_adjusted = idx - self.query_len
            code = self.corpus_code_data[idx_adjusted]
            # label = self.corpus_code_label[idx_adjusted]
            label = self.corpus_doc_id[idx_adjusted]
            if self.train_atomic:
                return (
                    torch.LongTensor(code),
                    self.corpus_code_label[idx_adjusted],
                    self.doc_id_2_idx[label],
                )
            else:
                return torch.LongTensor(code), label, self.doc_id_2_idx[label]

    def calculate_stat(self):
        """
        This is a helper function to calculate the statistics of the code length.
        """
        # calculate the average length of the code
        code_lengths = []
        question_lengths = []
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        for _, row in self.data.iterrows():
            code = np.loadtxt(
                f"{self.code_path}/{self.split}_code/{row['question_id']}.code",
            ).astype(int)
            code_lengths.append(len(code))

            question_id = row["question_id"]
            question_id_tokens = tokenizer.encode(question_id)
            question_lengths.append(len(question_id_tokens))

        return {
            "code_length": {
                "mean": np.mean(code_lengths),
                "max": np.max(code_lengths),
                "min": np.min(code_lengths),
                "median": np.median(code_lengths),
                "std": np.std(code_lengths),
            },
            "question_length": {
                "mean": np.mean(question_lengths),
                "max": np.max(question_lengths),
                "min": np.min(question_lengths),
                "median": np.median(question_lengths),
                "std": np.std(question_lengths),
            },
        }


class SlueSQA5TextDataset(BaseSpeechGRDataset):
    """SLUE SQA5 dataset backed by text tokenization."""

    def __init__(
        self,
        split: str = "train",
        *,
        max_length: int = 512,
        dataset_path: str = "/home/ricky/dodofk/dataset/slue_sqa5/",
        pq_filename: str = "slue_sqa5_pq10_llama32_3b_clean.csv",
        corpus_filename: str = "slue_sqa5_corpus.csv",
        text_encoder: Optional[TextEncoder] = None,
        text_tokenizer_name: str = "google/flan-t5-base",
        query_text_field: Optional[Any] = None,
        corpus_text_field: Optional[Any] = None,
        train_atomic: bool = False,
        atomic_offset: int = 50,
        include_corpus: bool = True,
        feature_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            split=split,
            dataset_path=dataset_path,
            corpus_filename=corpus_filename,
            pq_filename=pq_filename,
        )

        self.max_length = max_length
        self.train_atomic = train_atomic
        self.atomic_offset = atomic_offset
        self.include_corpus = include_corpus and split == "train"
        self._feature_cache_dir = feature_cache_dir

        self.encoder = text_encoder or TextEncoder(
            tokenizer_name=text_tokenizer_name,
            max_length=max_length,
            padding=False,
            truncation=True,
            add_special_tokens=True,
        )

        self._cache_enabled = False
        if (
            feature_cache_dir
            and hasattr(self.encoder, "has_precomputed")
            and hasattr(self.encoder, "load_feature")
        ):
            try:
                self._cache_enabled = self.encoder.has_precomputed(
                    split, feature_cache_dir
                )
            except (FileNotFoundError, OSError):
                logger.warning(
                    "Cache directory %s missing for split %s; falling back to on-the-fly encoding",
                    feature_cache_dir,
                    split,
                )
                self._cache_enabled = False

        self.query_text_fields = _normalize_field_list(
            query_text_field, ["post_query", "question_text"]
        )
        self.corpus_text_fields = _normalize_field_list(
            corpus_text_field,
            ["normalized_document_text", "document_text", "document"],
        )

        self.query_text_map = self._build_query_text_map()
        self.doc_text_map = self._build_doc_text_map()

        self.corpus_examples: List[Tuple[Dict[str, List[int]], Any, int]] = []
        if self.include_corpus:
            self.corpus_examples = self._build_corpus_examples()

        if self.train_atomic:
            self.corpus_atomic_offset = self.atomic_offset
            self.valid_ids = sorted(
                set(label for _, label, _ in self.corpus_examples)
            )
        else:
            self.corpus_atomic_offset = 0
            self.valid_ids = list(self.doc_text_map.keys())

    def _build_query_text_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        candidates = [self.data, self.pq_data]
        for field in self.query_text_fields:
            for frame in candidates:
                if field not in frame.columns:
                    continue
                for _, row in frame.iterrows():
                    qid = str(row.get("question_id"))
                    text = _normalize_text(row.get(field))
                    if not qid or not text:
                        continue
                    mapping.setdefault(qid, text)
        missing = [
            str(row["question_id"])
            for _, row in self.data.iterrows()
            if str(row["question_id"]) not in mapping
        ]
        if missing:
            raise KeyError(
                f"Missing text for question ids: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        return mapping

    def _build_doc_text_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for field in self.corpus_text_fields:
            if field not in self.corpus_data.columns:
                continue
            for _, row in self.corpus_data.iterrows():
                doc_id = str(row.get("document_id"))
                text = _normalize_text(row.get(field))
                if not doc_id or not text:
                    continue
                mapping.setdefault(doc_id, text)
        missing = [
            str(row["document_id"])
            for _, row in self.corpus_data.iterrows()
            if str(row["document_id"]) not in mapping
        ]
        if missing:
            raise KeyError(
                f"Missing text for document ids: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        return mapping

    def _maybe_cached(self, sample_id: str) -> Optional[Dict[str, List[int]]]:
        if not self._cache_enabled or not self._feature_cache_dir:
            return None
        try:
            cached = self.encoder.load_feature(
                self.split, self._feature_cache_dir, sample_id
            )
        except (KeyError, FileNotFoundError):
            return None
        if isinstance(cached, dict):
            return {key: _tensor_to_list(val) for key, val in cached.items()}
        raise TypeError(
            "Expected cached value to be a mapping, got " f"{type(cached)!r}"
        )

    def _encode_text(self, sample_id: str, text: str) -> Dict[str, List[int]]:
        cached = self._maybe_cached(sample_id)
        if cached is not None:
            return cached

        encoded = self.encoder.encode(text, max_length=self.max_length)
        return {key: _tensor_to_list(value) for key, value in encoded.items()}

    def _build_corpus_examples(self) -> List[Tuple[Dict[str, List[int]], Any, int]]:
        examples: List[Tuple[Dict[str, List[int]], Any, int]] = []
        for doc_id, text in self.doc_text_map.items():
            doc_idx = self.ensure_doc_index(doc_id)
            encoded = self._encode_text(doc_id, text)
            label = doc_idx + self.atomic_offset if self.train_atomic else doc_id
            examples.append((encoded, label, doc_idx))
        return examples

    def __len__(self) -> int:
        return self.query_len + len(self.corpus_examples)

    def __getitem__(self, idx):
        if idx < self.query_len:
            row = self.data.iloc[idx]
            question_id = str(row["question_id"])
            document_id = str(row["document_id"])
            text = self.query_text_map[question_id]
            encoded = self._encode_text(question_id, text)
            doc_idx = self.ensure_doc_index(document_id)
            label = doc_idx + self.atomic_offset if self.train_atomic else document_id
            return encoded, label, doc_idx

        corpus_idx = idx - self.query_len
        return self.corpus_examples[corpus_idx]


class SlueSQA5DatasetContinuous(BaseSpeechGRDataset):
    """
    This dataset is used for continuous speech DSI task.
    Instead of h5py format, this version loads features from individual .npy files.
    """

    def __init__(
        self,
        split: str = "train",
        max_length: int = 512,
        dataset_path: str = "/home/ricky/dodofk/dataset/slue_sqa5/",
        feature_path: str = "/home/ricky/dodofk/dataset/slue_sqa5_wavlm_large",
        corpus_filename: str = "slue_sqa5_corpus.csv",
        pq_filename: str = "slue_sqa5_pq10_llama32_3b_clean.csv",
        truncate_offset: int = 50,
        special_token: int = 32000,
        downsample_factor: int = 2,
        offset: int = 50,
    ):
        assert split in ["train", "validation", "test", "verified_test"]

        super().__init__(
            split=split,
            dataset_path=dataset_path,
            corpus_filename=corpus_filename,
            pq_filename=pq_filename,
        )

        self.feature_path = feature_path
        self.max_length = max_length
        self.truncate_offset = truncate_offset
        self.special_token = special_token
        self.offset = offset
        self.downsample_factor = downsample_factor

        for doc_id in self.corpus_data["document_id"]:
            self.ensure_doc_index(doc_id)

        # Load data and corpus for the current split
        self.items = []
        if split == "train":
            # For training, include corpus documents
            self._load_corpus()

        # Load query data for current split
        self._load_query_data()

        # Store valid document IDs for evaluation
        self.valid_ids = list(set(self.corpus_data["document_id"]))

    def _load_query_data(self):
        """Load question features for the current split"""
        split_dir = os.path.join(self.feature_path, self.split)

        for _, row in tqdm(
            self.data.iterrows(),
            desc=f"Loading {self.split} features",
            total=len(self.data),
        ):
            question_id = row["question_id"]
            document_id = row["document_id"]

            # Load feature from NPY file
            feature_path = os.path.join(split_dir, f"{question_id}.npy")
            if not os.path.exists(feature_path):
                print(f"Warning: Feature file not found for {question_id}")
                continue

            try:
                feature = np.load(feature_path)

                # Apply any preprocessing (like downsampling if needed)
                if self.downsample_factor > 1:
                    feature = feature[:: self.downsample_factor]

                # Process query feature (add special tokens)
                processed_feature = self._preprocess_query_feature(feature)

                # Add to items list
                self.items.append(
                    {
                        "feature": processed_feature,
                        "doc_id": document_id,
                        "q_id": question_id,
                        "is_query": True,
                        "doc_idx": self.doc_id_2_idx.get(document_id, -1),
                    }
                )
            except Exception as e:
                print(f"Error loading feature for {question_id}: {str(e)}")

    def _load_corpus(self):
        """Load corpus document features for training"""
        corpus_dir = os.path.join(self.feature_path, "corpus")

        for _, row in tqdm(
            self.corpus_data.iterrows(),
            desc="Loading corpus features",
            total=len(self.corpus_data),
        ):
            doc_id = row["document_id"]

            # Load feature from NPY file
            feature_path = os.path.join(corpus_dir, f"{doc_id}.npy")
            if not os.path.exists(feature_path):
                print(f"Warning: Corpus feature file not found for {doc_id}")
                continue

            try:
                feature = np.load(feature_path)

                # Apply any preprocessing (like downsampling if needed)
                if self.downsample_factor > 1:
                    feature = feature[:: self.downsample_factor]

                # For long features, split into chunks
                if len(feature) > self.max_length:
                    chunks = self._truncate_corpus_feature(feature)
                    for i, chunk in enumerate(chunks):
                        self.items.append(
                            {
                                "feature": chunk,
                                "doc_id": doc_id,
                                "chunk_id": i,
                                "is_query": False,
                                "doc_idx": self.doc_id_2_idx.get(doc_id, -1),
                            }
                        )
                else:
                    self.items.append(
                        {
                            "feature": feature,
                            "doc_id": doc_id,
                            "chunk_id": 0,
                            "is_query": False,
                            "doc_idx": self.doc_id_2_idx.get(doc_id, -1),
                        }
                    )
            except Exception as e:
                print(f"Error loading corpus feature for {doc_id}: {str(e)}")

    def _preprocess_query_feature(self, feature: np.ndarray) -> np.ndarray:
        """
        We add special token at beginning and EOS token at end.
        If too long, truncate to max_length.
        """
        # Add special token at beginning and EOS token (1) at end
        feature = np.concatenate([[self.special_token], feature, [1]])

        # Truncate if too long
        if len(feature) > self.max_length:
            feature = np.concatenate([feature[: self.max_length - 1], [1]])

        return feature

    def _truncate_corpus_feature(self, feature: np.ndarray) -> List[np.ndarray]:
        """
        Split corpus features into overlapping chunks to handle long sequences.
        """
        chunks = []
        start_idx, reach_end = 0, False

        while start_idx < len(feature) and not reach_end:
            end_idx = min(start_idx + self.max_length, len(feature))
            if end_idx == len(feature):
                reach_end = True
            chunks.append(feature[start_idx:end_idx])
            start_idx = end_idx - self.truncate_offset

        return chunks

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return torch.tensor(item["feature"]), item["doc_id"], item["doc_idx"]


class SlueSQA5WhisperDataset(Dataset):
    """
    Dataset class for SLUE SQA5 using Whisper encoder features.
    This version loads data directly from HuggingFace dataset and extracts
    features using Whisper encoder from the audio arrays.
    """

    def __init__(
        self,
        split: str = "train",
        truncate_offset: int = 50,
        whisper_model_name: str = "openai/whisper-base",
        device: str = "cuda",
        model_name_or_path: str = "google/flan-t5-base",
        include_corpus: bool = True,
        debug_max_samples: Optional[int] = None,  # For debugging: limit dataset size
        # SpecAugment parameters - matching SpecAugmentLB defaults for LB policy
        apply_spec_augment: bool = False,
        time_warp_param: int = 80,  # Added time_warp_param to match SpecAugmentLB
        freq_mask_param: int = 27,  # Changed from 80 to 27 to match LB policy
        time_mask_param: int = 100,  # Changed from 80 to 100 to match LB policy
        num_freq_masks: int = 2,  # Keep for backward compatibility but not used in SpecAugmentLB
        num_time_masks: int = 2,  # Keep for backward compatibility but not used in SpecAugmentLB
    ):
        assert split in [
            "train",
            "validation",
            "test",
            "verified_test",
        ], "split should be in ['train', 'validation', 'test', 'verified_test']"

        self.split = split
        self.truncate_offset = truncate_offset
        self.include_corpus = include_corpus

        logging.info(
            f"Initializing SlueSQA5WhisperDataset - split: {split}, device: {device}"
        )
        if apply_spec_augment and split == "train":
            logging.info(
                f"SpecAugment enabled: freq_mask={freq_mask_param}, time_mask={time_mask_param}"
            )

        # Load dataset from HuggingFace
        self.dataset = load_dataset("asapp/slue-phase-2", "sqa5")
        self.data = self.dataset[split]

        # Apply debug limit if specified
        if debug_max_samples is not None:
            logging.info(
                f"DEBUG MODE: Limiting {split} split to {debug_max_samples} samples"
            )
            self.data = self.data.select(range(min(debug_max_samples, len(self.data))))

        self.query_len = len(self.data)

        # Initialize tokenizer for labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.encoder = WhisperEncoder(
            whisper_model_name=whisper_model_name,
            device=device,
            apply_spec_augment=apply_spec_augment,
            time_warp_param=time_warp_param,
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
        )

        # Build document id to index mapping from the dataset efficiently
        # We'll collect document IDs during corpus building to avoid multiple iterations
        self.doc_id_2_idx = {}
        self.all_doc_ids = set()

        # For training, we can include corpus documents if needed
        self.corpus_records: List[Dict[str, Any]] = []
        if split == "train" and include_corpus:
            self.corpus_records = self._build_corpus_data(debug_max_samples)
        else:
            # For non-training splits, still need all doc IDs for constrained beam search
            self._collect_all_doc_ids_efficiently()

        self.valid_ids = list(self.all_doc_ids)

    def _collect_all_doc_ids_efficiently(self):
        """Efficiently collect all document IDs from all splits for constrained beam search."""
        logging.info(
            "Collecting all document IDs from all splits for constrained beam search"
        )

        for split_name in ["train", "validation", "test", "verified_test"]:
            if split_name in self.dataset:
                logging.info(f"Collecting document IDs from {split_name} split")
                for item in self.dataset[split_name]:
                    doc_id = item["document_id"]
                    self.all_doc_ids.add(doc_id)
                    if doc_id not in self.doc_id_2_idx:
                        self.doc_id_2_idx[doc_id] = len(self.doc_id_2_idx)

        logging.info(f"Total unique document IDs collected: {len(self.all_doc_ids)}")

    def _build_corpus_data(self, debug_max_samples: Optional[int] = None):
        """Build corpus data by collecting unique documents from ALL splits and build doc_id mappings simultaneously."""
        corpus_data = []
        seen_doc_ids = set()

        # Collect unique documents from ALL splits for comprehensive corpus
        # and build document ID mappings at the same time
        for split_name in ["train", "validation", "test", "verified_test"]:
            if split_name in self.dataset:
                logging.info(f"Collecting corpus documents from {split_name} split")
                split_data = self.dataset[split_name]

                # Apply debug limit to each split if specified
                if debug_max_samples is not None:
                    max_per_split = debug_max_samples // 4  # Distribute across 4 splits
                    split_data = split_data.select(
                        range(min(max_per_split, len(split_data)))
                    )
                    logging.info(
                        f"DEBUG MODE: Limiting {split_name} to {len(split_data)} samples for corpus"
                    )

                for item in split_data:
                    doc_id = item["document_id"]

                    # Add to all_doc_ids set for valid_ids
                    self.all_doc_ids.add(doc_id)

                    # Build doc_id_2_idx mapping
                    if doc_id not in self.doc_id_2_idx:
                        self.doc_id_2_idx[doc_id] = len(self.doc_id_2_idx)

                    # Add to corpus data if not seen before
                    if doc_id not in seen_doc_ids:
                        features = self.encoder.encode_audio(
                            item["document_audio"]["array"],
                            item["document_audio"]["sampling_rate"],
                        )
                        corpus_data.append(
                            {
                                "document_id": doc_id,
                                "features": features,
                            }
                        )
                        seen_doc_ids.add(doc_id)

        logging.info(
            f"Built corpus data with {len(corpus_data)} unique documents from all splits"
        )
        logging.info(f"Total unique document IDs collected: {len(self.all_doc_ids)}")
        return corpus_data

    def _process_docid(self, doc_id: str) -> str:
        """Process document ID by replacing underscores with spaces."""
        return doc_id.replace("_", " ")
    def __len__(self):
        return self.query_len + len(self.corpus_records)

    def __getitem__(self, idx):
        if idx < self.query_len:
            # Handle query data
            item = self.data[idx]
            question_audio = item["question_audio"]["array"]
            sampling_rate = item["question_audio"]["sampling_rate"]
            document_id = item["document_id"]

            # Extract features from question audio
            features = self.encoder.encode_audio(question_audio, sampling_rate)

            # Add marker for query task (similar to special_token in discrete version)
            query_marker = torch.zeros(
                (1, features.shape[1])
            )  # Add a zero vector as marker
            features = torch.cat([query_marker, features], dim=0)

            # No truncation - let Q-Former handle the sequence compression

            # Process document ID by replacing underscores with spaces
            processed_doc_id = self._process_docid(document_id)

            if self.split == "train":
                return (
                    features,
                    processed_doc_id,
                    -1,
                )  # we don't use self-neg currently, so keep it -1
            else:
                return (
                    features,
                    processed_doc_id,
                    -1,
                )  # fake query id for validation set

        else:
            # Handle corpus data for indexing (training only)
            idx_adjusted = idx - self.query_len
            corpus_item = self.corpus_records[idx_adjusted]
            doc_id = corpus_item["document_id"]

            # Features precomputed during corpus build
            features = corpus_item["features"]

            # No truncation - let Q-Former handle the sequence compression

            # Process document ID by replacing underscores with spaces
            processed_doc_id = self._process_docid(doc_id)

            return (
                features,
                processed_doc_id,
                -1,
            )  # we don't use self-neg currently, so keep it -1
    # print("max length: ", np.max(label_length))
    # print("min length: ", np.min(label_length))

    # for i in range(len(test_dataset)):
    #     print(test_dataset.__getitem__(i))
