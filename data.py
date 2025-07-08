"""
This code is modified from github `ArvinZhuang/DSI-QG` repo.

The file contains dataset class for our speech DSI task.
"""

from dataclasses import dataclass
from tqdm import tqdm
import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding, AutoTokenizer
from transformers import WhisperProcessor, WhisperModel
from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
import pandas as pd
import os
from typing import Optional, Tuple, List
import h5py
import logging
import librosa
import soundfile as sf
from transformers.models.whisper.feature_extraction_whisper import (
    WhisperFeatureExtractor,
)
import torch.nn.functional as F
from torchaudio.transforms import FrequencyMasking, TimeMasking

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpecAugmentLB(torch.nn.Module):
    """
    This class is used to apply SpecAugment to the log mel spectrogram input features. Using 'LB' policy.
    """

    def __init__(self, time_warp_param=80, freq_mask_param=27, time_mask_param=100):
        super().__init__()
        self.time_warp_param = time_warp_param
        self.freq_mask = FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = TimeMasking(time_mask_param=time_mask_param)

    def forward(self, spec):
        # spec: (batch, n_mels, time)
        # 1) time‐warp (simple center‐based)
        batch, n_mels, T = spec.shape
        original_length = T

        # Skip time warp if sequence is too short
        if T <= 2 * self.time_warp_param:
            warped_spec = spec
        else:
            # choose a warp center away from edges
            center = torch.randint(
                self.time_warp_param, T - self.time_warp_param, (batch,)
            )
            warped = []
            for i, c in enumerate(center):
                left = spec[i, :, :c].unsqueeze(0)
                right = spec[i, :, c:].unsqueeze(0)
                # Fix padding logic to match notebook implementation
                left = F.pad(left, (self.time_warp_param, 0))[
                    :, :, : c + self.time_warp_param
                ]
                right = F.pad(right, (0, self.time_warp_param))[
                    :, :, : T - c + self.time_warp_param
                ]
                warped.append(
                    torch.cat(
                        [left[:, :, :c], right[:, :, self.time_warp_param :]], dim=2
                    )
                )
            warped_spec = torch.cat(warped, dim=0)

        # debug warped_sepc shape
        print("Debug; warped_spec shape: ", warped_spec.shape)

        # 2) freq & time masks
        warped_spec = self.freq_mask(warped_spec)
        warped_spec = self.time_mask(warped_spec)

        # # Ensure output length matches input length (critical for Whisper)
        # current_length = warped_spec.shape[2]
        # if current_length != original_length:
        #     if current_length > original_length:
        #         # Truncate if too long
        #         warped_spec = warped_spec[:, :, :original_length]
        #     else:
        #         # Pad if too short
        #         pad_length = original_length - current_length
        #         warped_spec = F.pad(warped_spec, (0, pad_length))

        return warped_spec


class SlueSQA5DatasetV2(Dataset):
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
        special_token: int = 32000,  # specific the special token to use for query task
        lookup_file_name: Optional[str] = None,
        train_atomic: bool = False,
        atomic_offset: int = 50,
    ):
        assert split in [
            "train",
            "validation",
            "test",
            "verified_test",
        ], "split should be in ['train', 'validation', 'test', 'verified_test']"

        # Load main data CSV (assumed to contain at least "document_id" and "passage" columns)
        path_to_data = os.path.join(dataset_path, f"{split}.csv")
        self.data = pd.read_csv(path_to_data)
        self.query_len = len(self.data)
        self.max_length = max_length
        self.truncate_offset = truncate_offset
        self.split = split
        self.code_path = code_path
        self.special_token = special_token
        self.lookup_file_name = lookup_file_name
        self.train_atomic = train_atomic
        self.atomic_offset = atomic_offset
        # Load pq data used to build a mapping from document_id to a unique identifier
        self.corpus_data = pd.read_csv(os.path.join(dataset_path, corpus_filename))
        # pq_data only used for build up document id to index mapping
        pq_data = pd.read_csv(os.path.join(dataset_path, pq_filename))

        self.pq_data = pq_data.dropna(
            subset=["post_query", "document_id"],
        )
        # Build mapping from document_id to a string identifier
        self.doc_id_2_id = {}
        for _, row in self.pq_data.iterrows():
            if row["document_id"] not in self.doc_id_2_id:
                self.doc_id_2_id[row["document_id"]] = str(row["idx"])

        # self.valid_ids = set(self.doc_id_2_id.values())

        # Initialize tokenizer and build a discrete code lookup (from unused tokens in pq data)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.epoch = epoch
        self.discrete_code_num = discrete_code_num
        self.build_code_lookup()
        self.doc_id_2_idx = (
            {}
        )  # build a str to int mapping to enable transformers library

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
            self.valid_ids = list(
                set(self.corpus_code_label)
            )  # try to change numberic-id to docid
        else:
            self.valid_ids = list(
                set(self.corpus_doc_id)
            )  # try to change numberic-id to docid

    def build_code_lookup(self):
        """
        Build a discrete code lookup based on the post-query text from pq data.
        This reuses logic from the original SlueSQA5Dataset.
        """
        if self.train_atomic:
            self.discrete_code_lookup = list(
                range(self.atomic_offset, self.atomic_offset + self.discrete_code_num)
            )
            self.code_to_idx = {
                idx: idx + self.atomic_offset for idx in range(self.discrete_code_num)
            }

            self.corpus_atomic_offset = (
                self.atomic_offset + self.discrete_code_num
            )  # update for passage use
        elif self.lookup_file_name:
            lookup = np.loadtxt(self.lookup_file_name).astype(int)
            self.discrete_code_lookup = lookup
            self.code_to_idx = {
                idx: code for idx, code in enumerate(self.discrete_code_lookup)
            }
        else:
            corpus = self.pq_data["post_query"].tolist()
            all_set = set(list(range(self.tokenizer.vocab_size)))
            used_set = set()
            for c in corpus:
                for t in self.tokenizer(c)["input_ids"]:
                    used_set.add(t)
            unused_tokens = sorted(list(all_set - used_set))
            # Remove tokens < 20 to avoid special tokens
            unused_tokens = [t for t in unused_tokens if t >= 20]

            if len(unused_tokens) < self.discrete_code_num:
                raise ValueError(
                    f"Not enough unused tokens to build a discrete code lookup. Got {len(unused_tokens)} tokens, but {self.discrete_code_num} are required."
                )

            # Use the first N tokens as discrete code lookup
            self.discrete_code_lookup = unused_tokens[: self.discrete_code_num]
            self.code_to_idx = {
                idx: code for idx, code in enumerate(self.discrete_code_lookup)
            }

    def build_corpus_code(self) -> Tuple[List[torch.LongTensor], List[str]]:
        # load train corpus data

        corpus_code_data = []
        corpus_code_label = []
        corpus_doc_id = []

        if self.train_atomic:
            current_offset = self.corpus_atomic_offset

        for doc_id in self.corpus_data["document_id"]:
            if doc_id not in self.doc_id_2_idx:
                self.doc_id_2_idx[doc_id] = len(self.doc_id_2_idx)

            code_file_path = os.path.join(
                self.code_path,
                "document_code",
                f"{doc_id}.code",
            )
            code = np.loadtxt(code_file_path).astype(int)
            code = np.vectorize(self.code_to_idx.get)(code)

            start_idx = 0
            reach_end = False

            while start_idx < len(code) and not reach_end:
                end_idx = min(start_idx + self.max_length, len(code))

                if end_idx == len(code):
                    reach_end = True

                if self.train_atomic:
                    corpus_code_data.append(code[start_idx:end_idx])
                    corpus_code_label.append(self.doc_id_2_idx[doc_id] + current_offset)
                    if (
                        self.doc_id_2_idx[doc_id] + current_offset
                        > self.tokenizer.vocab_size
                    ):
                        raise Exception(
                            f"Document id {doc_id} + current_offset {current_offset} is greater than the tokenizer vocab size {self.tokenizer.vocab_size}"
                        )
                    corpus_doc_id.append(doc_id)
                else:
                    corpus_code_data.append(code[start_idx:end_idx])
                    corpus_code_label.append(self.doc_id_2_id[doc_id])
                    corpus_doc_id.append(doc_id)
                start_idx = end_idx - self.truncate_offset

        return corpus_code_data, corpus_code_label, corpus_doc_id

    def __len__(self):
        # Return total length including pq data if needed
        return self.query_len + len(self.corpus_code_data)

    def __getitem__(self, idx):
        if idx < self.query_len:
            row = self.data.iloc[idx]
            question_id = row["question_id"]
            document_id = row["document_id"]
            # Attempt to load a precomputed passage discrete code file.
            # It is assumed that such files are stored under a folder named "{split}_passage_code".
            code_file_path = f"{self.code_path}/{self.split}_code/{question_id}.code"
            code = np.loadtxt(code_file_path).astype(int)
            # Map original codes to discrete codes via our lookup mapping
            code = np.vectorize(self.code_to_idx.get)(code)

            if code.ndim == 0:
                print(
                    "Debug: code is a scalar for question_id: ",
                    question_id,
                    " document_id: ",
                    document_id,
                    "=" * 10,
                )
                code = np.array([code])
            # if code.shape
            code = np.concatenate(
                [[self.special_token], code, [1]]
            )  # Append EOS token (assumed token id 1) # pick token 32000 as an indicate to query task (which is added token for flan t5)
            if len(code) > self.max_length:
                # print("Code length is too long, need to be truncated ===========")
                code = np.concatenate([code[: self.max_length - 1], [1]])
            if self.split == "train":
                if self.train_atomic:
                    return (
                        torch.LongTensor(code),
                        self.doc_id_2_idx[document_id] + self.corpus_atomic_offset,
                        self.doc_id_2_idx[document_id],
                    )
                else:
                    return (
                        torch.LongTensor(code),
                        document_id,
                        self.doc_id_2_idx[document_id],
                    )
            else:
                if self.train_atomic:
                    return (
                        torch.LongTensor(code),
                        self.doc_id_2_idx[document_id] + self.corpus_atomic_offset,
                        self.doc_id_2_idx[document_id],
                    )
                else:
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


class SlueSQA5DatasetContinuous(Dataset):
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

        self.split = split
        self.feature_path = feature_path
        self.max_length = max_length
        self.truncate_offset = truncate_offset
        self.special_token = special_token
        self.offset = offset
        self.downsample_factor = downsample_factor

        # Load main data CSV with question IDs and document IDs
        self.data = pd.read_csv(os.path.join(dataset_path, f"{split}.csv"))
        self.query_len = len(self.data)

        # Load corpus data for reference
        self.corpus_data = pd.read_csv(os.path.join(dataset_path, corpus_filename))

        # Load pq data to build document ID mappings
        pq_data = pd.read_csv(os.path.join(dataset_path, pq_filename))
        self.pq_data = pq_data.dropna(subset=["post_query", "document_id"])

        # Build mapping from document_id to a string identifier
        self.doc_id_2_id = {}
        for _, row in self.pq_data.iterrows():
            if row["document_id"] not in self.doc_id_2_id:
                self.doc_id_2_id[row["document_id"]] = str(row["idx"])

        # Initialize document ID to index mapping for model training
        self.doc_id_2_idx = {}
        # Build document id to index mapping
        for doc_id in self.corpus_data["document_id"]:
            if doc_id not in self.doc_id_2_idx:
                self.doc_id_2_idx[doc_id] = len(self.doc_id_2_idx)

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


@dataclass
class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{"input_ids": x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        inputs["query_doc_id"] = torch.Tensor([x[2] for x in features])
        return inputs
    
@dataclass
class IndexingCollatorWithAtomic(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{"input_ids": x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)
        labels = torch.Tensor([[label, self.tokenizer.eos_token_id] for label in docids])
        inputs["labels"] = labels
        inputs["query_doc_id"] = torch.Tensor([x[2] for x in features])
        return inputs


@dataclass
class IndexingCollatorWithMetadata(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{"input_ids": x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        inputs["query_doc_id"] = torch.Tensor([x[2] for x in features])

        return inputs


@dataclass
class ContinuousIndexingCollator(DataCollatorWithPadding):
    """
    Data collator for continuous speech features.
    Inherits from DataCollatorWithPadding to handle padding of continuous features.
    """

    def __call__(self, features):
        # Extract features, document IDs and indices
        feature_tensors = [x[0] for x in features]
        docids = [x[1] for x in features]
        doc_indices = [x[2] for x in features]

        # Create a dict of features that DataCollatorWithPadding can handle
        # We'll wrap our feature tensors in a format compatible with the parent class
        batch_dict = [{"input_ids": feat.tolist()} for feat in feature_tensors]

        # Use the parent class to handle padding
        padded_inputs = super().__call__(batch_dict)

        # Get padded features from the result
        padded_features = padded_inputs["input_ids"]

        # Convert back to tensor if needed
        if not isinstance(padded_features, torch.Tensor):
            padded_features = torch.tensor(padded_features)

        # Tokenize document IDs for labels
        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        # Replace padding token IDs with -100 for loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Create inputs dictionary with the appropriate keys
        inputs = {
            "input_features": padded_features,
            "attention_mask": padded_inputs["attention_mask"],
            "labels": labels,
            # "query_doc_id": torch.tensor(doc_indices, dtype=torch.long)
        }

        return inputs


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
        self.device = device
        self.include_corpus = include_corpus

        # SpecAugment settings
        self.apply_spec_augment = (
            apply_spec_augment and split == "train"
        )  # Only apply during training
        self.time_warp_param = time_warp_param
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

        # Initialize SpecAugment module if enabled
        if self.apply_spec_augment:
            self.spec_augment = SpecAugmentLB(
                time_warp_param=time_warp_param,
                freq_mask_param=freq_mask_param,
                time_mask_param=time_mask_param,
            )
            # Set to eval mode since we don't train this module
            self.spec_augment.eval()
        else:
            self.spec_augment = None

        logging.info(
            f"Initializing SlueSQA5WhisperDataset - split: {split}, device: {device}"
        )
        if self.apply_spec_augment:
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

        # Initialize Whisper model and processor
        self._load_whisper_model(whisper_model_name)

        # Build document id to index mapping from the dataset efficiently
        # We'll collect document IDs during corpus building to avoid multiple iterations
        self.doc_id_2_idx = {}
        self.all_doc_ids = set()

        # For training, we can include corpus documents if needed
        self.corpus_data = []
        if split == "train" and include_corpus:
            self.corpus_data = self._build_corpus_data(debug_max_samples)
        else:
            # For non-training splits, still need all doc IDs for constrained beam search
            self._collect_all_doc_ids_efficiently()

        self.valid_ids = list(self.all_doc_ids)

    def _load_whisper_model(self, model_name: str):
        """Load Whisper processor and model, move only encoder to specified device."""
        self.whisper_processor = WhisperProcessor.from_pretrained(model_name)
        self.whisper_model = WhisperModel.from_pretrained(model_name)

        # Move only encoder to device and set to eval mode to save GPU memory
        self.whisper_model.encoder = self.whisper_model.encoder.to(self.device)
        self.whisper_model.encoder.eval()

        # Keep the decoder on CPU since we don't use it
        self.whisper_model.decoder = self.whisper_model.decoder.cpu()

        logging.info(
            f"Loaded Whisper model {model_name} with encoder on device {self.device} and decoder on CPU"
        )

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
                        corpus_data.append(
                            {
                                "document_id": doc_id,
                                "document_audio": item["document_audio"]["array"],
                                "sampling_rate": item["document_audio"][
                                    "sampling_rate"
                                ],
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

    def _apply_spec_augment(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to Whisper log-mel spectrogram features using SpecAugmentLB.

        Args:
            input_features: Log-mel features of shape [batch, n_mels, time_steps]

        Returns:
            Augmented features of the same shape
        """
        if not self.apply_spec_augment or self.spec_augment is None:
            return input_features

        # Debug: Check input shape
        original_shape = input_features.shape

        # Apply SpecAugment using the SpecAugmentLB module
        with torch.no_grad():  # No gradients needed for augmentation
            augmented_features = self.spec_augment(input_features)

        # Debug: Check output shape and log if different
        output_shape = augmented_features.shape
        if original_shape != output_shape:
            logging.warning(
                f"SpecAugment changed shape from {original_shape} to {output_shape}"
            )

        return augmented_features

    def _extract_whisper_features(
        self, audio: np.ndarray, sampling_rate: int = 16000
    ) -> torch.Tensor:
        """Extract features using Whisper encoder."""
        try:
            # Process audio with whisper processor to get log-mel features
            inputs = self.whisper_processor(
                audio, sampling_rate=sampling_rate, return_tensors="pt"
            )

            # Get input_features (log-mel spectrogram)
            input_features = inputs.input_features  # Shape: [1, n_mels, time_steps]

            # Apply SpecAugment to log-mel features if enabled
            if self.apply_spec_augment:
                input_features = self._apply_spec_augment(input_features)

            # Move inputs to device
            input_features = input_features.to(self.device)

            # Extract features using encoder only
            with torch.no_grad():
                encoder_outputs = self.whisper_model.encoder(input_features)
                # Get the last hidden state
                features = encoder_outputs.last_hidden_state.squeeze(
                    0
                )  # Remove batch dimension

            return features.cpu()  # Move back to CPU for storage

        except Exception as e:
            logging.error(f"Error extracting Whisper features: {str(e)}")
            # Return dummy features as fallback
            return torch.zeros((1, self.whisper_model.config.d_model))

    def __len__(self):
        return self.query_len + len(self.corpus_data)

    def __getitem__(self, idx):
        if idx < self.query_len:
            # Handle query data
            item = self.data[idx]
            question_audio = item["question_audio"]["array"]
            sampling_rate = item["question_audio"]["sampling_rate"]
            document_id = item["document_id"]

            # Extract features from question audio
            features = self._extract_whisper_features(question_audio, sampling_rate)

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
            corpus_item = self.corpus_data[idx_adjusted]

            document_audio = corpus_item["document_audio"]
            sampling_rate = corpus_item["sampling_rate"]
            doc_id = corpus_item["document_id"]

            # Extract features from document audio
            features = self._extract_whisper_features(document_audio, sampling_rate)

            # No truncation - let Q-Former handle the sequence compression

            # Process document ID by replacing underscores with spaces
            processed_doc_id = self._process_docid(doc_id)

            return (
                features,
                processed_doc_id,
                -1,
            )  # we don't use self-neg currently, so keep it -1


@dataclass
class WhisperIndexingCollator(DataCollatorWithPadding):
    """
    Data collator for Whisper features in indexing tasks.
    Handles padding of continuous Whisper features.
    """

    def __call__(self, features):
        # Extract features, document IDs and indices
        feature_tensors = [
            x[0] for x in features
        ]  # List of [seq_len, feature_dim] tensors
        docids = [x[1] for x in features]
        doc_indices = [x[2] for x in features]

        # Pad feature tensors to same length
        max_seq_len = max(feat.shape[0] for feat in feature_tensors)
        feature_dim = feature_tensors[0].shape[1]

        padded_features = []
        attention_masks = []

        for feat in feature_tensors:
            seq_len = feat.shape[0]

            # Create attention mask
            attention_mask = torch.ones(max_seq_len)
            attention_mask[seq_len:] = 0
            attention_masks.append(attention_mask)

            # Pad features
            if seq_len < max_seq_len:
                padding = torch.zeros(max_seq_len - seq_len, feature_dim)
                padded_feat = torch.cat([feat, padding], dim=0)
            else:
                padded_feat = feat

            padded_features.append(padded_feat)

        # Stack into batch
        batched_features = torch.stack(padded_features)
        batched_attention_masks = torch.stack(attention_masks)

        # Tokenize document IDs for labels
        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        # Replace padding token IDs with -100 for loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Create inputs dictionary
        inputs = {
            "input_features": batched_features,  # [batch_size, max_seq_len, feature_dim]
            "attention_mask": batched_attention_masks,  # [batch_size, max_seq_len]
            "labels": labels,
            "query_doc_id": torch.tensor(doc_indices, dtype=torch.long),
        }

        return inputs


if __name__ == "__main__":

    # dataset = SlueSQA5DatasetV2(split="train")
    # print("length: ", len(dataset))
    # print(dataset.__getitem__(0), "first")
    # print(dataset.__getitem__(60000), "some doc task")
    # print(list(dataset.valid_ids)[:20], list(dataset.valid_ids)[-20:])

    # print("validation set")
    # dataset = SlueSQA5DatasetV2(split="validation")
    # print(dataset.__getitem__(0), "first")
    # print(dataset.__getitem__(1240), 'final')
    # print(list(dataset.valid_idzs)[:20], list(dataset.valid_ids)[-20:])

    # print("test set")
    test_dataset = SlueSQA5DatasetV2(
        split="train",
        code_path="/home/ricky/dodofk/dataset/slue_sqa_code_l22_c500",
        discrete_code_num=500,
        train_atomic=True,
        atomic_offset=50,
    )


    # for i in range(5):
    #     print("debug getitem: ", i)
    #     print(test_dataset.__getitem__(i))
    #     print("debug last item: ", len(test_dataset) - i - 1, test_dataset.__getitem__(len(test_dataset) - i - 1))
    collator = IndexingCollatorWithAtomic(tokenizer=test_dataset.tokenizer)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(test_dataset, batch_size=2, collate_fn=collator, num_workers=0)
    for idx, batch in enumerate(dataloader):
        if idx > 10:
            break
        print("Idx: ", idx)
        print(batch)
    # collator = IndexingCollator(tokenizer=test_dataset.tokenizer)s
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(test_dataset, batch_size=2, collate_fn=collator)
    # for idx, batch in enumerate(dataloader):
    #     if idx > 10:
    #         break
    #     print("Idx: ", idx)
    #     print(batch)

    # from torch.utils.data import DataLoader

    # dataloader = DataLoader(test_dataset, batch_size=2, collate_fn=collator)

    # label_length = []
    # for batch in dataloader:
    #     label_length.append(len(batch["labels"][0]))
    #     print(batch["labels"])
    #     break

    # test continuous dataset
    # from torch.utils.data import DataLoader

    # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    # test_dataset = SlueSQA5WhisperDataset(
    #     split="train",
    #     debug_max_samples=100,  # Debug with only 100 samples
    #     apply_spec_augment=True,  # Enable SpecAugment
    #     time_warp_param=80,
    #     freq_mask_param=27,
    #     time_mask_param=100,
    # )
    # collator = WhisperIndexingCollator(tokenizer=tokenizer)
    # dataloader = DataLoader(test_dataset, batch_size=2, collate_fn=collator)
    # for idx, batch in enumerate(dataloader):
    #     print(batch["input_features"].shape, batch["labels"].shape)
    #     if idx > 10:
    #         break

    # print(label_length)
    # print("avg length: ", np.mean(label_length))
    # print("std length: ", np.std(label_length))
    # print("max length: ", np.max(label_length))
    # print("min length: ", np.min(label_length))

    # for i in range(len(test_dataset)):
    #     print(test_dataset.__getitem__(i))
