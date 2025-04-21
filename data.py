"""
This code is modified from github `ArvinZhuang/DSI-QG` repo.

The file contains dataset class for our speech DSI task.
"""

from dataclasses import dataclass
from tqdm import tqdm
import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding, AutoTokenizer
from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
import pandas as pd
import os
from typing import Optional, Tuple, List
import h5py
import logging

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
        code_path: str = "/home/ricky/dodofk/dataset/slue_sqa_code_c512",
        pq_filename: str = "slue_sqa5_pq10_llama32_3b_clean.csv",
        corpus_filename: str = "slue_sqa5_corpus.csv",
        model_name_or_path: str = "google/flan-t5-base",
        epoch: Optional[int] = None,
        discrete_code_num: int = 128,
        truncate_offset: int = 50,
        special_token: int = 32000,  # specific the special token to use for query task
        lookup_file_name: Optional[str] = None,
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

        self.valid_ids = list(
            set(self.corpus_doc_id)
        )  # try to change numberic-id to docid

    def build_code_lookup(self):
        """
        Build a discrete code lookup based on the post-query text from pq data.
        This reuses logic from the original SlueSQA5Dataset.
        """
        if self.lookup_file_name:
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
                raise ValueError(f"Not enough unused tokens to build a discrete code lookup. Got {len(unused_tokens)} tokens, but {self.discrete_code_num} are required.")
            
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
        for doc_id in self.corpus_data["document_id"]:
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
                print("Debug: code is a scalar for question_id: ", question_id, " document_id: ", document_id, "="*10)
                code = np.array([code])
            # if code.shape 
            code = np.concatenate(
                [[self.special_token], code, [1]]
            )  # Append EOS token (assumed token id 1) # pick token 32000 as an indicate to query task (which is added token for flan t5)
            if len(code) > self.max_length:
                # print("Code length is too long, need to be truncated ===========")
                code = np.concatenate([code[: self.max_length - 1], [1]])
            return torch.LongTensor(code), document_id
        else:
            # For extra PQ data (used for indexing), only used in train
            idx_adjusted = idx - self.query_len
            code = self.corpus_code_data[idx_adjusted]
            # label = self.corpus_code_label[idx_adjusted]
            label = self.corpus_doc_id[idx_adjusted]
            return torch.LongTensor(code), label

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
    def __init__(
        self,
        split: str = "train",
        max_length: int = 512,
        dataset_path: str = "/home/ricky/dodofk/dataset/slue_sqa5/",
        feature_path: str = "/home/ricky/dodofk/dataset/slue_sqa5_wavlm_large",
        truncate_offset: int = 50,
        special_token: int = 32000,
        downsample_factor: int = 2,
        offset: int = 50,
    ):
        """
        The dataset is used for continuous speech DSI task.
        """
        assert split in ["train", "validation", "test", "verified_test"]
        
        self.input, self.label = [], []
        self.max_length = max_length
        self.truncate_offset = truncate_offset
        self.special_token = special_token
        
        
        if self.split == "train":
            # only train data need to train with corpus
            self._load_corpus()   
        
        # load split data
        self._load_data()
            
    
    def _load_data(self):
        """
        This function load correspond split query task data 
        """
        # load only train data
        split_h5_filepath = os.path.join(self.feature_path, f"{self.split}.h5")
        split_h5_file = h5py.File(split_h5_filepath, "r")
        
        self.features = split_h5_file["features"]
        self.labels = split_h5_file["labels"]
        
        for feature, label in zip(self.features, self.labels):
            feature = np.array(feature)
            
            self.label.append(label)
            self.input.append(self._preprocess_query_data(feature))
                    
    def _load_corpus(self):
        corpus_h5_filepath = os.path.join(self.feature_path, "slue_sqa5_corpus.h5")
        corpus_h5_file = h5py.File(corpus_h5_filepath, "r")
        self.corpus_ids = corpus_h5_file["ids"]
        self.corpus_features = corpus_h5_file["features"]
        
        for docid, feature in zip(self.corpus_ids, self.corpus_features):
            feature = np.array(feature)
            
            if len(feature) > self.max_length:
                chunks = self._truncate_data(feature)
                for chunk in chunks:
                    self.input.append(chunk)
                    self.label.append(docid)
            else:
                self.input.append(feature)
                self.label.append(docid)
    
    def _truncate_index_data(self, feature: np.ndarray) -> List[np.ndarray]:
        chunks = []
        
        start_idx, reach_end = 0, False
        while start_idx < len(feature) and not reach_end:
            end_idx = min(start_idx + self.max_length, len(feature))
            if end_idx == len(feature):
                reach_end = True
            chunks.append(feature[start_idx:end_idx])
            start_idx = end_idx - self.truncate_offset
            
        return chunks
    
    def _preprocess_query_data(self, feature: np.ndarray) -> List[np.ndarray]:
        """
        We truncate the feature to max_length - 2, and add special token and eos token to the feature.
        
        Do not make them to multiple chunk for simplicity.
        """
        feature = np.concatenate([[self.special_token], feature, [1]])

        if len(feature) > self.max_length:
            feature = np.concatenate([feature[:self.max_length - 1], [1]])
            
        return feature
            
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        return torch.Tensor(self.input[idx]), self.label[idx]

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
    # print(list(dataset.valid_ids)[:20], list(dataset.valid_ids)[-20:])

    print("test set")
    test_dataset = SlueSQA5DatasetV2(
        split="validation",
        code_path="/home/ricky/dodofk/dataset/slue_sqa_code_l22_c1000",
        discrete_code_num=1000,
    )
    # collator = IndexingCollator(tokenizer=test_dataset.tokenizer)
    
    for i in range(len(test_dataset)):
        print(test_dataset.__getitem__(i))

