#!/usr/bin/env python
"""
This code is used to pretrain a T5 model on our spoken discrete unit.

Author: Ricky Liu
"""
import math
import random
import torch
import logging
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path

from omegaconf import DictConfig
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
import wandb
import glob
import os
import numpy as np
from tqdm import tqdm

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#############################################
# Helper Function: T5-Style Span Masking
#############################################


def random_spans_noise_masking(
    token_ids: Union[List[int], torch.Tensor],
    mask_prob: float = 0.15,
    mean_span_length: int = 3,
    sentinel_start_id: int = 32000,
) -> Tuple[List[int], List[int]]:
    """
    Applies T5-style span masking to a sequence of token IDs.

    Args:
        token_ids (list or torch.Tensor): The original token sequence.
        mask_prob (float): Fraction of tokens to mask.
        mean_span_length (int): Average length of each masked span.
        sentinel_start_id (int): The starting ID for the extra/sentinel tokens.

    Returns:
        masked_input_ids (list): Token sequence with masked spans replaced by sentinel tokens.
        labels (list): Sequence with the masked spans (preceded by their sentinel tokens)
                       and -100 for unmasked tokens.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    num_tokens: int = len(token_ids)
    num_to_mask: int = int(round(num_tokens * mask_prob))
    if num_to_mask == 0:
        return token_ids, [-100] * num_tokens

    # Generate random span lengths until covering num_to_mask tokens.
    spans: List[int] = []
    remaining_to_mask: int = num_to_mask
    while remaining_to_mask > 0:
        current_span_len: int = min(
            random.randint(1, mean_span_length * 2), remaining_to_mask
        )
        spans.append(current_span_len)
        remaining_to_mask -= current_span_len

    # Randomly choose start indices for each span and sort them.
    span_starts: List[int] = random.sample(range(num_tokens), k=len(spans))
    span_starts.sort()

    masked_input_ids: List[int] = []
    labels: List[int] = []
    current_pos: int = 0
    current_sentinel_id: int = sentinel_start_id

    for span_start, span_length in sorted(zip(span_starts, spans), key=lambda x: x[0]):
        if span_start < current_pos:
            continue  # skip overlapping spans
        # Copy tokens before the masked span.
        while current_pos < span_start:
            masked_input_ids.append(token_ids[current_pos])
            labels.append(-100)
            current_pos += 1

        # Insert sentinel token into the input.
        masked_input_ids.append(current_sentinel_id)
        labels.append(-100)

        # In the labels, output the sentinel token followed by the original span tokens.
        labels.append(current_sentinel_id)
        for i in range(span_length):
            masked_idx: int = span_start + i
            if masked_idx >= num_tokens:
                break
            labels.append(token_ids[masked_idx])
        current_sentinel_id += 1
        current_pos = span_start + span_length

    # Copy remaining tokens.
    while current_pos < num_tokens:
        masked_input_ids.append(token_ids[current_pos])
        labels.append(-100)
        current_pos += 1

    return masked_input_ids, labels


#############################################
# Data Collator with T5 Span Masking
#############################################


class DataCollatorForT5SpanCorruption(DataCollatorForSeq2Seq):
    """
    Custom data collator that applies T5-style random span masking on the fly.
    Assumes each example is a dict with key "input_ids" (a list of token IDs).
    """

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        model: Optional[T5ForConditionalGeneration] = None,
        mask_prob: float = 0.15,
        mean_span_length: int = 3,
        sentinel_start_id: int = 32000,
        **kwargs: Any,
    ) -> None:
        super().__init__(tokenizer=tokenizer, model=model, **kwargs)
        self.mask_prob: float = mask_prob
        self.mean_span_length: int = mean_span_length
        self.sentinel_start_id: int = sentinel_start_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        new_features: List[Dict[str, Any]] = []
        for f in features:
            token_ids: Union[List[int], torch.Tensor] = f["input_ids"]
            masked_input_ids, labels = random_spans_noise_masking(
                token_ids,
                mask_prob=self.mask_prob,
                mean_span_length=self.mean_span_length,
                sentinel_start_id=self.sentinel_start_id,
            )
            new_features.append(
                {
                    "input_ids": torch.LongTensor(masked_input_ids),
                    "labels": torch.LongTensor(labels),
                }
            )
        batch: Dict[str, Any] = super().__call__(new_features)
        return batch


#############################################
# Dummy Dataset (Replace with Your Own)
#############################################


class DiscreteCodeDataset(Dataset):
    """
    This data is used for all kinds of dataset to load all the code.
    """

    def __init__(
        self,
        max_length: int = 512,
        chunk_offset: int = 20,
        code_dir: str = "outputs/slue_wavtok/precomputed",
        discrete_code_num: int = 500,
        split: str = "train",
        token_file: str = "outputs/slue_wavtok/flan-t5-base-unused_tokens.txt",
    ):
        self.discrete_code_num: int = discrete_code_num
        self.code_dir: str = code_dir
        self.max_length: int = max_length
        self.code_files: List[str] = glob.glob(os.path.join(code_dir, "*.code"))
        self.code_files = sorted(self.code_files)  # sort to make it deterministic 
        # debug only to keep the code small
        self.chunk_offset: int = chunk_offset
        
        assert split in ["train", "val"], "split must be either train or val"
        self.split = split
        self.code_lookup = np.loadtxt(token_file, dtype=int)
        
        
        # preprocess the codes
        self.codes: List[np.ndarray] = self.build_codes()
    

    def build_codes(self) -> List[np.ndarray]:
        codes: List[np.ndarray] = []
        for file in tqdm(self.code_files, desc=f"Building codes for {self.split} set"):
            code_cnt = np.loadtxt(
                file,
                dtype=int,
            )
            
            # split the code into chunks
            code, _ = code_cnt[0], code_cnt[1]
            
            # convert code to tokens
            code = np.array([self.code_lookup[c] for c in code])
            
            stride = self.max_length - self.chunk_offset
            if len(code) < self.max_length:
                codes.append(code)
            else:
                for i in range(0, len(code), stride):
                    if i + self.max_length > len(code):
                        codes.append(code[i:])
                    else:
                        codes.append(code[i:i+self.max_length])
                  
        if self.split == "train":
            logging.info(f"Training set size: {int(0.92 * len(codes))}")
            return codes[:int(0.92 * len(codes))]
        else: # left 8% for validation
            logging.info(f"Validation set size: {len(codes) - int(0.92 * len(codes))}")
            return codes[int(0.92 * len(codes)):]

    def __len__(self) -> int:
        return len(self.codes)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "input_ids": self.codes[idx],
        }

#############################################
# Dataclass Arguments
#############################################


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="google/flan-t5-base",
        metadata={
            "help": "Path to the pretrained model or model identifier from huggingface.co/models"
        },
    )
    sentinel_start_id: int = field(
        default=32001,
        metadata={
            "help": "Starting ID for extra/sentinel tokens (e.g. T5 extra tokens)"
        },
    )
    final_model_dir: str = field(
        default="flan_t5_span_masking_final",
        metadata={"help": "Directory to save the final model checkpoint"},
    )


@dataclass
class DataTrainingArguments:
    seq_length: int = field(
        default=512, metadata={"help": "Input sequence length for training examples"}
    )
    mask_prob: float = field(
        default=0.2, metadata={"help": "Masking probability for T5 span corruption"}
    )
    mean_span_length: int = field(
        default=5, metadata={"help": "Mean span length for T5 span corruption"}
    )
    train_samples: int = field(
        default=1000,
        metadata={"help": "Number of training samples in the dummy dataset"},
    )
    eval_samples: int = field(
        default=100,
        metadata={"help": "Number of evaluation samples in the dummy dataset"},
    )
    chunk_offset: int = field(
        default=20, metadata={"help": "Chunk offset for the code dataset"},
    )
    code_dir: str = field(
        default="outputs/slue_wavtok/precomputed",
        metadata={"help": "Directory to the code dataset"},
    )
    discrete_code_num: int = field(
        default=500, metadata={"help": "Number of discrete code in the dataset"},
    )


@dataclass
class WandBArguments:
    project: str = field(
        default="t5-span-masking", metadata={"help": "WandB project name"}
    )
    description: Optional[str] = field(
        default=None, metadata={"help": "Project description/notes for WandB"}
    )


#############################################
# Main Training Script
#############################################


def run(cfg: DictConfig) -> None:
    model_args = ModelArguments(**cfg.model)
    data_args = DataTrainingArguments(**cfg.data)
    wandb_args = WandBArguments(**cfg.wandb)
    training_args = TrainingArguments(**cfg.training.training_args)

    # Ensure wandb is used for logging.
    if not training_args.report_to or len(training_args.report_to) == 0:
        training_args.report_to = ["wandb"]

    # Set up wandb init arguments.
    wandb_init_args: Dict[str, Any] = {
        "project": wandb_args.project,
        "config": training_args.to_dict(),
    }
    if wandb_args.description is not None:
        wandb_init_args["notes"] = wandb_args.description

    wandb.init(**wandb_init_args)

    logger.info("Training/evaluation parameters: %s", training_args)

    # 1. Load the model and tokenizer.
    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(model_args.model_name_or_path)

    # 2. (Optional) If adding new tokens, update the tokenizer and resize model embeddings.
    # new_tokens = ["<audio_0>", "<audio_1>", ...]
    # tokenizer.add_tokens(new_tokens)
    # model.resize_token_embeddings(len(tokenizer))

    # 3. Create training and evaluation datasets.
    logging.info("Create training and evaluation datasets")
    train_dataset = DiscreteCodeDataset(
        max_length=data_args.seq_length,
        chunk_offset=data_args.chunk_offset,
        code_dir=data_args.code_dir,
        discrete_code_num=data_args.discrete_code_num,
        split="train",
    )
    eval_dataset = DiscreteCodeDataset(
        max_length=data_args.seq_length,
        chunk_offset=data_args.chunk_offset,
        code_dir=data_args.code_dir,
        discrete_code_num=data_args.discrete_code_num,
        split="val",
    )

    # 4. Create the custom data collator.
    data_collator = DataCollatorForT5SpanCorruption(
        tokenizer=tokenizer,
        model=model,
        mask_prob=data_args.mask_prob,
        mean_span_length=data_args.mean_span_length,
        sentinel_start_id=model_args.sentinel_start_id,
    )

    logging.info("Set up the trainer")
    # 5. Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logging.info("Start training")
    # 6. Start training.
    trainer.train()

    Path(model_args.final_model_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(model_args.final_model_dir)
    wandb.finish()
