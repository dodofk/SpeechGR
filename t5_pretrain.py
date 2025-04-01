#!/usr/bin/env python
"""
This code is used to pretrain a T5 model on our spoken discrete unit.

Author: Ricky Liu
"""
import random
import torch
import logging
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
)
import wandb
import glob
import os
import numpy as np

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
                    "input_ids": masked_input_ids,
                    "labels": labels,
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
        code_dir: str = "/home/ricky/dodofk/dataset/medium",
        discrete_code_num: int = 500,
    ):
        self.discrete_code_num: int = discrete_code_num
        self.code_dir: str = code_dir
        self.max_length: int = max_length
        self.code_files: List[str] = glob.glob(os.path.join(code_dir, "*.code"))

    def __len__(self) -> int:
        return len(self.code_files)


class DummyDataset(Dataset):
    def __init__(
        self, tokenizer: T5Tokenizer, num_samples: int = 1000, seq_length: int = 512
    ) -> None:
        self.tokenizer: T5Tokenizer = tokenizer
        self.num_samples: int = num_samples
        self.seq_length: int = seq_length

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Create a dummy sequence by sampling random token IDs.
        token_ids: List[int] = [
            random.randint(0, self.tokenizer.vocab_size - 1)
            for _ in range(self.seq_length)
        ]
        return {"input_ids": token_ids}


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
        default=32000,
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
        default=0.15, metadata={"help": "Masking probability for T5 span corruption"}
    )
    mean_span_length: int = field(
        default=3, metadata={"help": "Mean span length for T5 span corruption"}
    )
    train_samples: int = field(
        default=1000,
        metadata={"help": "Number of training samples in the dummy dataset"},
    )
    eval_samples: int = field(
        default=100,
        metadata={"help": "Number of evaluation samples in the dummy dataset"},
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


def main() -> None:
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, WandBArguments, TrainingArguments)
    )
    model_args, data_args, wandb_args, training_args = (
        parser.parse_args_into_dataclasses()
    )

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
    train_dataset = DummyDataset(
        tokenizer, num_samples=data_args.train_samples, seq_length=data_args.seq_length
    )
    eval_dataset = DummyDataset(
        tokenizer, num_samples=data_args.eval_samples, seq_length=data_args.seq_length
    )

    # 4. Create the custom data collator.
    data_collator = DataCollatorForT5SpanCorruption(
        tokenizer=tokenizer,
        model=model,
        mask_prob=data_args.mask_prob,
        mean_span_length=data_args.mean_span_length,
        sentinel_start_id=model_args.sentinel_start_id,
    )

    # 5. Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # 6. Start training.
    trainer.train()

    # 7. Save the final model checkpoint to the specified directory.
    trainer.save_model(model_args.final_model_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
