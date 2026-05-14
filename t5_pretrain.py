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
from dataclasses import asdict, dataclass, field
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
from tqdm import tqdm
from hub_checkpoint import HubCheckpointArguments, build_hub_checkpoint_callbacks
from unit_store import PackedUnitStore, load_packed_store, resolve_unit_code_path

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _uses_wandb(report_to) -> bool:
    if report_to is None:
        return False
    if isinstance(report_to, str):
        values = [report_to]
    else:
        values = list(report_to)
    values = {str(value).lower() for value in values}
    return "wandb" in values or "all" in values

#############################################
# Helper Function: T5-Style Span Masking
#############################################


def _random_segmentation(num_items: int, num_segments: int) -> List[int]:
    """Partition num_items into num_segments non-empty random segments."""
    if num_segments <= 0 or num_segments > num_items:
        raise ValueError(
            f"num_segments must be in [1, {num_items}], got {num_segments}"
        )
    if num_segments == 1:
        return [num_items]

    cuts = sorted(random.sample(range(1, num_items), num_segments - 1))
    return [end - start for start, end in zip([0] + cuts, cuts + [num_items])]


def _random_spans_noise_mask(
    length: int,
    mask_prob: float,
    mean_span_length: int,
) -> List[bool]:
    """Return a non-overlapping T5-style span mask."""
    if length < 2:
        return [True] * length

    num_noise_tokens = int(round(length * mask_prob))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(round(num_noise_tokens / mean_span_length))
    num_noise_spans = min(max(num_noise_spans, 1), num_noise_tokens)
    num_noise_spans = min(num_noise_spans, length - num_noise_tokens)

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(
        length - num_noise_tokens,
        num_noise_spans,
    )

    mask = []
    for nonnoise_len, noise_len in zip(nonnoise_span_lengths, noise_span_lengths):
        mask.extend([False] * nonnoise_len)
        mask.extend([True] * noise_len)
    return mask[:length]


def random_spans_noise_masking(
    token_ids: Union[List[int], torch.Tensor],
    mask_prob: float = 0.15,
    mean_span_length: int = 3,
    sentinel_start_id: int = 32000,
    sentinel_direction: int = 1,
    eos_token_id: Optional[int] = 1,
) -> Tuple[List[int], List[int]]:
    """
    Applies T5-style span masking to a sequence of token IDs.

    Args:
        token_ids (list or torch.Tensor): The original token sequence.
        mask_prob (float): Fraction of tokens to mask.
        mean_span_length (int): Average length of each masked span.
        sentinel_start_id (int): The starting ID for the extra/sentinel tokens.
        sentinel_direction (int): Use -1 for standard T5 extra_id ordering.
        eos_token_id (int): Token appended to both input and target sequences.

    Returns:
        masked_input_ids (list): Token sequence with masked spans replaced by sentinel tokens.
        labels (list): Compact T5 target:
                       sentinel_0 masked_span_0 sentinel_1 masked_span_1 ... eos
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    if sentinel_direction not in {-1, 1}:
        raise ValueError("sentinel_direction must be 1 or -1")
    if not token_ids:
        return ([eos_token_id] if eos_token_id is not None else []), (
            [eos_token_id] if eos_token_id is not None else []
        )

    mask = _random_spans_noise_mask(
        length=len(token_ids),
        mask_prob=mask_prob,
        mean_span_length=mean_span_length,
    )
    masked_input_ids: List[int] = []
    labels: List[int] = []
    span_idx = -1
    in_noise_span = False

    for token_id, is_noise in zip(token_ids, mask):
        if is_noise:
            if not in_noise_span:
                span_idx += 1
                sentinel_id = sentinel_start_id + sentinel_direction * span_idx
                masked_input_ids.append(sentinel_id)
                labels.append(sentinel_id)
                in_noise_span = True
            labels.append(token_id)
        else:
            masked_input_ids.append(token_id)
            in_noise_span = False

    if eos_token_id is not None:
        masked_input_ids.append(eos_token_id)
        labels.append(eos_token_id)
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
        sentinel_direction: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(tokenizer=tokenizer, model=model, **kwargs)
        self.mask_prob: float = mask_prob
        self.mean_span_length: int = mean_span_length
        self.sentinel_start_id: int = sentinel_start_id
        self.sentinel_direction: int = sentinel_direction
        self.eos_token_id: Optional[int] = tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        new_features: List[Dict[str, Any]] = []
        for f in features:
            token_ids: Union[List[int], torch.Tensor] = f["input_ids"]
            masked_input_ids, labels = random_spans_noise_masking(
                token_ids,
                mask_prob=self.mask_prob,
                mean_span_length=self.mean_span_length,
                sentinel_start_id=self.sentinel_start_id,
                sentinel_direction=self.sentinel_direction,
                eos_token_id=self.eos_token_id,
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
        code_dir: str = "/home/ricky/dodofk/dataset/ll6k_code_l22_c500",
        discrete_code_num: int = 500,
        split: str = "train",
        token_file: str = "/home/ricky/dodofk/dataset/slue_sqa5/flan-t5-base-unused_tokens.txt",
        validation_fraction: float = 0.08,
        min_chunk_length: int = 64,
        sentinel_start_id: Optional[int] = None,
        sentinel_direction: int = 1,
        max_sentinels: int = 100,
    ):
        self.discrete_code_num: int = discrete_code_num
        self.code_dir: str = resolve_unit_code_path(code_dir)
        self.max_length: int = max_length
        self.packed_store = self._load_store(self.code_dir)
        self.code_files: List[str] = []
        self.record_ids: List[str] = []
        if self.packed_store is None:
            self.code_files = sorted(glob.glob(os.path.join(self.code_dir, "*.code")))
        else:
            self.record_ids = sorted(self.packed_store.ids)
        # debug only to keep the code small
        self.chunk_offset: int = chunk_offset
        self.validation_fraction = validation_fraction
        self.min_chunk_length = min_chunk_length
        
        assert split in ["train", "val"], "split must be either train or val"
        if self.packed_store is None and not self.code_files:
            raise ValueError(
                f"No .code files or packed librispeech.npz found under {code_dir}"
            )
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if not 0 <= self.chunk_offset < self.max_length:
            raise ValueError("chunk_offset must be smaller than max_length")
        self.split = split
        self.code_lookup = np.loadtxt(token_file, dtype=int)[: self.discrete_code_num]
        if len(self.code_lookup) < self.discrete_code_num:
            raise ValueError(
                f"token_file only provides {len(self.code_lookup)} token ids, "
                f"but discrete_code_num={self.discrete_code_num}"
            )
        if sentinel_start_id is not None:
            sentinel_ids = {
                sentinel_start_id + sentinel_direction * i for i in range(max_sentinels)
            }
            collisions = sorted(set(self.code_lookup.tolist()) & sentinel_ids)
            if collisions:
                raise ValueError(
                    "Sentinel ids collide with unit token ids: "
                    f"{collisions[:10]}. Regenerate token_file or change sentinel ids."
                )
        
        
        # preprocess the codes
        self.codes: List[np.ndarray] = self.build_codes()
    
    @staticmethod
    def _load_store(code_dir: str) -> Optional[PackedUnitStore]:
        if code_dir.endswith(".npz"):
            return load_packed_store(code_dir)
        return load_packed_store(os.path.join(code_dir, "librispeech.npz"))

    def _split_items(self) -> List[str]:
        items = self.record_ids if self.packed_store is not None else self.code_files
        split_idx = int(round((1.0 - self.validation_fraction) * len(items)))
        split_idx = min(max(split_idx, 1), len(items) - 1)
        if self.split == "train":
            return items[:split_idx]
        return items[split_idx:]

    @staticmethod
    def _read_code_file(path: str) -> np.ndarray:
        code = np.loadtxt(path, dtype=int)
        if code.ndim == 0:
            return np.array([int(code)])
        if code.ndim == 1:
            return code
        return code[0]

    def build_codes(self) -> List[np.ndarray]:
        codes: List[np.ndarray] = []
        split_items = self._split_items()
        for item in tqdm(split_items, desc=f"Building codes for {self.split} set"):
            if self.packed_store is None:
                code = self._read_code_file(item)
            else:
                code = self.packed_store.get_code(item)
            
            # convert code to tokens
            code = np.array([self.code_lookup[c] for c in code])
            
            stride = self.max_length - self.chunk_offset
            if len(code) < self.max_length:
                if len(code) >= self.min_chunk_length:
                    codes.append(code)
            else:
                for i in range(0, len(code), stride):
                    if i + self.max_length > len(code):
                        chunk = code[i:]
                    else:
                        chunk = code[i:i+self.max_length]
                    if len(chunk) >= self.min_chunk_length:
                        codes.append(chunk)
                  
        logging.info("%s set size: %d", self.split, len(codes))
        return codes

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
        default=32099,
        metadata={
            "help": "Starting ID for extra/sentinel tokens (e.g. T5 extra tokens)"
        },
    )
    sentinel_direction: int = field(
        default=-1,
        metadata={"help": "Use -1 for standard T5 extra_id_0, extra_id_1 ordering"},
    )
    final_model_dir: str = field(
        default="flan_t5_span_masking_final",
        metadata={"help": "Directory to save the final model checkpoint"},
    )
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional checkpoint/model path to initialize from"},
    )
    save_final_model: bool = field(
        default=True,
        metadata={"help": "Save final model after training."},
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
        default="/home/ricky/dodofk/dataset/ll6k_code_l22_c500",
        metadata={"help": "Directory to the code dataset"},
    )
    token_file: str = field(
        default="/home/ricky/dodofk/dataset/slue_sqa5/flan-t5-base-unused_tokens.txt",
        metadata={"help": "File mapping discrete unit ids to tokenizer ids"},
    )
    discrete_code_num: int = field(
        default=500, metadata={"help": "Number of discrete code in the dataset"},
    )
    validation_fraction: float = field(
        default=0.08, metadata={"help": "Fraction of files reserved for validation"}
    )
    min_chunk_length: int = field(
        default=64, metadata={"help": "Drop shorter tail chunks during pretraining"}
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
        (
            ModelArguments,
            DataTrainingArguments,
            WandBArguments,
            HubCheckpointArguments,
            TrainingArguments,
        )
    )
    model_args, data_args, wandb_args, hub_args, training_args = (
        parser.parse_args_into_dataclasses()
    )

    use_wandb = _uses_wandb(training_args.report_to)
    if use_wandb and training_args.process_index == 0:
        wandb_init_args: Dict[str, Any] = {
            "project": wandb_args.project,
            "name": training_args.run_name,
            "config": {
                "training": training_args.to_dict(),
                "model": asdict(model_args),
                "data": asdict(data_args),
            },
        }
        if wandb_args.description is not None:
            wandb_init_args["notes"] = wandb_args.description
        wandb.init(**wandb_init_args)

    logger.info("Training/evaluation parameters: %s", training_args)

    # 1. Load the model and tokenizer.
    model_load_path = model_args.model_path or model_args.model_name_or_path
    model = T5ForConditionalGeneration.from_pretrained(model_load_path)
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
        token_file=data_args.token_file,
        validation_fraction=data_args.validation_fraction,
        min_chunk_length=data_args.min_chunk_length,
        sentinel_start_id=model_args.sentinel_start_id,
        sentinel_direction=model_args.sentinel_direction,
        split="train",
    )
    eval_dataset = DiscreteCodeDataset(
        max_length=data_args.seq_length,
        chunk_offset=data_args.chunk_offset,
        code_dir=data_args.code_dir,
        discrete_code_num=data_args.discrete_code_num,
        token_file=data_args.token_file,
        validation_fraction=data_args.validation_fraction,
        min_chunk_length=data_args.min_chunk_length,
        sentinel_start_id=model_args.sentinel_start_id,
        sentinel_direction=model_args.sentinel_direction,
        split="val",
    )

    # 4. Create the custom data collator.
    data_collator = DataCollatorForT5SpanCorruption(
        tokenizer=tokenizer,
        model=model,
        mask_prob=data_args.mask_prob,
        mean_span_length=data_args.mean_span_length,
        sentinel_start_id=model_args.sentinel_start_id,
        sentinel_direction=model_args.sentinel_direction,
    )

    logging.info("Set up the trainer")
    # 5. Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=build_hub_checkpoint_callbacks(hub_args, tokenizer=tokenizer),
    )

    logging.info("Start training")
    # 6. Start training.
    trainer.train()

    # 7. Save the final model checkpoint to the specified directory.
    if model_args.save_final_model:
        trainer.save_model(model_args.final_model_dir)
    else:
        logger.info("Skipping final model save because save_final_model=False")
    if use_wandb and training_args.process_index == 0 and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
