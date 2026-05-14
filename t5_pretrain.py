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
from pathlib import Path
import numpy as np
from tqdm import tqdm
from hub_checkpoint import HubCheckpointArguments, build_hub_checkpoint_callbacks
from unit_store import (
    DEFAULT_SLUE_UNIT_HF_PATH,
    PACKED_SLUE_PATTERNS,
    PackedUnitStore,
    load_packed_store,
    resolve_unit_code_path,
)
from unit_token_lookup import (
    DEFAULT_LOOKUP_TEXT_COLUMN,
    DEFAULT_TOKEN_LOOKUP_PATH,
    attach_lookup_to_config,
    build_lookup_from_csv,
    build_reserved_safe_token_lookup,
    load_token_lookup,
    save_token_lookup,
)

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
    token_ids: Union[List[int], torch.Tensor, np.ndarray],
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
        token_ids = token_ids.detach().cpu().tolist()
    elif isinstance(token_ids, np.ndarray):
        token_ids = token_ids.reshape(-1).tolist()
    else:
        token_ids = list(token_ids)

    if sentinel_direction not in {-1, 1}:
        raise ValueError("sentinel_direction must be 1 or -1")
    if len(token_ids) == 0:
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
        code_dir: str = DEFAULT_SLUE_UNIT_HF_PATH,
        discrete_code_num: int = 500,
        split: str = "train",
        token_file: Optional[str] = DEFAULT_TOKEN_LOOKUP_PATH,
        tokenizer_name_or_path: str = "google/flan-t5-base",
        lookup_source_csv: Optional[str] = None,
        lookup_text_column: str = DEFAULT_LOOKUP_TEXT_COLUMN,
        validation_fraction: float = 0.08,
        min_chunk_length: int = 64,
        sentinel_start_id: Optional[int] = None,
        sentinel_direction: int = 1,
        max_sentinels: int = 100,
    ):
        self.discrete_code_num: int = discrete_code_num
        self.code_dir: str = resolve_unit_code_path(
            code_dir,
            allow_patterns=PACKED_SLUE_PATTERNS,
        )
        self.max_length: int = max_length
        self.packed_stores = self._load_stores(self.code_dir)
        self.code_files: List[str] = []
        self.record_items: List[Tuple[int, str]] = []
        if not self.packed_stores:
            self.code_files = sorted(glob.glob(os.path.join(self.code_dir, "*.code")))
        else:
            for store_idx, store in enumerate(self.packed_stores):
                self.record_items.extend(
                    (store_idx, record_id) for record_id in sorted(store.ids)
                )
        # debug only to keep the code small
        self.chunk_offset: int = chunk_offset
        self.validation_fraction = validation_fraction
        self.min_chunk_length = min_chunk_length
        self.token_file = token_file
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.lookup_source_csv = lookup_source_csv
        self.lookup_text_column = lookup_text_column
        
        assert split in ["train", "val"], "split must be either train or val"
        if not self.packed_stores and not self.code_files:
            raise ValueError(
                f"No .code files or packed .npz unit stores found under {code_dir}"
            )
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if not 0 <= self.chunk_offset < self.max_length:
            raise ValueError("chunk_offset must be smaller than max_length")
        self.split = split
        self.code_lookup = self._load_or_build_code_lookup()
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
    def _load_stores(code_dir: str) -> List[PackedUnitStore]:
        if code_dir.endswith(".npz"):
            store = load_packed_store(code_dir)
            return [store] if store is not None else []

        stores: List[PackedUnitStore] = []
        for filename in [
            "librispeech.npz",
            "documents.npz",
            "train.npz",
            "validation.npz",
            "test.npz",
            "verified_test.npz",
        ]:
            store = load_packed_store(os.path.join(code_dir, filename))
            if store is not None:
                logging.info(
                    "Loaded packed unit store %s",
                    os.path.join(code_dir, filename),
                )
                stores.append(store)
        return stores

    def _load_or_build_code_lookup(self) -> np.ndarray:
        if self.token_file and Path(self.token_file).exists():
            logging.info("Loading unit token lookup from %s", self.token_file)
            return load_token_lookup(self.token_file, self.discrete_code_num)

        if self.lookup_source_csv:
            lookup = build_lookup_from_csv(
                csv_path=self.lookup_source_csv,
                tokenizer_name_or_path=self.tokenizer_name_or_path,
                discrete_code_num=self.discrete_code_num,
                text_column=self.lookup_text_column,
            )
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
            lookup = build_reserved_safe_token_lookup(
                tokenizer=tokenizer,
                discrete_code_num=self.discrete_code_num,
            )
        if self.token_file:
            output_path = save_token_lookup(self.token_file, lookup)
            logging.info("Saved generated unit token lookup to %s", output_path)
        return lookup

    def _split_items(self) -> List[str]:
        items = self.record_items if self.packed_stores else self.code_files
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
            if self.packed_stores:
                store_idx, record_id = item
                code = self.packed_stores[store_idx].get_code(record_id)
            else:
                code = self._read_code_file(item)
            
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
        default=DEFAULT_SLUE_UNIT_HF_PATH,
        metadata={
            "help": "Directory, .npz file, or hf://datasets/... path for packed unit data"
        },
    )
    token_file: Optional[str] = field(
        default=DEFAULT_TOKEN_LOOKUP_PATH,
        metadata={
            "help": "File mapping discrete unit ids to tokenizer ids. "
            "If it does not exist, it is generated from tokenizer reserved-token rules."
        },
    )
    lookup_source_csv: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional CSV used to avoid text-token collisions when generating token_file. "
            "If omitted, lookup is generated only from tokenizer reserved-token rules."
        },
    )
    lookup_text_column: str = field(
        default=DEFAULT_LOOKUP_TEXT_COLUMN,
        metadata={"help": "Text column used to find unused T5 tokens."},
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
        tokenizer_name_or_path=model_args.model_name_or_path,
        lookup_source_csv=data_args.lookup_source_csv,
        lookup_text_column=data_args.lookup_text_column,
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
        tokenizer_name_or_path=model_args.model_name_or_path,
        lookup_source_csv=data_args.lookup_source_csv,
        lookup_text_column=data_args.lookup_text_column,
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

    attach_lookup_to_config(
        model.config,
        train_dataset.code_lookup,
        discrete_code_num=data_args.discrete_code_num,
        unit_token_lookup_file=data_args.token_file,
        lookup_source_csv=data_args.lookup_source_csv,
        lookup_text_column=data_args.lookup_text_column,
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
        save_token_lookup(
            Path(model_args.final_model_dir) / "unit_token_lookup.txt",
            train_dataset.code_lookup,
        )
    else:
        logger.info("Skipping final model save because save_final_model=False")
    if use_wandb and training_args.process_index == 0 and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
