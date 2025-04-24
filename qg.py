#!/usr/bin/env python
"""
Train Flan-T5 for query generation on SLUE‑SQA5 using discrete codes.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
)
from datasets import load_metric
import wandb

# ---------------------------------------------
#  Logging setup
# ---------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------
#  Dataset for Query Generation
# ---------------------------------------------
class QueryGenDataset(Dataset):
    """
    Input: document discrete code sequence
    Output: query discrete code sequence

    Expects:
      dataset CSV with columns ['question_id','document_id'] under dataset_path/<split>.csv
      precomputed codes under code_path/{split}_code/<question_id>.code for queries
      precomputed codes under code_path/document_code/<document_id>.code for docs
    """

    def __init__(
        self,
        split: str,
        max_length: int = 512,
        dataset_path: str = "/home/ricky/dodofk/dataset/slue_sqa5/",
        code_path: str = "/home/ricky/dodofk/dataset/slue_sqa_code_c512",
        discrete_code_num: int = 512,
        special_token: int = 32000,
        lookup_file_name: Optional[str] = "/home/ricky/dodofk/dataset/slue_sqa5/flan-t5-base-unused_tokens.txt",
    ):
        assert split in ["train", "validation", "test", "verified_test"], \
            "split must be one of ['train','validation','test','verified_test']"

        self.split = split
        self.max_length = max_length
        self.code_path = code_path
        self.special_token = special_token

        # load mapping CSV
        csv_path = os.path.join(dataset_path, f"{split}.csv")
        self.data = pd.read_csv(csv_path)

        self.discrete_code_num = discrete_code_num
        self._build_code_lookup(lookup_file_name)

    def _build_code_lookup(self, lookup_file_name: Optional[str]):
        if lookup_file_name:
            lookup = np.loadtxt(lookup_file_name).astype(int)
            self.code_lookup = lookup
        else:
            raise ValueError("lookup_file_name is required")
        # invert lookup: original -> idx in [0,discrete_code_num)
        self.code_to_idx = {idx: orig for idx, orig in enumerate(self.code_lookup)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        qid = str(row['question_id'])
        did = str(row['document_id'])

        # load question (query) code
        q_code = np.loadtxt(
            os.path.join(self.code_path, f"{self.split}_code/{qid}.code")
        ).astype(int)
        q_code = np.vectorize(self.code_to_idx.get)(q_code)
        # wrap with special_token and eos (id=1)
        q_seq = np.concatenate([[self.special_token], q_code, [1]])
        if len(q_seq) > self.max_length:
            q_seq = np.concatenate([q_seq[: self.max_length - 1], [1]])

        # load document code
        d_code = np.loadtxt(
            os.path.join(self.code_path, f"document_code/{did}.code")
        ).astype(int)
        d_code = np.vectorize(self.code_to_idx.get)(d_code)
        d_seq = np.concatenate([[self.special_token], d_code, [1]])
        if len(d_seq) > self.max_length:
            d_seq = np.concatenate([d_seq[: self.max_length - 1], [1]])

        return {
            "input_ids": torch.LongTensor(d_seq),
            "labels": torch.LongTensor(q_seq),
        }


# ---------------------------------------------
#  Evaluation metric function
# ---------------------------------------------
class CustomEval:
    def __init__(self, model_args):
        self.model_args = model_args
        self.bleu_metric = load_metric("bleu")
        self.rouge_metric = load_metric("rouge")

    def __call__(self, eval_preds):
        return self.compute_metrics(eval_preds)

    def compute_metrics(self, eval_preds):
        """
        eval_preds: a tuple (predictions, label_ids)
      - predictions: np.ndarray of shape (batch, seq_len) from model.generate(...)
      - label_ids:    np.ndarray of shape (batch, seq_len) where pads are -100
    """
        preds, labels = eval_preds
        # if your model returns (preds, scores), unpack
        if isinstance(preds, tuple):
            preds = preds[0]

        batch_size = preds.shape[0]
        decoded_preds = []
        decoded_labels = []
        for i in range(batch_size):
            # filter out special / EOS tokens from preds
            pred_tokens = [
                str(tok)
                for tok in preds[i]
                if tok not in {self.model_args.special_token, 1}
            ]

            # likewise for labels, also drop the -100 pads
            label_seq = [tok for tok in labels[i] if tok != -100]
            label_tokens = [
                str(tok)
                for tok in label_seq
                if tok not in {self.model_args.special_token, 1}
            ]

            decoded_preds.append(pred_tokens)
            decoded_labels.append(label_tokens)
        # ---- BLEU (expects tokenized lists) ----
        # references must be List[List[List[str]]], so we wrap each label in an extra list
        bleu = self.bleu_metric.compute(
            predictions=decoded_preds,
            references=[[ref] for ref in decoded_labels],
        )
        # ---- ROUGE-L (expects strings) ----
        pred_strs  = [" ".join(x) for x in decoded_preds]
        label_strs = [" ".join(x) for x in decoded_labels]
        rouge = self.rouge_metric.compute(
            predictions=pred_strs,
            references=label_strs,
            rouge_types=["rougeL"],
        )["rougeL"].mid

        return {
            "bleu": bleu["bleu"],
            "rougeL_precision": rouge.precision,
            "rougeL_recall":    rouge.recall,
            "rougeL_f1":        rouge.fmeasure,
        }


#  Argument Definitions
# ---------------------------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="google/flan-t5-base",
        metadata={"help": "Pretrained model identifier or path"},
    )
    final_model_dir: str = field(
        default="flan_t5_base_QG",
        metadata={"help": "Directory to save the final model checkpoint"},
    )
    special_token: int = field(
        default=32000,
        metadata={"help": "ID for the special query/document token"},
    )


@dataclass
class DataTrainingArguments:
    dataset_path: str = field(
        default="/home/ricky/dodofk/dataset/slue_sqa5/",
        metadata={"help": "Base path to SLUE-SQA5 CSV files"},
    )
    code_path: str = field(
        default="/home/ricky/dodofk/dataset/slue_sqa_code_c512",
        metadata={"help": "Path to precomputed .code files"},
    )
    split: str = field(
        default="train",
        metadata={"help": "Which split to use: train/validation/test/verified_test"},
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Max sequence length for both src and tgt"},
    )
    discrete_code_num: int = field(
        default=512,
        metadata={"help": "Size of discrete code lookup"},
    )
    lookup_file_name: Optional[str] = field(
        default="/home/ricky/dodofk/dataset/slue_sqa5/flan-t5-base-unused_tokens.txt",
        metadata={"help": "Optional .npy lookup file"},
    )


@dataclass
class WandBArguments:
    project: str = field(
        default="Audio-QG",
        metadata={"help": "WandB project name for logging"},
    )
    description: Optional[str] = field(
        default=None,
        metadata={"help": "Notes/description for WandB run"},
    )


# ---------------------------------------------
#  Main Training
# ---------------------------------------------

def main() -> None:
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, WandBArguments, TrainingArguments)
    )
    model_args, data_args, wandb_args, training_args = (
        parser.parse_args_into_dataclasses()
    )

    # ensure wandb logging
    if not training_args.report_to:
        training_args.report_to = ["wandb"]

    wandb.init(
        project=wandb_args.project,
        config=training_args.to_dict(),
        notes=wandb_args.description,
    )

    logger.info("Training/evaluation parameters: %s", training_args)

    # 1. Load model + tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(model_args.model_name_or_path)

    # 2. Prepare datasets
    train_ds = QueryGenDataset(
        split=data_args.split,
        max_length=data_args.max_length,
        dataset_path=data_args.dataset_path,
        code_path=data_args.code_path,
        discrete_code_num=data_args.discrete_code_num,
        special_token=model_args.special_token,
        lookup_file_name=data_args.lookup_file_name,
    )
    # for eval, use validation split
    eval_ds = QueryGenDataset(
        split="validation",
        max_length=data_args.max_length,
        dataset_path=data_args.dataset_path,
        code_path=data_args.code_path,
        discrete_code_num=data_args.discrete_code_num,
        special_token=model_args.special_token,
        lookup_file_name=data_args.lookup_file_name,
    )

    # 3. Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    compute_metrics = CustomEval(model_args)

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 5. Train
    logger.info("Starting training for query generation...")
    trainer.train()

    # 6. Save
    logger.info("Saving final model to %s", model_args.final_model_dir)
    trainer.save_model(model_args.final_model_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
