#!/usr/bin/env python
"""
Train Flan-T5 for query generation on SLUE‑SQA5 using discrete codes.
"""
import os
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
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
    TrainerCallback,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
)
import evaluate
import wandb

# ---------------------------------------------
#  Logging setup
# ---------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def log_mem(stage: str):
    a = torch.cuda.memory_allocated()  / 1e9
    r = torch.cuda.memory_reserved()   / 1e9
    print(f"[{stage}] allocated={a:.2f} GB  reserved={r:.2f} GB")
    
    
class MemoryCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        log_mem("Training Begin")
        
    def on_step_end(self, args, state, control, **kwargs):
        log_mem(f"after step {state.global_step}")
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        log_mem(f"before epoch {state.epoch}")
        
    def on_evaluate(self, args, state, control, **kwargs):
        log_mem("Evaluate")
         

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
        lookup_file_name: Optional[
            str
        ] = "/home/ricky/dodofk/dataset/slue_sqa5/flan-t5-base-unused_tokens.txt",
        offset: int = 30,
        label_max_length: int = 300, # as some of the query is extreme long, we need to truncate the label
    ):
        assert split in [
            "train",
            "validation",
            "test",
            "verified_test",
        ], "split must be one of ['train','validation','test','verified_test']"

        self.split = split
        self.max_length = max_length
        self.code_path = code_path
        self.special_token = special_token
        self.offset = offset
        self.label_max_length = label_max_length
        # load mapping CSV
        csv_path = os.path.join(dataset_path, f"{split}.csv")
        self.df = pd.read_csv(csv_path)
        self.data = []

        self.discrete_code_num = discrete_code_num
        self._build_code_lookup(lookup_file_name)
        self._build_data()

        print("Info dataset length: ", len(self.data))

    def _build_code_lookup(self, lookup_file_name: Optional[str]):
        if lookup_file_name:
            lookup = np.loadtxt(lookup_file_name).astype(int)
            self.code_lookup = lookup
        else:
            raise ValueError("lookup_file_name is required")
        # invert lookup: original -> idx in [0,discrete_code_num)
        self.code_to_idx = {idx: orig for idx, orig in enumerate(self.code_lookup)}

    def _build_data(self):
        # truncate to 512 for each doc
        for _, row in self.df.iterrows():
            qid = str(row["question_id"])
            did = str(row["document_id"])

            q_code = np.loadtxt(
                os.path.join(self.code_path, f"{self.split}_code/{qid}.code")
            ).astype(int)
            q_code = np.vectorize(self.code_to_idx.get)(q_code)
            
            if len(q_code) > self.label_max_length:
                q_code = q_code[:self.label_max_length]
                
            q_seq = np.concatenate([q_code, [1]]) # as the length is not extreme strict, it could add 1 to the end
            

            d_code = np.loadtxt(
                os.path.join(self.code_path, f"document_code/{did}.code")
            ).astype(int)
            d_code = np.vectorize(self.code_to_idx.get)(d_code)

            cur_idx = 0
            while cur_idx < len(d_code):
                end_idx = min(cur_idx + self.max_length - 1, len(d_code))
                self.data.append(
                    {
                        "input_ids": np.concatenate([d_code[cur_idx:end_idx], [1]]),
                        "labels": q_seq,
                    }
                )
                # Ensure we don't go backwards or stay in the same place
                step = max(1, self.max_length - self.offset)
                cur_idx += step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_row = self.data[idx]
        return {
            "input_ids": torch.LongTensor(data_row["input_ids"]),
            "labels": torch.LongTensor(data_row["labels"]),
        }


class QGTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        return super().compute_loss(model, inputs, return_outputs)
    

    
    def prediction_step(
        self, model, inputs, prediction_loss_only=False, ignore_keys=None
    ):
        outputs = self.model.generate(
            input_ids=inputs["input_ids"].to(self.args.device),
            attention_mask=inputs["attention_mask"].to(self.args.device),
            max_length=200,  # as our label is  all smaller than 200
        )

        return (
            None,
            outputs,
            inputs["labels"],
        )


# ---------------------------------------------
#  Evaluation metric function
# ---------------------------------------------
class CustomEval:
    def __init__(self, model_args):
        self.model_args = model_args
        # self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")

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

        # ---- ROUGE-L (expects strings) ----
        pred_strs = [" ".join(x) for x in decoded_preds]
        label_strs = [" ".join(x) for x in decoded_labels]
        rouge = self.rouge_metric.compute(
            predictions=pred_strs,
            references=label_strs,
        )

        
        # save the pred and label strs with timestamp as json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"qg_output/pred_label_strs_{timestamp}.json", "w") as f:
            json.dump({"pred_strs": pred_strs, "label_strs": label_strs}, f)
        
        return {
            "rougeL": rouge["rougeL"],
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeLsum": rouge["rougeLsum"],
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
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model checkpoint to load"},
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
    if model_args.model_path:
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_path)
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path
        )

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
    trainer = QGTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # callbacks=[MemoryCallback()],
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
