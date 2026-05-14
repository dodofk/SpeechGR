#!/usr/bin/env python
"""
Train Flan-T5 for query generation on SLUE‑SQA5 using discrete codes.
"""
import os
import logging
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional, Sequence, List, Dict, Any

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
from hub_checkpoint import HubCheckpointArguments, build_hub_checkpoint_callbacks
from unit_store import (
    DEFAULT_SLUE_UNIT_HF_PATH,
    PACKED_SLUE_PATTERNS,
    load_packed_store,
    resolve_unit_code_path,
)
from unit_token_lookup import (
    build_lookup_from_csv,
    load_token_lookup,
    lookup_from_model_config,
)

# ---------------------------------------------
#  Logging setup
# ---------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _uses_wandb(report_to) -> bool:
    if report_to is None:
        return False
    if isinstance(report_to, str):
        values = [report_to]
    else:
        values = list(report_to)
    values = {str(value).lower() for value in values}
    return "wandb" in values or "all" in values


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
        dataset_path: str = "",
        code_path: str = DEFAULT_SLUE_UNIT_HF_PATH,
        discrete_code_num: int = 512,
        special_token: int = 32000,
        lookup_file_name: Optional[str] = None,
        lookup_values: Optional[Sequence[int]] = None,
        model_name_or_path: str = "google/flan-t5-base",
        pq_filename: str = "slue_sqa5_pq10_llama32_3b_clean.csv",
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
        self.code_path = resolve_unit_code_path(
            code_path,
            allow_patterns=PACKED_SLUE_PATTERNS,
        )
        self.special_token = special_token
        self.offset = offset
        self.label_max_length = label_max_length
        self.model_name_or_path = model_name_or_path
        self.pq_filename = pq_filename
        self.query_store = load_packed_store(
            os.path.join(self.code_path, f"{self.split}.npz")
        )
        self.document_store = load_packed_store(
            os.path.join(self.code_path, "documents.npz")
        )
        # load mapping CSV
        csv_path = os.path.join(dataset_path, f"{split}.csv")
        self.df = pd.read_csv(csv_path)
        self.data = []

        self.discrete_code_num = discrete_code_num
        self._build_code_lookup(lookup_file_name, lookup_values, dataset_path)
        self._build_data()

        print("Info dataset length: ", len(self.data))

    def _load_query_code(self, qid: str) -> np.ndarray:
        if self.query_store is not None and qid in self.query_store:
            return self.query_store.get_code(qid)
        return np.loadtxt(
            os.path.join(self.code_path, f"{self.split}_code/{qid}.code")
        ).astype(int)

    def _load_document_code(self, did: str) -> np.ndarray:
        if self.document_store is not None and did in self.document_store:
            return self.document_store.get_code(did)
        return np.loadtxt(
            os.path.join(self.code_path, f"document_code/{did}.code")
        ).astype(int)

    def _build_code_lookup(
        self,
        lookup_file_name: Optional[str],
        lookup_values: Optional[Sequence[int]],
        dataset_path: str,
    ):
        if lookup_file_name:
            lookup = load_token_lookup(
                lookup_file_name,
                discrete_code_num=self.discrete_code_num,
            )
        elif lookup_values is not None:
            lookup = np.asarray(lookup_values, dtype=int)[: self.discrete_code_num]
            if len(lookup) < self.discrete_code_num:
                raise ValueError(
                    f"lookup_values only provides {len(lookup)} token ids, "
                    f"but discrete_code_num={self.discrete_code_num}"
                )
        else:
            lookup = build_lookup_from_csv(
                csv_path=os.path.join(dataset_path, self.pq_filename),
                tokenizer_name_or_path=self.model_name_or_path,
                discrete_code_num=self.discrete_code_num,
            )
            self.code_lookup = lookup
        self.code_lookup = lookup
        # invert lookup: original -> idx in [0,discrete_code_num)
        self.code_to_idx = {idx: orig for idx, orig in enumerate(self.code_lookup)}

    def _build_data(self):
        # truncate to 512 for each doc
        for _, row in self.df.iterrows():
            qid = str(row["question_id"])
            did = str(row["document_id"])

            q_code = self._load_query_code(qid)
            q_code = np.vectorize(self.code_to_idx.get)(q_code)
            
            if len(q_code) > self.label_max_length:
                q_code = q_code[:self.label_max_length]
                
            q_seq = np.concatenate([q_code, [1]]) # as the length is not extreme strict, it could add 1 to the end
            

            d_code = self._load_document_code(did)
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
    def __init__(self, *args, generation_max_length: int = 302, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_max_length = generation_max_length

    def compute_loss(self, model, inputs, return_outputs=False):
        return super().compute_loss(model, inputs, return_outputs)
    

    
    def prediction_step(
        self, model, inputs, prediction_loss_only=False, ignore_keys=None
    ):
        outputs = self.model.generate(
            input_ids=inputs["input_ids"].to(self.args.device),
            attention_mask=inputs["attention_mask"].to(self.args.device),
            max_length=self.generation_max_length,
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
    def __init__(self, model_args, pad_token_id: int):
        self.model_args = model_args
        self.ignored_token_ids = {self.model_args.special_token, 1, pad_token_id}
        # self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")

    def __call__(self, eval_preds):
        return self.compute_metrics(eval_preds)

    @staticmethod
    def _unit_overlap(pred_tokens: List[str], label_tokens: List[str]) -> Dict[str, float]:
        if not pred_tokens and not label_tokens:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not pred_tokens or not label_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        pred_counts = Counter(pred_tokens)
        label_counts = Counter(label_tokens)
        overlap = sum((pred_counts & label_counts).values())
        precision = overlap / len(pred_tokens)
        recall = overlap / len(label_tokens)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

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
                if tok not in self.ignored_token_ids
            ]

            # likewise for labels, also drop the -100 pads
            label_seq = [tok for tok in labels[i] if tok != -100]
            label_tokens = [
                str(tok)
                for tok in label_seq
                if tok not in self.ignored_token_ids
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
        os.makedirs("qg_output", exist_ok=True)
        with open(f"qg_output/pred_label_strs_{timestamp}.json", "w") as f:
            json.dump({"pred_strs": pred_strs, "label_strs": label_strs}, f)

        overlaps = [
            self._unit_overlap(pred_tokens, label_tokens)
            for pred_tokens, label_tokens in zip(decoded_preds, decoded_labels)
        ]
        pred_lengths = [len(tokens) for tokens in decoded_preds]
        label_lengths = [len(tokens) for tokens in decoded_labels]
        
        return {
            "rougeL": rouge["rougeL"],
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeLsum": rouge["rougeLsum"],
            "unit_precision": float(np.mean([x["precision"] for x in overlaps])),
            "unit_recall": float(np.mean([x["recall"] for x in overlaps])),
            "unit_f1": float(np.mean([x["f1"] for x in overlaps])),
            "pred_len": float(np.mean(pred_lengths)),
            "label_len": float(np.mean(label_lengths)),
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
    save_final_model: bool = field(
        default=True,
        metadata={"help": "Save final model after training."},
    )


@dataclass
class DataTrainingArguments:
    dataset_path: str = field(
        default_factory=lambda: os.environ.get("DATASET_PATH", ""),
        metadata={"help": "Base path to SLUE-SQA5 CSV files"},
    )
    code_path: str = field(
        default=DEFAULT_SLUE_UNIT_HF_PATH,
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
    label_max_length: int = field(
        default=300,
        metadata={"help": "Max query unit length before appending EOS"},
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Max generated sequence length. Defaults to label_max_length + 2."
        },
    )
    discrete_code_num: int = field(
        default=500,
        metadata={"help": "Size of discrete code lookup"},
    )
    lookup_file_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional lookup txt mapping unit ids to T5 token ids. "
            "If omitted, uses model config lookup or rebuilds from SLUE pseudo-query text."
        },
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
        wandb.init(
            project=wandb_args.project,
            name=training_args.run_name,
            config={
                "training": training_args.to_dict(),
                "model": asdict(model_args),
                "data": asdict(data_args),
            },
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
    config_lookup = lookup_from_model_config(
        model.config,
        discrete_code_num=data_args.discrete_code_num,
    )
    generation_max_length = (
        data_args.generation_max_length
        if data_args.generation_max_length is not None
        else data_args.label_max_length + 2
    )

    # 2. Prepare datasets
    train_ds = QueryGenDataset(
        split=data_args.split,
        max_length=data_args.max_length,
        dataset_path=data_args.dataset_path,
        code_path=data_args.code_path,
        discrete_code_num=data_args.discrete_code_num,
        special_token=model_args.special_token,
        lookup_file_name=data_args.lookup_file_name,
        lookup_values=config_lookup,
        model_name_or_path=model_args.model_name_or_path,
        label_max_length=data_args.label_max_length,
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
        lookup_values=config_lookup,
        model_name_or_path=model_args.model_name_or_path,
        label_max_length=data_args.label_max_length,
    )
    
    # 3. Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    compute_metrics = CustomEval(model_args, pad_token_id=tokenizer.pad_token_id)

    # 4. Initialize Trainer
    trainer = QGTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        generation_max_length=generation_max_length,
        callbacks=build_hub_checkpoint_callbacks(hub_args, tokenizer=tokenizer),
    )

    # 5. Train
    logger.info("Starting training for query generation...")
    trainer.train()

    # 6. Save
    if model_args.save_final_model:
        logger.info("Saving final model to %s", model_args.final_model_dir)
        trainer.save_model(model_args.final_model_dir)
    else:
        logger.info("Skipping final model save because save_final_model=False")
    if use_wandb and training_args.process_index == 0 and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
