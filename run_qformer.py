# run_qformer.py
# --------------------------------------------------------------------------
# Train a Q-Former-T5 model on SLUE-SQA-5 discrete-unit data.
#
# Requires:
#   • qformer_t5.py (the architecture sent earlier)
#   • data.py  – SlueSQA5DatasetV2, IndexingCollator
#   • trainer.py – DSITrainer
# --------------------------------------------------------------------------
import json
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)

from data import SlueSQA5DatasetV2, IndexingCollator, SlueSQA5WhisperDataset, WhisperIndexingCollator
from trainer import DSITrainer
from utils import RestrictDecodeVocab
from model import QFormerT5  # <-- our new model


# reproducibility
set_seed(42)


# ────────────────────────────────────────────────────────────────────
#  CLI arguments
# ────────────────────────────────────────────────────────────────────
@dataclass
class RunArguments:
    # ----- model / Q-Former hyper-params ---------------------------
    model_name: str = field(default="google/flan-t5-base")
    model_path: Optional[str] = field(default=None)  # resume from ckpt
    model_type: str = field(default="qformer")  # Model type: 'qformer' or other future models
    use_whisper_features: bool = field(default=False)  # Use Whisper features instead of discrete units
    d_model_front: int = field(default=768)  # discrete-unit emb dim or Whisper feature dim
    n_queries: int = field(default=1)
    qformer_depth: int = field(default=2)
    win_size_f: int = field(default=17)
    win_stride_f: int = field(default=17)
    freeze_t5_encoder: bool = field(default=True)

    # ----- Whisper-specific params --------------------------------
    whisper_model_name: str = field(default="openai/whisper-base")  # Whisper model for feature extraction
    device: str = field(default="cuda")  # Device for Whisper model
    apply_spec_augment: bool = field(default=False)  # Apply SpecAugment to Whisper features
    time_warp_param: int = field(default=80)  # Time warp parameter for SpecAugment
    freq_mask_param: int = field(default=27)  # Frequency mask parameter for SpecAugment
    time_mask_param: int = field(default=100)  # Time mask parameter for SpecAugment

    # ----- data ----------------------------------------------------
    max_length: Optional[int] = field(default=None)  # Maximum sequence length. None for Whisper setup
    id_max_length: int = field(default=128)
    discrete_code_num: int = field(default=500)
    dataset_path: str = field(default="/home/ricky/dodofk/dataset/slue_sqa5/")
    code_path: str = field(default="/home/ricky/dodofk/dataset/slue_sqa_code_c512")
    special_token: int = field(default=32000)
    lookup_file_name: Optional[str] = field(default=None)
    debug_max_samples: Optional[int] = field(default=None)  # Limit dataset size for debugging

    # ----- misc ----------------------------------------------------
    run_notes: str = field(default="")
    top_k: int = field(default=10)
    num_return_sequences: int = field(default=10)


# ────────────────────────────────────────────────────────────────────
#  Metric helper
# ────────────────────────────────────────────────────────────────────
def make_compute_metrics(tokenizer, valid_ids: List[str]):
    def compute_metrics(eval_preds):
        hit1 = hit10 = hit20 = 0
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank = tokenizer.batch_decode(beams, skip_special_tokens=True)
            gold = tokenizer.decode(label, skip_special_tokens=True)

            # keep distinct, valid IDs
            rank = [
                d for i, d in enumerate(rank) if d not in rank[:i] and d in valid_ids
            ]

            if gold in rank[:1]:
                hit1 += 1
            if gold in rank[:10]:
                hit10 += 1
            if gold in rank[:20]:
                hit20 += 1

        total = len(eval_preds.predictions)
        metrics = {
            "Hits@1": hit1 / total,
            "Hits@10": hit10 / total,
            "Hits@20": hit20 / total,
        }

        # log & save
        with open("eval_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        artifact = wandb.Artifact("eval_results", type="evaluation")
        artifact.add_file("eval_results.json")
        wandb.log_artifact(artifact)
        wandb.log(metrics) 
        return metrics

    return compute_metrics


# ────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────
def main():
    parser = HfArgumentParser((TrainingArguments, RunArguments))
    training_args, run_args = parser.parse_args_into_dataclasses()

    # 1) WandB (main process only)
    if training_args.local_rank in (0, -1):
        wandb.login()
        wandb.init(project="DSI", name=training_args.run_name, notes=run_args.run_notes)

    # 2) Tokeniser
    tokenizer = AutoTokenizer.from_pretrained(run_args.model_name, cache_dir="cache")

    # 3) Model
    if run_args.model_path:
        model = QFormerT5.from_pretrained(run_args.model_path)  # resume
    else:
        if run_args.model_type == "qformer":
            model = QFormerT5(
                base_name=run_args.model_name,
                d_model_front=run_args.d_model_front,
                win_size_f=run_args.win_size_f,
                win_stride_f=run_args.win_stride_f,
                n_queries=run_args.n_queries,
                depth=run_args.qformer_depth,
                freeze_t5_encoder=run_args.freeze_t5_encoder,
                use_whisper_features=run_args.use_whisper_features,
            )
        else:
            raise ValueError(f"Unknown model_type: {run_args.model_type}")

    # 4) Datasets
    if run_args.use_whisper_features:
        # Use Whisper dataset
        train_ds = SlueSQA5WhisperDataset(
            split="train",
            whisper_model_name=run_args.whisper_model_name,
            device=run_args.device,
            model_name_or_path=run_args.model_name,
            include_corpus=True,
            debug_max_samples=run_args.debug_max_samples,
            apply_spec_augment=run_args.apply_spec_augment,
            time_warp_param=run_args.time_warp_param,
            freq_mask_param=run_args.freq_mask_param,
            time_mask_param=run_args.time_mask_param,
        )
        valid_ds = SlueSQA5WhisperDataset(
            split="validation",
            whisper_model_name=run_args.whisper_model_name,
            device=run_args.device,
            model_name_or_path=run_args.model_name,
            include_corpus=False,
            debug_max_samples=run_args.debug_max_samples,
            apply_spec_augment=False,  # No augmentation for validation
        )
        
        # Use Whisper collator
        data_collator = WhisperIndexingCollator(tokenizer=tokenizer)
        
    else:
        # Use discrete unit dataset
        train_ds = SlueSQA5DatasetV2(
            split="train",
            max_length=run_args.max_length,
            model_name_or_path=run_args.model_name,
            code_path=run_args.code_path,
            dataset_path=run_args.dataset_path,
            special_token=run_args.special_token,
            discrete_code_num=run_args.discrete_code_num,
            lookup_file_name=run_args.lookup_file_name,
        )
        valid_ds = SlueSQA5DatasetV2(
            split="validation",
            max_length=run_args.max_length,
            model_name_or_path=run_args.model_name,
            code_path=run_args.code_path,
            dataset_path=run_args.dataset_path,
            special_token=run_args.special_token,
            discrete_code_num=run_args.discrete_code_num,
            lookup_file_name=run_args.lookup_file_name,
        )
        
        # Use discrete unit collator
        data_collator = IndexingCollator(tokenizer, padding="longest")

    # 5) Restrict vocab & metrics
    restrict_vocab = RestrictDecodeVocab(train_ds.valid_ids, tokenizer)
    metrics_fn = make_compute_metrics(tokenizer, train_ds.valid_ids)

    # 6) Trainer
    trainer = DSITrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        compute_metrics=metrics_fn,
        id_max_length=run_args.id_max_length,
        restrict_decode_vocab=restrict_vocab,
        train_continuous_embedding=run_args.use_whisper_features,
        use_whisper_features=run_args.use_whisper_features,
    )

    trainer.train()


if __name__ == "__main__":
    main()
