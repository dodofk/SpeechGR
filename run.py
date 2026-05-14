from data import (
    IndexingCollator,
    SlueSQA5DatasetV2,
    IndexingCollatorWithAtomic,
)

from utils import RestrictDecodeVocab

from transformers import (
    T5Tokenizer,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    BartTokenizer,
    BartTokenizerFast,
    BartForConditionalGeneration,
    TrainingArguments,
    TrainerCallback,
    MT5Tokenizer,
    MT5TokenizerFast,
    MT5ForConditionalGeneration,
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
)

from trainer import DSITrainer, DocTqueryTrainer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from dataclasses import asdict, dataclass, field
from typing import Optional, List
import json
import os
from tqdm import tqdm
from model import ContinousEmbT5
from hub_checkpoint import HubCheckpointArguments, build_hub_checkpoint_callbacks
from unit_store import DEFAULT_SLUE_UNIT_HF_PATH
from unit_token_lookup import lookup_from_model_config

set_seed(42)


def _uses_wandb(report_to) -> bool:
    if report_to is None:
        return False
    if isinstance(report_to, str):
        values = [report_to]
    else:
        values = list(report_to)
    values = {str(value).lower() for value in values}
    return "wandb" in values or "all" in values


@dataclass
class RunArguments:
    model_name: str = field(default=None)
    model_path: Optional[str] = field(default=None)
    max_length: Optional[int] = field(default=512)
    id_max_length: Optional[int] = field(default=128)
    top_k: Optional[int] = field(default=10)
    num_return_sequences: Optional[int] = field(default=10)
    run_notes: str = field(default="")
    code_path: str = field(default=DEFAULT_SLUE_UNIT_HF_PATH)
    dataset_path: str = field(default_factory=lambda: os.environ.get("DATASET_PATH", ""))
    special_token: Optional[int] = field(default=32000)
    discrete_code_num: Optional[int] = field(default=500)
    lookup_file_name: Optional[str] = field(default=None)
    train_continuous_embedding: Optional[bool] = field(default=False)
    downsample_factor: int = field(default=2)
    ssl_feat_dim: int = field(default=1024)  # should manually maintain both dim
    hidden_dim: int = field(default=768)
    run_atomic: bool = field(default=False)
    atomic_offset: int = field(default=50)


def make_compute_metrics(tokenizer, valid_ids):
    def compute_metrics(eval_preds):
        hit_at_1 = 0
        hit_at_10 = 0
        hit_at_20 = 0
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank_list = tokenizer.batch_decode(beams, skip_special_tokens=True)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
            # filter out duplicates and invalid docids
            filtered_rank_list = []
            for docid in rank_list:
                if docid not in filtered_rank_list and docid in valid_ids:
                    filtered_rank_list.append(docid)
            hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
            hits_at_20 = np.where(np.array(filtered_rank_list)[:20] == label_id)[0]
            if len(hits) != 0:
                hit_at_10 += 1
                if hits[0] == 0:
                    hit_at_1 += 1
            if len(hits_at_20) != 0:
                hit_at_20 += 1

        metrics = {
            "Hits@1": hit_at_1 / len(eval_preds.predictions),
            "Hits@10": hit_at_10 / len(eval_preds.predictions),
            "Hits@20": hit_at_20 / len(eval_preds.predictions),
        }

        # Save the metrics to a JSON file
        with open("eval_results.json", "w") as f:
            json.dump(metrics, f, indent=4)

        if wandb.run is not None:
            artifact = wandb.Artifact(
                "eval_results", type="evaluation", description="Evaluation results"
            )
            artifact.add_file("eval_results.json")
            wandb.log_artifact(artifact)

        # Also save the raw predictions and label_ids.
        raw_data = {
            "predictions": (
                eval_preds.predictions.tolist()
                if isinstance(eval_preds.predictions, np.ndarray)
                else eval_preds.predictions
            ),
            "label_ids": (
                eval_preds.label_ids.tolist()
                if isinstance(eval_preds.label_ids, np.ndarray)
                else eval_preds.label_ids
            ),
        }
        with open("eval_raw.json", "w") as f:
            json.dump(raw_data, f, indent=4)

        if wandb.run is not None:
            raw_artifact = wandb.Artifact(
                "eval_raw", type="raw_data", description="Raw predictions and label ids"
            )
            raw_artifact.add_file("eval_raw.json")
            wandb.log_artifact(raw_artifact)
            wandb.log(metrics)

        return metrics

    return compute_metrics


def main():

    parser = HfArgumentParser((TrainingArguments, RunArguments, HubCheckpointArguments))
    training_args, run_args, hub_args = parser.parse_args_into_dataclasses()

    use_wandb = _uses_wandb(training_args.report_to)
    if use_wandb and training_args.process_index == 0:
        wandb.login()
        wandb.init(
            project="DSI",
            name=training_args.run_name,
            notes=run_args.run_notes,
            config={
                "training": training_args.to_dict(),
                "run": asdict(run_args),
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(run_args.model_name, cache_dir="cache")
    fast_tokenizer = AutoTokenizer.from_pretrained(
        run_args.model_name, cache_dir="cache"
    )

    if run_args.train_continuous_embedding:
        if run_args.model_path:
            model = ContinousEmbT5.from_pretrained(
                run_args.model_path,
                cache_dir="cache",
                ssl_feat_dim=run_args.ssl_feat_dim,
                downsample_factor=run_args.downsample_factor,
            )

    elif "mt5" in run_args.model_name:
        if run_args.model_path:
            model = MT5ForConditionalGeneration.from_pretrained(
                run_args.model_path, cache_dir="cache"
            )
        else:
            model = MT5ForConditionalGeneration.from_pretrained(
                run_args.model_name, cache_dir="cache"
            )

        # make model parameters contigious before training
        for param in model.parameters():
            param.data = param.data.contiguous()

    elif "bart" in run_args.model_name:
        if run_args.model_path:
            model = BartForConditionalGeneration.from_pretrained(
                run_args.model_path, cache_dir="cache"
            )
        else:
            model = BartForConditionalGeneration.from_pretrained(
                run_args.model_name, cache_dir="cache"
            )

    else:
        if run_args.model_path:
            model = T5ForConditionalGeneration.from_pretrained(
                run_args.model_path, cache_dir="cache"
            )
            print(f"Load with model path: {run_args.model_path}")
        else:
            model = T5ForConditionalGeneration.from_pretrained(
                run_args.model_name, cache_dir="cache"
            )

    config_lookup = lookup_from_model_config(
        model.config,
        discrete_code_num=run_args.discrete_code_num,
    )

    train_dataset = SlueSQA5DatasetV2(
        split="train",
        max_length=run_args.max_length,
        model_name_or_path=run_args.model_name,
        code_path=run_args.code_path,
        dataset_path=run_args.dataset_path,
        special_token=run_args.special_token,
        discrete_code_num=run_args.discrete_code_num,
        lookup_file_name=run_args.lookup_file_name,
        lookup_values=config_lookup,
        train_atomic=run_args.run_atomic,
        atomic_offset=run_args.atomic_offset,
    )

    valid_dataset = SlueSQA5DatasetV2(
        split="validation",
        max_length=run_args.max_length,
        model_name_or_path=run_args.model_name,
        code_path=run_args.code_path,
        dataset_path=run_args.dataset_path,
        special_token=run_args.special_token,
        discrete_code_num=run_args.discrete_code_num,
        lookup_file_name=run_args.lookup_file_name,
        lookup_values=config_lookup,
        train_atomic=run_args.run_atomic,
        atomic_offset=run_args.atomic_offset,
    )

    test_dataset = SlueSQA5DatasetV2(
        split="test",
        max_length=run_args.max_length,
        model_name_or_path=run_args.model_name,
        code_path=run_args.code_path,
        dataset_path=run_args.dataset_path,
        special_token=run_args.special_token,
        discrete_code_num=run_args.discrete_code_num,
        lookup_file_name=run_args.lookup_file_name,
        lookup_values=config_lookup,
        train_atomic=run_args.run_atomic,
        atomic_offset=run_args.atomic_offset,
    )

    restrict_decode_vocab = RestrictDecodeVocab(
        valid_ids=train_dataset.valid_ids, tokenizer=tokenizer
    )

    if run_args.run_atomic:
        collator = IndexingCollatorWithAtomic(tokenizer=tokenizer, padding="longest")
    else:
        collator = IndexingCollator(tokenizer=tokenizer, padding="longest")

    trainer = DSITrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # eval_dataset=test_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(fast_tokenizer, train_dataset.valid_ids),
        id_max_length=run_args.id_max_length,
        restrict_decode_vocab=restrict_decode_vocab,
        train_continuous_embedding=run_args.train_continuous_embedding,
        callbacks=build_hub_checkpoint_callbacks(hub_args, tokenizer=tokenizer),
    )

    trainer.train()


if __name__ == "__main__":
    main()
