from __future__ import annotations

import json

import numpy as np
import wandb
from omegaconf import DictConfig
from torch.utils.data import Subset
from transformers import AutoTokenizer, T5ForConditionalGeneration, set_seed

from speechgr.data import IndexingCollator, SlueSQA5DatasetV2
from speechgr import (
    DataConfig,
    ModelConfig,
    RankingConfig,
    WandbConfig,
    build_training_arguments,
    to_dataclass,
)
from speechgr.trainer import DSIRankingTrainer, NegLambdaScheduleCallback, RankingLossCallback
from speechgr.utils import RestrictDecodeVocab


def make_compute_metrics(tokenizer, valid_ids):
    def compute_metrics(eval_preds):
        hit_at_1 = 0
        hit_at_10 = 0
        hit_at_20 = 0
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank_list = tokenizer.batch_decode(beams, skip_special_tokens=True)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
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

        with open("eval_results.json", "w") as f:
            json.dump(metrics, f, indent=4)

        wandb.log(metrics)
        return metrics

    return compute_metrics


def _maybe_init_wandb(training_args, wandb_cfg: WandbConfig, run_notes: str) -> None:
    if training_args.local_rank not in (0, -1):
        return
    wandb.login()
    notes = run_notes or wandb_cfg.notes
    wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        name=training_args.run_name,
        notes=notes,
    )


def _build_dataset(split: str, data_cfg: DataConfig, model_name: str, max_length: int):
    return SlueSQA5DatasetV2(
        split=split,
        max_length=max_length,
        model_name_or_path=model_name,
        code_path=data_cfg.code_path,
        dataset_path=data_cfg.dataset_path,
        special_token=data_cfg.special_token,
        discrete_code_num=data_cfg.discrete_code_num,
        lookup_file_name=data_cfg.lookup_file_name,
    )


def run(cfg: DictConfig) -> None:
    set_seed(cfg.get("seed", 42))

    data_cfg = to_dataclass(cfg.data, DataConfig)
    model_cfg = to_dataclass(cfg.model, ModelConfig)
    ranking_cfg = to_dataclass(cfg.ranking, RankingConfig)
    wandb_cfg = to_dataclass(cfg.logging, WandbConfig)
    training_args = build_training_arguments(cfg.training.training_args)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name, cache_dir="cache")
    fast_tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name, cache_dir="cache"
    )

    source = model_cfg.model_path or model_cfg.model_name
    model = T5ForConditionalGeneration.from_pretrained(source, cache_dir="cache")

    train_dataset = _build_dataset("train", data_cfg, model_cfg.model_name, ranking_cfg.max_length)
    valid_dataset = _build_dataset(
        "validation", data_cfg, model_cfg.model_name, ranking_cfg.max_length
    )
    test_dataset = _build_dataset("test", data_cfg, model_cfg.model_name, ranking_cfg.max_length)

    eval_dataset = test_dataset if ranking_cfg.do_inference else valid_dataset

    restrict_decode_vocab = RestrictDecodeVocab(
        valid_ids=train_dataset.valid_ids, tokenizer=tokenizer
    )
    metrics_fn = make_compute_metrics(fast_tokenizer, train_dataset.valid_ids)

    if ranking_cfg.do_debug:
        train_dataset = Subset(train_dataset, range(100))
        eval_dataset = Subset(eval_dataset, range(100))

    trainer = DSIRankingTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=IndexingCollator(tokenizer, padding="longest"),
        compute_metrics=metrics_fn,
        id_max_length=ranking_cfg.id_max_length,
        restrict_decode_vocab=restrict_decode_vocab,
    )

    neg_lambda_callback = NegLambdaScheduleCallback(
        inbatch_schedule=[(0, 0.0), (5, 0.1)],
    )
    neg_lambda_callback.trainer = trainer

    trainer.add_callback(neg_lambda_callback)
    trainer.add_callback(RankingLossCallback())

    _maybe_init_wandb(training_args, wandb_cfg, ranking_cfg.run_notes)

    trainer.train()
