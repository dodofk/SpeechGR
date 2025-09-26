from __future__ import annotations

import json

import numpy as np
import wandb
from omegaconf import DictConfig
from transformers import AutoTokenizer, set_seed

from speechgr.data import (
    IndexingCollator,
    SlueSQA5DatasetV2,
    SlueSQA5WhisperDataset,
    WhisperIndexingCollator,
)
from speechgr.model import QFormerT5
from speechgr import (
    DataConfig,
    QFormerConfig,
    QFormerModelConfig,
    WandbConfig,
    build_training_arguments,
    to_dataclass,
)
from speechgr.trainer import DSITrainer
from speechgr.utils import RestrictDecodeVocab


def make_compute_metrics(tokenizer, valid_ids):
    def compute_metrics(eval_preds):
        hit1 = hit10 = hit20 = 0
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank = tokenizer.batch_decode(beams, skip_special_tokens=True)
            gold = tokenizer.decode(label, skip_special_tokens=True)
            rank = [d for i, d in enumerate(rank) if d not in rank[:i] and d in valid_ids]

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

        with open("eval_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        artifact = wandb.Artifact("eval_results", type="evaluation")
        artifact.add_file("eval_results.json")
        wandb.log_artifact(artifact)
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


def _load_model(model_cfg: QFormerModelConfig) -> QFormerT5:
    if model_cfg.model_path:
        return QFormerT5.from_pretrained(model_cfg.model_path)
    if model_cfg.model_type != "qformer":
        raise ValueError(f"Unsupported model_type: {model_cfg.model_type}")
    return QFormerT5(
        base_name=model_cfg.model_name,
        d_model_front=model_cfg.d_model_front,
        win_size_f=model_cfg.win_size_f,
        win_stride_f=model_cfg.win_stride_f,
        n_queries=model_cfg.n_queries,
        depth=model_cfg.qformer_depth,
        freeze_t5_encoder=model_cfg.freeze_t5_encoder,
        use_whisper_features=model_cfg.use_whisper_features,
    )


def _build_discrete_datasets(cfg: QFormerConfig, data_cfg: DataConfig, model_name: str, tokenizer):
    train_ds = SlueSQA5DatasetV2(
        split="train",
        max_length=cfg.max_length,
        dataset_path=data_cfg.dataset_path,
        code_path=data_cfg.code_path,
        encoder_name=data_cfg.encoder_name or "wavtokenizer",
        include_corpus=data_cfg.include_corpus,
        train_atomic=data_cfg.train_atomic,
        atomic_offset=data_cfg.atomic_offset,
        special_token=data_cfg.special_token,
        corpus_splits=data_cfg.include_corpus_splits,
        corpus_chunk_size=data_cfg.corpus_chunk_size,
        corpus_chunk_stride=data_cfg.corpus_chunk_stride,
        corpus_min_tokens=data_cfg.corpus_min_tokens,
    )
    valid_ds = SlueSQA5DatasetV2(
        split="validation",
        max_length=cfg.max_length,
        dataset_path=data_cfg.dataset_path,
        code_path=data_cfg.code_path,
        encoder_name=data_cfg.encoder_name or "wavtokenizer",
        include_corpus=False,
        train_atomic=data_cfg.train_atomic,
        atomic_offset=data_cfg.atomic_offset,
        special_token=data_cfg.special_token,
        corpus_splits=data_cfg.include_corpus_splits,
        corpus_chunk_size=data_cfg.corpus_chunk_size,
        corpus_chunk_stride=data_cfg.corpus_chunk_stride,
        corpus_min_tokens=data_cfg.corpus_min_tokens,
    )
    collator = IndexingCollator(tokenizer, padding="longest")
    return train_ds, valid_ds, collator


def _build_whisper_datasets(model_cfg: QFormerModelConfig, tokenizer):
    train_ds = SlueSQA5WhisperDataset(
        split="train",
        whisper_model_name=model_cfg.whisper_model_name,
        device=model_cfg.device,
        model_name_or_path=model_cfg.model_name,
        include_corpus=True,
        debug_max_samples=model_cfg.debug_max_samples,
        apply_spec_augment=model_cfg.apply_spec_augment,
        time_warp_param=model_cfg.time_warp_param,
        freq_mask_param=model_cfg.freq_mask_param,
        time_mask_param=model_cfg.time_mask_param,
    )
    valid_ds = SlueSQA5WhisperDataset(
        split="validation",
        whisper_model_name=model_cfg.whisper_model_name,
        device=model_cfg.device,
        model_name_or_path=model_cfg.model_name,
        include_corpus=False,
        debug_max_samples=model_cfg.debug_max_samples,
        apply_spec_augment=False,
    )
    collator = WhisperIndexingCollator(tokenizer=tokenizer)
    return train_ds, valid_ds, collator


def run(cfg: DictConfig) -> None:
    set_seed(cfg.get("seed", 42))

    data_cfg = to_dataclass(cfg.data, DataConfig)
    model_cfg = to_dataclass(cfg.model, QFormerModelConfig)
    qformer_cfg = to_dataclass(cfg.qformer, QFormerConfig)
    wandb_cfg = to_dataclass(cfg.logging, WandbConfig)
    training_args = build_training_arguments(cfg.training.training_args)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name, cache_dir="cache")

    if model_cfg.use_whisper_features:
        train_ds, valid_ds, collator = _build_whisper_datasets(model_cfg, tokenizer)
    else:
        train_ds, valid_ds, collator = _build_discrete_datasets(
            qformer_cfg, data_cfg, model_cfg.model_name, tokenizer
        )

    restrict_vocab = RestrictDecodeVocab(train_ds.valid_ids, tokenizer)
    metrics_fn = make_compute_metrics(tokenizer, train_ds.valid_ids)

    model = _load_model(model_cfg)

    trainer = DSITrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        compute_metrics=metrics_fn,
        id_max_length=qformer_cfg.id_max_length,
        restrict_decode_vocab=restrict_vocab,
        train_continuous_embedding=model_cfg.use_whisper_features,
        use_whisper_features=model_cfg.use_whisper_features,
    )

    trainer.max_length = qformer_cfg.max_length or 20
    trainer.num_return_sequences = qformer_cfg.num_return_sequences
    trainer.top_k = qformer_cfg.top_k

    _maybe_init_wandb(training_args, wandb_cfg, qformer_cfg.run_notes)

    trainer.train()
