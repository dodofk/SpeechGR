from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    MT5ForConditionalGeneration,
    T5ForConditionalGeneration,
    set_seed,
)

from speechgr.data import (
    IndexingCollator,
    IndexingCollatorWithAtomic,
    SlueSQA5DatasetV2,
    SlueSQA5TextDataset,
    TextIndexingCollator,
    DiscreteUnitDataset,
)
from speechgr.encoders import TextEncoder
from speechgr.model import ContinousEmbT5
from speechgr import (
    DataConfig,
    ModelConfig,
    RunConfig,
    WandbConfig,
    build_training_arguments,
    to_dataclass,
)
from speechgr.trainer import DSITrainer
from speechgr.utils_legacy import RestrictDecodeVocab


logger = logging.getLogger(__name__)


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

        artifact = wandb.Artifact(
            "eval_results", type="evaluation", description="Evaluation results"
        )
        artifact.add_file("eval_results.json")
        wandb.log_artifact(artifact)

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

        raw_artifact = wandb.Artifact(
            "eval_raw", type="raw_data", description="Raw predictions and label ids"
        )
        raw_artifact.add_file("eval_raw.json")
        wandb.log_artifact(raw_artifact)
        wandb.log(metrics)

        return metrics

    return compute_metrics


def _maybe_init_wandb(
    training_args,
    wandb_cfg: WandbConfig,
    run_notes: str,
    full_cfg: Optional[DictConfig] = None,
) -> None:
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
    if wandb_cfg.log_hydra and full_cfg is not None:
        try:
            wandb.config.update(
                OmegaConf.to_container(full_cfg, resolve=True), allow_val_change=True
            )
        except Exception as exc:  # pragma: no cover - wandb best effort
            logger.warning("Failed to log Hydra config to WandB: %s", exc)


def _load_model(model_cfg: ModelConfig):
    if model_cfg.train_continuous_embedding and model_cfg.model_path:
        return ContinousEmbT5.from_pretrained(
            model_cfg.model_path,
            cache_dir="cache",
            ssl_feat_dim=model_cfg.ssl_feat_dim,
            downsample_factor=model_cfg.downsample_factor,
        )

    if "mt5" in model_cfg.model_name:
        source = model_cfg.model_path or model_cfg.model_name
        model = MT5ForConditionalGeneration.from_pretrained(source, cache_dir="cache")
        for param in model.parameters():
            param.data = param.data.contiguous()
        return model

    if "bart" in model_cfg.model_name:
        source = model_cfg.model_path or model_cfg.model_name
        return BartForConditionalGeneration.from_pretrained(source, cache_dir="cache")

    if model_cfg.model_path:
        print(f"Load with model path: {model_cfg.model_path}")
        return T5ForConditionalGeneration.from_pretrained(
            model_cfg.model_path, cache_dir="cache"
        )

    return T5ForConditionalGeneration.from_pretrained(
        model_cfg.model_name, cache_dir="cache"
    )


def _build_dataset(
    split: str,
    data_cfg: DataConfig,
    model_name: str,
    max_length: int,
    text_encoder: Optional[TextEncoder] = None,
):
    query_max_length = data_cfg.query_max_length or max_length
    corpus_max_length = data_cfg.corpus_max_length or query_max_length

    dataset_kwargs = {
        "split": split,
        "dataset_path": data_cfg.dataset_path,
    }
    if data_cfg.pq_filename:
        dataset_kwargs["pq_filename"] = data_cfg.pq_filename
    if data_cfg.corpus_filename:
        dataset_kwargs["corpus_filename"] = data_cfg.corpus_filename

    modality = data_cfg.modality.lower()
    if modality == "text":
        encoder = text_encoder or TextEncoder(
            tokenizer_name=data_cfg.text_tokenizer_name or model_name,
            max_length=query_max_length,
            padding=False,
            truncation=True,
            add_special_tokens=True,
        )
        return SlueSQA5TextDataset(
            text_encoder=encoder,
            text_tokenizer_name=data_cfg.text_tokenizer_name or model_name,
            query_text_field=data_cfg.query_text_field,
            corpus_text_field=data_cfg.corpus_text_field,
            train_atomic=data_cfg.train_atomic,
            atomic_offset=data_cfg.atomic_offset,
            include_corpus=data_cfg.include_corpus,
            feature_cache_dir=data_cfg.feature_cache_dir,
            **dataset_kwargs,
        )

    if modality == "discrete_precomputed":
        if not data_cfg.precompute_root:
            raise ValueError(
                "precompute_root must be provided for discrete_precomputed modality"
            )
        return DiscreteUnitDataset(
            split=split,
            dataset_path=data_cfg.dataset_path,
            precompute_root=data_cfg.precompute_root,
            encoder_name=data_cfg.encoder_name or "wavtokenizer",
            include_corpus=data_cfg.include_corpus,
            max_length=query_max_length,
            query_max_length=query_max_length,
            corpus_max_length=corpus_max_length,
            train_atomic=data_cfg.train_atomic,
            atomic_offset=data_cfg.atomic_offset,
            corpus_splits=data_cfg.include_corpus_splits,
            corpus_chunk_size=data_cfg.corpus_chunk_size,
            corpus_chunk_stride=data_cfg.corpus_chunk_stride,
            corpus_min_tokens=data_cfg.corpus_min_tokens,
            special_token=data_cfg.special_token,
        )

    if not data_cfg.code_path:
        raise ValueError("code_path must be provided for discrete modality")

    return SlueSQA5DatasetV2(
        split=split,
        dataset_path=data_cfg.dataset_path,
        precompute_root=data_cfg.precompute_root or data_cfg.code_path,
        encoder_name=data_cfg.encoder_name or "wavtokenizer",
        include_corpus=data_cfg.include_corpus,
        train_atomic=data_cfg.train_atomic,
        atomic_offset=data_cfg.atomic_offset,
        max_length=query_max_length,
        query_max_length=query_max_length,
        corpus_max_length=corpus_max_length,
        corpus_splits=data_cfg.include_corpus_splits,
        corpus_chunk_size=data_cfg.corpus_chunk_size,
        corpus_chunk_stride=data_cfg.corpus_chunk_stride,
        corpus_min_tokens=data_cfg.corpus_min_tokens,
        special_token=data_cfg.special_token,
        **{k: v for k, v in dataset_kwargs.items() if k not in {"split", "dataset_path"}},
    )


def run(cfg: DictConfig) -> None:
    set_seed(cfg.get("seed", 42))

    data_cfg = to_dataclass(cfg.data, DataConfig)
    model_cfg = to_dataclass(cfg.model, ModelConfig)
    run_cfg = to_dataclass(cfg.run, RunConfig)
    wandb_cfg = to_dataclass(cfg.logging, WandbConfig)
    training_args = build_training_arguments(cfg.training.training_args)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name, cache_dir="cache")
    fast_tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name, cache_dir="cache"
    )

    model = _load_model(model_cfg)

    modality = data_cfg.modality.lower()
    shared_text_encoder: Optional[TextEncoder] = None
    if modality == "text":
        shared_text_encoder = TextEncoder(
            tokenizer_name=data_cfg.text_tokenizer_name or model_cfg.model_name,
            max_length=run_cfg.max_length,
            padding=False,
            truncation=True,
            add_special_tokens=True,
        )
    encoder_name_override = None
    if modality == "discrete_precomputed":
        encoder_name_override = data_cfg.encoder_name or "wavtokenizer"

    train_dataset = _build_dataset(
        "train",
        data_cfg,
        model_cfg.model_name,
        run_cfg.max_length,
        shared_text_encoder,
    )
    valid_dataset = _build_dataset(
        "validation",
        data_cfg,
        model_cfg.model_name,
        run_cfg.max_length,
        shared_text_encoder,
    )

    restrict_decode_vocab = RestrictDecodeVocab(
        valid_ids=train_dataset.valid_ids, tokenizer=tokenizer
    )

    if modality == "text":
        collator = TextIndexingCollator(
            text_encoder=shared_text_encoder or TextEncoder(
                tokenizer_name=data_cfg.text_tokenizer_name or model_cfg.model_name,
                max_length=run_cfg.max_length,
                padding=False,
                truncation=True,
                add_special_tokens=True,
            ),
            label_tokenizer=tokenizer,
            train_atomic=data_cfg.train_atomic,
        )
    else:
        collator_cls = (
            IndexingCollatorWithAtomic if data_cfg.train_atomic else IndexingCollator
        )
        collator = collator_cls(tokenizer=tokenizer, padding="longest")

    trainer = DSITrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(fast_tokenizer, train_dataset.valid_ids),
        id_max_length=run_cfg.id_max_length,
        restrict_decode_vocab=restrict_decode_vocab,
    )

    trainer.max_length = run_cfg.max_length
    trainer.num_return_sequences = run_cfg.num_return_sequences
    trainer.top_k = run_cfg.top_k
    trainer.generation_max_length = run_cfg.generation_max_length

    _maybe_init_wandb(training_args, wandb_cfg, run_cfg.run_notes, cfg)

    trainer.train()
