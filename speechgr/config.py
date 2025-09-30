from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Type, TypeVar, Union

from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments


T = TypeVar("T")


def _to_dict(cfg: Union[DictConfig, Dict[str, Any]]) -> Dict[str, Any]:
    """Return a plain dict from a Hydra config section."""

    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def to_dataclass(cfg: Union[DictConfig, Dict[str, Any]], cls: Type[T]) -> T:
    """Instantiate ``cls`` using values from ``cfg``."""

    return cls(**_to_dict(cfg))


def build_training_arguments(cfg: Union[DictConfig, Dict[str, Any]]) -> TrainingArguments:
    """Create :class:`~transformers.TrainingArguments` from a config mapping."""

    cfg_dict = _to_dict(cfg)
    output_dir = Path(cfg_dict.get("output_dir", "models"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return TrainingArguments(**cfg_dict)


@dataclass
class DataConfig:
    dataset_path: str
    code_path: Optional[str] = None
    lookup_file_name: Optional[str] = None
    special_token: int = 32000
    discrete_code_num: int = 500
    train_atomic: bool = False
    atomic_offset: Optional[int] = None
    pq_filename: Optional[str] = None
    corpus_filename: Optional[str] = None
    modality: str = "discrete"
    text_tokenizer_name: Optional[str] = None
    query_text_field: Optional[Any] = None
    corpus_text_field: Optional[Any] = None
    include_corpus: bool = True
    feature_cache_dir: Optional[str] = None
    question_cache_file: Optional[str] = None
    document_cache_file: Optional[str] = None
    precompute_root: Optional[str] = None
    encoder_name: Optional[str] = None
    include_corpus_splits: Optional[Iterable[str]] = None
    corpus_chunk_size: Optional[int] = None
    corpus_chunk_stride: Optional[int] = None
    corpus_min_tokens: int = 1
    query_max_length: Optional[int] = 512
    corpus_max_length: Optional[int] = 512


@dataclass
class ModelConfig:
    model_name: str
    model_path: Optional[str] = None
    train_continuous_embedding: bool = False
    ssl_feat_dim: int = 1024
    downsample_factor: int = 2
    hidden_dim: int = 768


@dataclass
class QFormerModelConfig(ModelConfig):
    model_type: str = "qformer"
    d_model_front: int = 768
    n_queries: int = 1
    qformer_depth: int = 2
    win_size_f: int = 17
    win_stride_f: int = 17
    freeze_t5_encoder: bool = False
    use_whisper_features: bool = False
    whisper_model_name: str = "openai/whisper-base"
    device: str = "cuda"
    apply_spec_augment: bool = False
    time_warp_param: int = 80
    freq_mask_param: int = 27
    time_mask_param: int = 100
    debug_max_samples: Optional[int] = None


@dataclass
class RunConfig:
    max_length: int = 512
    id_max_length: int = 128
    top_k: int = 10
    num_return_sequences: int = 10
    generation_max_length: Optional[int] = None
    run_notes: str = ""


@dataclass
class RankingConfig:
    max_length: int = 512
    id_max_length: int = 128
    run_notes: str = ""
    do_inference: bool = False
    do_debug: bool = False


@dataclass
class QFormerConfig:
    max_length: Optional[int] = None
    id_max_length: int = 128
    discrete_code_num: int = 500
    special_token: int = 32000
    run_notes: str = ""
    top_k: int = 10
    num_return_sequences: int = 10


@dataclass
class WandbConfig:
    project: str = "speechgr"
    entity: Optional[str] = None
    notes: str = ""
    log_hydra: bool = True
