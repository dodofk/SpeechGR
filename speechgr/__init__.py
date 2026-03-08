"""Utility helpers for SpeechGR configuration management."""

from importlib import import_module

from .config import (
    DataConfig,
    ModelConfig,
    QFormerModelConfig,
    RunConfig,
    RankingConfig,
    QFormerConfig,
    WandbConfig,
    build_training_arguments,
    to_dataclass,
)

__all__ = [
    "DataConfig",
    "ModelConfig",
    "QFormerModelConfig",
    "RunConfig",
    "RankingConfig",
    "QFormerConfig",
    "WandbConfig",
    "build_training_arguments",
    "to_dataclass",
    "data",
    "model",
    "trainer",
    "utils",
]

_LAZY_MODULES = {
    "data": "speechgr.data",
    "model": "speechgr.model",
    "trainer": "speechgr.trainer",
    "utils": "speechgr.utils",
}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        module = import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'speechgr' has no attribute '{name}'")
