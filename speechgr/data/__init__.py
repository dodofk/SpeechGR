"""Datasets and collators for SpeechGR."""

from importlib import import_module

__all__ = [
    "IndexingCollator",
    "IndexingCollatorWithAtomic",
    "IndexingCollatorWithMetadata",
    "ContinuousIndexingCollator",
    "WhisperIndexingCollator",
    "TextIndexingCollator",
    "DiscreteUnitDataset",
    "ContinuousDataset",
    "SlueSQA5WhisperCachedDataset",
    "SlueSQA5DatasetV2",
    "SlueSQA5TextDataset",
    "SlueSQA5WhisperDataset",
]

_LAZY_IMPORTS = {
    "IndexingCollator": ("speechgr.data.collators", "IndexingCollator"),
    "IndexingCollatorWithAtomic": (
        "speechgr.data.collators",
        "IndexingCollatorWithAtomic",
    ),
    "IndexingCollatorWithMetadata": (
        "speechgr.data.collators",
        "IndexingCollatorWithMetadata",
    ),
    "ContinuousIndexingCollator": (
        "speechgr.data.collators",
        "ContinuousIndexingCollator",
    ),
    "WhisperIndexingCollator": ("speechgr.data.collators", "WhisperIndexingCollator"),
    "TextIndexingCollator": ("speechgr.data.collators", "TextIndexingCollator"),
    "DiscreteUnitDataset": ("speechgr.data.slue.discrete", "DiscreteUnitDataset"),
    "ContinuousDataset": ("speechgr.data.slue.whisper", "ContinuousDataset"),
    "SlueSQA5WhisperCachedDataset": (
        "speechgr.data.slue.whisper",
        "SlueSQA5WhisperCachedDataset",
    ),
    "SlueSQA5DatasetV2": ("speechgr.data.slue.discrete", "SlueSQA5DatasetV2"),
    "SlueSQA5TextDataset": ("speechgr.data.slue.text", "SlueSQA5TextDataset"),
    "SlueSQA5WhisperDataset": (
        "speechgr.data.slue.whisper",
        "SlueSQA5WhisperDataset",
    ),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'speechgr.data' has no attribute '{name}'")
