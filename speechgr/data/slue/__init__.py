"""SLUE SQA5 dataset implementations."""

from importlib import import_module

__all__ = [
    "SLUESQA5Dataset",
    "DiscreteUnitDataset",
    "SlueSQA5DatasetV2",
    "SlueSQA5TextDataset",
    "ContinuousDataset",
    "SlueSQA5WhisperCachedDataset",
    "SlueSQA5WhisperDataset",
]

_LAZY_IMPORTS = {
    "SLUESQA5Dataset": ("speechgr.data.slue.base", "SLUESQA5Dataset"),
    "DiscreteUnitDataset": ("speechgr.data.slue.discrete", "DiscreteUnitDataset"),
    "SlueSQA5DatasetV2": ("speechgr.data.slue.discrete", "SlueSQA5DatasetV2"),
    "SlueSQA5TextDataset": ("speechgr.data.slue.text", "SlueSQA5TextDataset"),
    "ContinuousDataset": ("speechgr.data.slue.whisper", "ContinuousDataset"),
    "SlueSQA5WhisperCachedDataset": (
        "speechgr.data.slue.whisper",
        "SlueSQA5WhisperCachedDataset",
    ),
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
    raise AttributeError(f"module 'speechgr.data.slue' has no attribute '{name}'")
