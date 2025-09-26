"""Compatibility exports for SpeechGR dataset classes."""

from .slue_sqa5 import (
    DiscreteUnitDataset,
    ContinuousDataset,
    SlueSQA5DatasetV2,
    SlueSQA5TextDataset,
    SlueSQA5WhisperDataset,
)

__all__ = [
    "DiscreteUnitDataset",
    "ContinuousDataset",
    "SlueSQA5DatasetV2",
    "SlueSQA5TextDataset",
    "SlueSQA5WhisperDataset",
]
