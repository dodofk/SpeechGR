"""Modality encoders for SpeechGR."""

from importlib import import_module

from .base import ModalityEncoder

__all__ = [
    "ModalityEncoder",
    "DiscreteCodeEncoder",
    "HuBERTKMeansEncoder",
    "MimiEncoder",
    "TextEncoder",
    "WavTokenizerEncoder",
    "WhisperEncoder",
]

_LAZY_IMPORTS = {
    "DiscreteCodeEncoder": ("speechgr.encoders.discrete.encoder", "DiscreteCodeEncoder"),
    "HuBERTKMeansEncoder": ("speechgr.encoders.hubert.encoder", "HuBERTKMeansEncoder"),
    "MimiEncoder": ("speechgr.encoders.mimi.encoder", "MimiEncoder"),
    "TextEncoder": ("speechgr.encoders.text.encoder", "TextEncoder"),
    "WavTokenizerEncoder": (
        "speechgr.encoders.wavtokenizer.encoder",
        "WavTokenizerEncoder",
    ),
    "WhisperEncoder": ("speechgr.encoders.whisper.encoder", "WhisperEncoder"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'speechgr.encoders' has no attribute '{name}'")
