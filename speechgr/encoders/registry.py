"""Central registry for SpeechGR modality encoders."""

from __future__ import annotations

from typing import Dict, Type

from speechgr.encoders.discrete.encoder import DiscreteCodeEncoder
from speechgr.encoders.hubert.encoder import HuBERTKMeansEncoder
from speechgr.encoders.text.encoder import TextEncoder
from speechgr.encoders.wavtokenizer.encoder import WavTokenizerEncoder
from speechgr.encoders.whisper.encoder import WhisperEncoder

_REGISTRY: Dict[str, Type] = {
    "discrete": DiscreteCodeEncoder,
    "hubert_kmeans": HuBERTKMeansEncoder,
    "text": TextEncoder,
    "wavtokenizer": WavTokenizerEncoder,
    "whisper": WhisperEncoder,
}


def get_encoder_class(name: str):
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown encoder '{name}'. Available: {', '.join(sorted(_REGISTRY))}"
        ) from exc


def list_encoders() -> Dict[str, Type]:
    """Return a shallow copy of the encoder registry."""

    return dict(_REGISTRY)


__all__ = ["get_encoder_class", "list_encoders"]
