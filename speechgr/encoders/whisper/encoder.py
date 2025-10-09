"""Encoders for Whisper-based continuous features."""

from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Mapping
from typing import Iterable, Dict, Any, Optional, Union

import librosa
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import WhisperModel, WhisperProcessor

from speechgr.encoders.base import ModalityEncoder
from speechgr.encoders.whisper.specaugment import SpecAugmentLB

logger = logging.getLogger(__name__)


class WhisperEncoder(ModalityEncoder):
    def __init__(
        self,
        *,
        whisper_model_name: str,
        device: str,
        apply_spec_augment: bool,
        time_warp_param: int,
        freq_mask_param: int,
        time_mask_param: int,
        cache_dtype: Union[str, torch.dtype] = torch.float32,
        audio_field: str = "audio",
        sample_id_field: str = "id",
        cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(name="whisper", cfg=cfg)
        self.processor = WhisperProcessor.from_pretrained(whisper_model_name)
        self.model = WhisperModel.from_pretrained(whisper_model_name).to(device)
        self.device = device
        self.cache_dtype = self._resolve_dtype(cache_dtype)
        self.audio_field = audio_field
        self.sample_id_field = sample_id_field
        self.apply_spec_augment = apply_spec_augment
        self.spec_augment = (
            SpecAugmentLB(
                time_warp_param=time_warp_param,
                freq_mask_param=freq_mask_param,
                time_mask_param=time_mask_param,
            )
            if apply_spec_augment
            else None
        )
        if self.spec_augment is not None:
            self.spec_augment.eval()

    def _resample_if_needed(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        if sampling_rate == 16000:
            return audio
        return librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)

    @staticmethod
    def _resolve_dtype(value: Union[str, torch.dtype]) -> torch.dtype:
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            try:
                return getattr(torch, value)
            except AttributeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Unsupported dtype string '{value}'") from exc
        raise TypeError(f"Unsupported dtype specifier: {type(value)!r}")

    def encode_audio(self, audio: np.ndarray, sampling_rate: int) -> torch.Tensor:
        audio = self._resample_if_needed(audio, sampling_rate)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        features = inputs.input_features
        if self.apply_spec_augment and self.spec_augment is not None:
            features = self.spec_augment(features)
        with torch.no_grad():
            encoder_outputs = self.model.encoder(features.to(self.device))
        hidden_states = (
            encoder_outputs.last_hidden_state.squeeze(0).to(self.cache_dtype).cpu()
        )
        return hidden_states

    def supports_precompute(self) -> bool:
        return True

    def precompute(
        self,
        dataset_split: str,
        output_dir: str,
        samples: Iterable[Dict[str, Any]],
    ) -> None:
        cache: Dict[str, Dict[str, Any]] = {}

        for sample in samples:
            if isinstance(sample, Mapping):
                sample_mapping = sample
            else:  # pragma: no cover - defensive
                raise TypeError(
                    f"Expected Mapping samples; got {type(sample)!r}"
                )

            if self.audio_field not in sample_mapping:
                raise KeyError(
                    f"Expected audio field '{self.audio_field}' in sample keys "
                    f"{list(sample_mapping.keys())}"
                )

            audio_payload = sample_mapping[self.audio_field]
            audio, sampling_rate = self._decode_audio_payload(audio_payload)

            features = self.encode_audio(audio, sampling_rate)
            sample_id = sample_mapping.get(self.sample_id_field)
            if sample_id is None:
                raise KeyError(
                    f"Expected '{self.sample_id_field}' field in dataset sample during precompute"
                )
            cache[str(sample_id)] = {
                "features": features.cpu(),
                "length": int(features.shape[0]),
                "sampling_rate": 16000,
            }

        cache_path = self.cache_path(dataset_split, output_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        self._loaded_cache = cache
        self._loaded_cache_key = (dataset_split, output_dir)

    def _load_cache(self, path: Path) -> Dict[str, Any]:
        loaded = torch.load(path, map_location="cpu")
        normalized: Dict[str, Dict[str, Any]] = {}
        for sample_id, value in loaded.items():
            key = str(sample_id)
            if isinstance(value, dict):
                tensor = value.get("features")
                if isinstance(tensor, torch.Tensor):
                    value["features"] = tensor.to(self.cache_dtype)
                    if "length" not in value:
                        value["length"] = int(tensor.shape[0])
                normalized[key] = value
                continue

            tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
            tensor = tensor.to(self.cache_dtype)
            normalized[key] = {
                "features": tensor,
                "length": int(tensor.shape[0]) if tensor.ndim > 0 else tensor.numel(),
                "sampling_rate": 16000,
            }
        return normalized

    # ------------------------------------------------------------------
    def _decode_audio_payload(self, payload: Any) -> tuple[np.ndarray, int]:
        """Support multiple HF audio representations."""

        if isinstance(payload, Mapping):
            audio = payload["array"]
            sampling_rate = payload["sampling_rate"]
            return audio, int(sampling_rate)

        if hasattr(payload, "decode"):  # TorchCodec decoder provides decode()
            decoded = payload.decode()
        elif callable(payload):  # Fallback: try invoking the object
            decoded = payload()
        else:
            raise TypeError(
                "Audio payload must be a mapping or provide a decode() method"
            )

        if isinstance(decoded, Mapping):
            return decoded["array"], int(decoded.get("sampling_rate", 16000))

        if isinstance(decoded, tuple) and len(decoded) == 2:
            audio, sampling_rate = decoded
            return audio, int(sampling_rate)

        if isinstance(decoded, np.ndarray):
            sampling_rate = getattr(payload, "sampling_rate", 16000)
            return decoded, int(sampling_rate)

        raise TypeError(
            "Decoded audio payload not understood; expected ndarray or mapping"
        )


__all__ = ["WhisperEncoder"]
