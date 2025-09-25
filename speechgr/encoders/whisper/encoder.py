"""Encoders for Whisper-based continuous features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Dict, Any, Optional

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
        cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(name="whisper", cfg=cfg)
        self.processor = WhisperProcessor.from_pretrained(whisper_model_name)
        self.model = WhisperModel.from_pretrained(whisper_model_name).to(device)
        self.device = device
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

    def encode_audio(self, audio: np.ndarray, sampling_rate: int) -> torch.Tensor:
        audio = self._resample_if_needed(audio, sampling_rate)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        features = inputs.input_features
        if self.apply_spec_augment and self.spec_augment is not None:
            features = self.spec_augment(features)
        with torch.no_grad():
            encoder_outputs = self.model.encoder(features.to(self.device))
        hidden_states = encoder_outputs.last_hidden_state.squeeze(0).cpu()
        return hidden_states

    def supports_precompute(self) -> bool:
        return True

    def precompute(
        self,
        dataset_split: str,
        output_dir: str,
        samples: Iterable[Dict[str, Any]],
    ) -> None:
        cache: Dict[str, torch.Tensor] = {}

        for sample in samples:
            audio = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]
            features = self.encode_audio(audio, sampling_rate)
            sample_id = sample.get("id")
            if sample_id is None:
                raise KeyError("Expected 'id' field in dataset sample during precompute")
            cache[str(sample_id)] = features.cpu()

        cache_path = self.cache_path(dataset_split, output_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        self._loaded_cache = cache
        self._loaded_cache_key = (dataset_split, output_dir)

    def _load_cache(self, path: Path) -> Dict[str, Any]:
        loaded = torch.load(path, map_location="cpu")
        return {str(sample_id): tensor for sample_id, tensor in loaded.items()}


__all__ = ["WhisperEncoder"]
