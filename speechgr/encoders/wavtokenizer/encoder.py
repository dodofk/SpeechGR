"""Integration of WavTokenizer discrete unit encoder."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from speechgr.encoders.base import ModalityEncoder

# Ensure the local inventory package is importable without modifying its codebase.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_WAVTOKENIZER_ROOT = _REPO_ROOT / "inventory" / "WavTokenizer"
if str(_WAVTOKENIZER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WAVTOKENIZER_ROOT))

from encoder.utils import convert_audio  # type: ignore  # noqa: E402
from decoder.pretrained import WavTokenizer  # type: ignore  # noqa: E402


class WavTokenizerEncoder(ModalityEncoder):
    """Encode waveforms into discrete units using a pretrained WavTokenizer."""

    def __init__(
        self,
        *,
        config_path: str,
        model_path: str,
        bandwidth_id: int = 0,
        target_sample_rate: int = 24_000,
        target_channels: int = 1,
        device: str = "cpu",
        audio_field: str = "audio",
        sample_id_field: str = "id",
        cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(name="wavtokenizer", cfg=cfg)

        self.audio_field = audio_field
        self.sample_id_field = sample_id_field
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels

        self.device = torch.device(device)
        self.model = WavTokenizer.from_pretrained0802(config_path, model_path)
        self.model = self.model.to(self.device).eval()
        self.bandwidth_id = torch.tensor([bandwidth_id], device=self.device)

    def supports_precompute(self) -> bool:
        return True

    def encode_audio(self, audio: np.ndarray | torch.Tensor, sampling_rate: int) -> torch.Tensor:
        """Return a tensor of discrete codes for the supplied waveform."""

        waveform = self._to_waveform_tensor(audio)
        waveform = convert_audio(
            waveform,
            sampling_rate,
            self.target_sample_rate,
            self.target_channels,
        )
        waveform = waveform.to(self.device)

        with torch.no_grad():
            _features, discrete_code = self.model.encode_infer(
                waveform, bandwidth_id=self.bandwidth_id
            )
        return discrete_code.squeeze(0).cpu()

    def precompute(
        self,
        dataset_split: str,
        output_dir: str,
        samples: Iterable[Mapping[str, Any]],
    ) -> None:
        cache: Dict[str, Dict[str, torch.Tensor]] = {}
        for sample in samples:
            if self.audio_field not in sample:
                raise KeyError(
                    f"Expected audio field '{self.audio_field}' in dataset sample"
                )
            audio_entry = sample[self.audio_field]
            audio = audio_entry["array"]
            sampling_rate = audio_entry["sampling_rate"]

            sample_id = sample.get(self.sample_id_field)
            if sample_id is None:
                raise KeyError(
                    f"Expected id field '{self.sample_id_field}' in dataset sample"
                )

            codes = self.encode_audio(np.asarray(audio), int(sampling_rate))
            cache[str(sample_id)] = {"codes": codes.long()}

        cache_path = self.cache_path(dataset_split, output_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        self._loaded_cache = cache
        self._loaded_cache_key = (dataset_split, output_dir)

    def _load_cache(self, path: Path) -> Dict[str, Any]:
        loaded = torch.load(path, map_location="cpu")
        return {str(k): v for k, v in loaded.items()}

    def _to_waveform_tensor(self, audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            tensor = torch.from_numpy(audio).float()
        else:
            tensor = audio.float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor


__all__ = ["WavTokenizerEncoder"]
