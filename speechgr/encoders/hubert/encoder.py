"""HuBERT encoders powered by fairseq k-means checkpoints."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from speechgr.encoders.base import ModalityEncoder

try:
    import librosa  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    librosa = None  # type: ignore

logger = logging.getLogger(__name__)


class HuBERTKMeansEncoder(ModalityEncoder):
    """Generate HuBERT hidden states and map them to k-means cluster ids."""

    def __init__(
        self,
        *,
        ckpt_path: str,
        kmeans_path: str,
        layer: int = 6,
        device: str = "cuda",
        audio_field: str = "audio",
        sample_id_field: str = "id",
        fairseq_root: Optional[str] = None,
        max_chunk: int = 1_600_000,
        cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(name="hubert_kmeans", cfg=cfg)

        if fairseq_root:
            root = Path(fairseq_root).expanduser().resolve()
            if not root.exists():
                raise FileNotFoundError(f"fairseq root '{root}' does not exist")
            sys.path.insert(0, str(root))
            sys.path.insert(0, str(root / "examples" / "hubert" / "simple_kmeans"))

        from fairseq import checkpoint_utils  # type: ignore

        self.audio_field = audio_field
        self.sample_id_field = sample_id_field
        self.layer = layer
        self.max_chunk = max_chunk
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = models[0].eval().to(self.device)
        self.task = task
        self.expected_sample_rate = getattr(self.task.cfg, "sample_rate", 16_000)

        self.kmeans = joblib.load(kmeans_path)
        self._supports_precompute = True

    def supports_precompute(self) -> bool:
        return self._supports_precompute

    def encode_audio(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Encode an audio array into a sequence of k-means cluster ids."""

        waveform = self._ensure_sample_rate(audio, sampling_rate)
        features = self._extract_features(waveform)
        codes = self.kmeans.predict(features.numpy())
        return codes.astype(np.int64)

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
            cache[str(sample_id)] = {
                "codes": torch.tensor(codes, dtype=torch.long)
            }

        cache_path = self.cache_path(dataset_split, output_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        self._loaded_cache = cache
        self._loaded_cache_key = (dataset_split, output_dir)

    def load_feature(
        self, dataset_split: str, output_dir: str, sample_id: str
    ) -> Dict[str, torch.Tensor]:
        return super().load_feature(dataset_split, output_dir, sample_id)

    def _load_cache(self, path: Path) -> Dict[str, Any]:
        loaded = torch.load(path, map_location="cpu")
        return {str(k): v for k, v in loaded.items()}

    def _ensure_sample_rate(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        if sampling_rate == self.expected_sample_rate:
            return audio
        if librosa is None:
            raise RuntimeError(
                "librosa is required to resample audio for HuBERT encoders"
            )
        return librosa.resample(
            audio.astype(np.float32), sampling_rate, self.expected_sample_rate
        )

    def _extract_features(self, waveform: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(waveform).float().to(self.device).unsqueeze(0)
        if getattr(self.task.cfg, "normalize", False):
            tensor = F.layer_norm(tensor, tensor.shape)

        chunks = []
        for start in range(0, tensor.size(1), self.max_chunk):
            chunk = tensor[:, start : start + self.max_chunk]
            feat_chunk = self.model.extract_features(
                source=chunk,
                padding_mask=None,
                mask=False,
                output_layer=self.layer,
            )
            if isinstance(feat_chunk, tuple):
                chunk_tensor = feat_chunk[0]
            elif isinstance(feat_chunk, dict):
                chunk_tensor = feat_chunk["x"]
            else:  # pragma: no cover - defensive
                chunk_tensor = feat_chunk
            chunks.append(chunk_tensor.detach())

        features = torch.cat(chunks, dim=1).squeeze(0).cpu()
        return features


__all__ = ["HuBERTKMeansEncoder"]
