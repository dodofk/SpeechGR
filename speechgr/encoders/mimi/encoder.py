"""Minimal Mimi discrete-token encoder scaffolding."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from speechgr.encoders.base import ModalityEncoder


class _TransformersMimiTokenizer:
    """Lazy Hugging Face-backed Mimi adapter.

    The exact upstream Mimi entrypoint still needs to be locked in for this
    repository. This adapter keeps the integration local and only attempts model
    loading when a real checkpoint path is provided.
    """

    def __init__(self, model_name_or_path: str, device: torch.device) -> None:
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model = self.model.to(device).eval()

    def __call__(self, waveform: torch.Tensor, *, sampling_rate: int) -> Any:
        inputs = self.processor(
            raw_audio=waveform.squeeze(0).cpu().numpy(),
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            if hasattr(self.model, "encode"):
                return self.model.encode(**inputs)
            return self.model(**inputs)


class MimiEncoder(ModalityEncoder):
    """Encode audio into Mimi-style discrete codes.

    The implementation is intentionally lightweight for local scaffolding:
    tests can inject a callable tokenizer directly, while real runs can provide a
    Mimi checkpoint path once the external dependency is finalized.
    """

    def __init__(
        self,
        *,
        model_name_or_path: Optional[str] = None,
        target_sample_rate: int = 24_000,
        device: str = "cpu",
        audio_field: str = "audio",
        sample_id_field: str = "id",
        tokenizer: Optional[Any] = None,
        cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(name="mimi", cfg=cfg)

        self.model_name_or_path = model_name_or_path or None
        self.target_sample_rate = int(target_sample_rate)
        self.device = torch.device(device)
        self.audio_field = audio_field
        self.sample_id_field = sample_id_field
        self._tokenizer = tokenizer

    def supports_precompute(self) -> bool:
        return True

    def encode_audio(
        self, audio: np.ndarray | torch.Tensor, sampling_rate: int
    ) -> torch.Tensor:
        """Return a 1D tensor of Mimi token ids for the supplied waveform."""

        waveform = self._prepare_waveform(audio, sampling_rate)
        encoded = self._get_tokenizer()(waveform, sampling_rate=self.target_sample_rate)
        return self._normalize_codes(encoded)

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

            cache[str(sample_id)] = {
                "codes": self.encode_audio(np.asarray(audio), int(sampling_rate)).long()
            }

        cache_path = self.cache_path(dataset_split, output_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        self._loaded_cache = cache
        self._loaded_cache_key = (dataset_split, output_dir)

    def _load_cache(self, path: Path) -> Dict[str, Any]:
        loaded = torch.load(path, map_location="cpu")
        return {str(sample_id): value for sample_id, value in loaded.items()}

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if self.model_name_or_path is None:
            raise RuntimeError(
                "MimiEncoder requires either an injected tokenizer for tests or a "
                "configured Mimi checkpoint path. TODO: set model_name_or_path once "
                "the repository pins the external Mimi dependency and weights."
            )

        self._tokenizer = _TransformersMimiTokenizer(
            model_name_or_path=self.model_name_or_path,
            device=self.device,
        )
        return self._tokenizer

    def _prepare_waveform(
        self, audio: np.ndarray | torch.Tensor, sampling_rate: int
    ) -> torch.Tensor:
        waveform = self._to_waveform_tensor(audio)
        waveform = self._mix_down(waveform)
        if int(sampling_rate) != self.target_sample_rate:
            waveform = self._resample(
                waveform,
                orig_sr=int(sampling_rate),
                target_sr=self.target_sample_rate,
            )
        return waveform.float()

    @staticmethod
    def _to_waveform_tensor(audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            tensor = torch.from_numpy(audio).float()
        else:
            tensor = audio.float()
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        if tensor.ndim == 2:
            return tensor
        raise ValueError(f"Expected 1D or 2D audio tensor, got shape {tuple(tensor.shape)}")

    @staticmethod
    def _mix_down(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[0] == 1:
            return waveform
        return waveform.mean(dim=0, keepdim=True)

    @staticmethod
    def _resample(waveform: torch.Tensor, *, orig_sr: int, target_sr: int) -> torch.Tensor:
        target_length = max(1, round(waveform.shape[-1] * target_sr / orig_sr))
        return F.interpolate(
            waveform.unsqueeze(0),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).squeeze(0)

    @staticmethod
    def _normalize_codes(encoded: Any) -> torch.Tensor:
        payload = encoded
        if isinstance(payload, Mapping):
            for key in ("codes", "audio_codes", "input_ids"):
                if key in payload:
                    payload = payload[key]
                    break
        else:
            for attr in ("codes", "audio_codes", "input_ids"):
                value = getattr(payload, attr, None)
                if value is not None:
                    payload = value
                    break

        if isinstance(payload, (list, tuple)) and payload:
            payload = payload[0]

        if not isinstance(payload, torch.Tensor):
            try:
                payload = torch.as_tensor(payload)
            except Exception as exc:  # pragma: no cover - defensive path
                raise TypeError(
                    "Mimi encoder backend must return tensor-like codes or expose a "
                    "'codes'/'audio_codes' field."
                ) from exc

        tensor = payload.detach().cpu().long().squeeze()
        if tensor.ndim == 0:
            return tensor.reshape(1)
        if tensor.ndim > 1:
            return tensor.reshape(-1)
        return tensor


__all__ = ["MimiEncoder"]
