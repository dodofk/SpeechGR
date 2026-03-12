"""Minimal Mimi discrete-token encoder scaffolding."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm.auto import tqdm

from speechgr.encoders.base import ModalityEncoder


DEFAULT_MIMI_MODEL_NAME_OR_PATH = "kyutai/mimi"
DEFAULT_MIMI_CHECKPOINT_PATH = DEFAULT_MIMI_MODEL_NAME_OR_PATH

logger = logging.getLogger(__name__)


def _default_mimi_model_name_or_path() -> str:
    return os.environ.get("MIMI_MODEL_NAME_OR_PATH", DEFAULT_MIMI_MODEL_NAME_OR_PATH)


class _TransformersMimiTokenizer:
    """Lazy Hugging Face-backed Mimi adapter.

    The exact upstream Mimi entrypoint still needs to be locked in for this
    repository. This adapter keeps the integration local and only attempts model
    loading when a real checkpoint path is provided.
    """

    def __init__(self, model_name_or_path: str, device: torch.device) -> None:
        from transformers import AutoFeatureExtractor

        try:
            from transformers import MimiConfig
            from transformers import MimiModel
        except ImportError:  # pragma: no cover - compatibility fallback
            MimiConfig = None
            MimiModel = None
            from transformers import AutoConfig, AutoModel

        self.device = device
        if MimiConfig is not None:
            self.config = MimiConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
        else:  # pragma: no cover - exercised only on older transformers versions
            self.config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
        self.codebook_size = int(getattr(self.config, "codebook_size", 2048))
        self.num_quantizers = int(getattr(self.config, "num_quantizers", 32))
        self.num_semantic_quantizers = int(
            getattr(self.config, "num_semantic_quantizers", 1)
        )
        self.frame_rate = float(getattr(self.config, "frame_rate", 12.5))
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        if MimiModel is not None:
            self.model = MimiModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
        else:  # pragma: no cover - exercised only on older transformers versions
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
        self.model = self.model.to(device).eval()

    def __call__(self, waveform: Any, *, sampling_rate: int) -> Any:
        if isinstance(waveform, (list, tuple)):
            raw_audio = []
            for item in waveform:
                if isinstance(item, torch.Tensor):
                    raw_audio.append(item.squeeze().cpu().numpy())
                else:
                    raw_audio.append(np.asarray(item).squeeze())
        else:
            raw_audio = waveform.cpu().numpy()
            if waveform.ndim == 2 and waveform.shape[0] > 1:
                raw_audio = [raw_audio[idx] for idx in range(raw_audio.shape[0])]
            elif waveform.ndim == 2 and waveform.shape[0] == 1:
                raw_audio = raw_audio[0]
        inputs = self.feature_extractor(
            raw_audio=raw_audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            input_values = inputs.get("input_values")
            padding_mask = inputs.get("padding_mask")
            if isinstance(padding_mask, torch.Tensor):
                padding_mask = padding_mask.to(self.device)
            if hasattr(self.model, "encode") and input_values is not None:
                return self.model.encode(input_values, padding_mask=padding_mask)
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
        code_selection: str = "semantic_only",
        num_selected_quantizers: Optional[int] = None,
        output_vocab_size: Optional[int] = None,
        batch_size: int = 1,
        tokenizer: Optional[Any] = None,
        cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(name="mimi", cfg=cfg)

        self.model_name_or_path = model_name_or_path or _default_mimi_model_name_or_path()
        self.target_sample_rate = int(target_sample_rate)
        self.device = torch.device(device)
        self.audio_field = audio_field
        self.sample_id_field = sample_id_field
        self.code_selection = code_selection
        self.num_selected_quantizers = num_selected_quantizers
        self.output_vocab_size = output_vocab_size
        self.batch_size = max(1, int(batch_size))
        self._tokenizer = tokenizer

    def supports_precompute(self) -> bool:
        return True

    def encode_audio(
        self, audio: np.ndarray | torch.Tensor, sampling_rate: int
    ) -> torch.Tensor:
        """Return a 1D tensor of Mimi token ids for the supplied waveform."""

        waveform = self._prepare_waveform(audio, sampling_rate)
        encoded = self._get_tokenizer()(waveform, sampling_rate=self.target_sample_rate)
        codes = self._normalize_codes(
            encoded,
            code_selection=self.code_selection,
            num_selected_quantizers=self.num_selected_quantizers,
            codebook_size=self._infer_codebook_size(),
            num_semantic_quantizers=self._infer_num_semantic_quantizers(),
        )
        self._validate_output_vocab(codes)
        return codes

    def precompute(
        self,
        dataset_split: str,
        output_dir: str,
        samples: Iterable[Mapping[str, Any]],
    ) -> None:
        cache: Dict[str, Dict[str, torch.Tensor]] = {}
        logger.info(
            "Starting Mimi precompute split=%s output_dir=%s code_selection=%s batch_size=%s",
            dataset_split,
            output_dir,
            self.code_selection,
            self.batch_size,
        )
        total = None
        try:
            total = len(samples)
        except TypeError:
            total = None

        pending_ids: list[str] = []
        pending_waveforms: list[torch.Tensor] = []

        def flush_pending() -> None:
            if not pending_ids:
                return
            codes_batch = self._encode_waveforms(pending_waveforms)
            if len(codes_batch) != len(pending_ids):
                raise ValueError(
                    "Mimi batch encoding returned {} sequences for {} pending ids".format(
                        len(codes_batch),
                        len(pending_ids),
                    )
                )
            for sample_id, codes in zip(pending_ids, codes_batch):
                cache[str(sample_id)] = {"codes": codes.long()}
            pending_ids.clear()
            pending_waveforms.clear()

        for sample in tqdm(
            samples,
            desc=f"mimi precompute:{dataset_split}",
            total=total,
            unit="sample",
        ):
            if self.audio_field not in sample:
                raise KeyError(
                    f"Expected audio field '{self.audio_field}' in dataset sample"
                )

            audio_entry = sample[self.audio_field]
            audio, sampling_rate = self._extract_audio_payload(audio_entry)

            sample_id = sample.get(self.sample_id_field)
            if sample_id is None:
                raise KeyError(
                    f"Expected id field '{self.sample_id_field}' in dataset sample"
                )

            waveform = self._prepare_waveform(np.asarray(audio), int(sampling_rate))
            pending_ids.append(str(sample_id))
            pending_waveforms.append(waveform)
            if len(pending_ids) >= self.batch_size:
                flush_pending()

        flush_pending()

        cache_path = self.cache_path(dataset_split, output_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        self._loaded_cache = cache
        self._loaded_cache_key = (dataset_split, output_dir)
        logger.info(
            "Finished Mimi precompute split=%s samples=%d cache_path=%s",
            dataset_split,
            len(cache),
            cache_path,
        )

    def _load_cache(self, path: Path) -> Dict[str, Any]:
        loaded = torch.load(path, map_location="cpu")
        return {str(sample_id): value for sample_id, value in loaded.items()}

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        logger.info(
            "Loading Mimi model name_or_path=%s device=%s code_selection=%s",
            self.model_name_or_path,
            self.device,
            self.code_selection,
        )
        self._tokenizer = _TransformersMimiTokenizer(
            model_name_or_path=self.model_name_or_path,
            device=self.device,
        )
        logger.info(
            "Loaded Mimi model codebook_size=%s num_quantizers=%s num_semantic_quantizers=%s",
            getattr(self._tokenizer, "codebook_size", "?"),
            getattr(self._tokenizer, "num_quantizers", "?"),
            getattr(self._tokenizer, "num_semantic_quantizers", "?"),
        )
        return self._tokenizer

    def _encode_waveforms(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        if not waveforms:
            return []

        tokenizer = self._get_tokenizer()
        if len(waveforms) == 1:
            encoded = tokenizer(
                waveforms[0],
                sampling_rate=self.target_sample_rate,
            )
        else:
            encoded = tokenizer(
                [waveform.squeeze(0) for waveform in waveforms],
                sampling_rate=self.target_sample_rate,
            )
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

        if not isinstance(payload, torch.Tensor):
            payload = torch.as_tensor(payload)

        tensor = payload.detach().cpu().long()
        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        if tensor.ndim == 2:
            outputs = [row.contiguous() for row in tensor]
        elif tensor.ndim == 3:
            outputs = []
            for waveform, row in zip(waveforms, tensor):
                codes = self._normalize_codes(
                    row,
                    code_selection=self.code_selection,
                    num_selected_quantizers=self.num_selected_quantizers,
                    codebook_size=self._infer_codebook_size(),
                    num_semantic_quantizers=self._infer_num_semantic_quantizers(),
                )
                expected_len = self._expected_code_length(int(waveform.shape[-1]))
                outputs.append(codes[:expected_len].contiguous())
        else:
            raise ValueError(
                f"Unexpected Mimi batch code tensor shape {tuple(tensor.shape)}"
            )

        for codes in outputs:
            self._validate_output_vocab(codes)
        return outputs

    def _infer_codebook_size(self) -> int:
        tokenizer = self._get_tokenizer()
        return int(getattr(tokenizer, "codebook_size", 2048))

    def _infer_num_semantic_quantizers(self) -> int:
        tokenizer = self._get_tokenizer()
        return int(getattr(tokenizer, "num_semantic_quantizers", 1))

    def _infer_frame_rate(self) -> float:
        tokenizer = self._get_tokenizer()
        return float(getattr(tokenizer, "frame_rate", 12.5))

    def _expected_code_length(self, num_samples: int) -> int:
        num_frames = max(
            1,
            int(np.ceil(float(num_samples) * self._infer_frame_rate() / float(self.target_sample_rate))),
        )
        selection = self.code_selection.strip().lower()
        if selection == "semantic_only":
            multiplier = max(1, self._infer_num_semantic_quantizers())
        elif selection == "first_n":
            multiplier = max(1, int(self.num_selected_quantizers or 1))
        elif selection == "all_flattened":
            multiplier = max(1, int(getattr(self._get_tokenizer(), "num_quantizers", 1)))
        else:
            multiplier = 1
        return num_frames * multiplier

    def _validate_output_vocab(self, codes: torch.Tensor) -> None:
        if self.output_vocab_size is None or codes.numel() == 0:
            return
        max_code = int(codes.max().item())
        if max_code >= int(self.output_vocab_size):
            raise ValueError(
                "Mimi codes exceed configured output_vocab_size={} (max token id={}). "
                "Increase the configured discrete/codebook vocabulary or reduce the "
                "selected number of quantizers.".format(
                    self.output_vocab_size,
                    max_code,
                )
            )

    @staticmethod
    def _extract_audio_payload(audio_entry: Any) -> tuple[np.ndarray, int]:
        if isinstance(audio_entry, Mapping):
            if "array" in audio_entry and "sampling_rate" in audio_entry:
                return np.asarray(audio_entry["array"]), int(audio_entry["sampling_rate"])

            if "bytes" in audio_entry and audio_entry["bytes"] is not None:
                with io.BytesIO(audio_entry["bytes"]) as buffer:
                    audio, sampling_rate = sf.read(buffer)
                return np.asarray(audio), int(sampling_rate)

            if "path" in audio_entry and audio_entry["path"]:
                audio, sampling_rate = sf.read(audio_entry["path"])
                return np.asarray(audio), int(sampling_rate)

        raise ValueError(
            "Unsupported audio payload. Expected mapping with "
            "('array','sampling_rate') or ('bytes'|'path')."
        )

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
    def _normalize_codes(
        encoded: Any,
        *,
        code_selection: str,
        num_selected_quantizers: Optional[int],
        codebook_size: int,
        num_semantic_quantizers: int,
    ) -> torch.Tensor:
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

        tensor = payload.detach().cpu().long()
        if tensor.ndim == 0:
            return tensor.reshape(1)

        if tensor.ndim == 3:
            if tensor.shape[0] != 1:
                raise ValueError(
                    "Expected Mimi codes with batch dimension 1, got shape {}".format(
                        tuple(tensor.shape)
                    )
                )
            tensor = tensor[0]

        if tensor.ndim == 2 and tensor.shape[0] == 1:
            return tensor[0].contiguous()

        if tensor.ndim == 2:
            selection = code_selection.strip().lower()
            if selection == "semantic_only":
                keep = max(1, min(num_semantic_quantizers, tensor.shape[0]))
                selected = tensor[:keep]
                if selected.shape[0] == 1:
                    return selected[0].contiguous()
                return MimiEncoder._interleave_quantizers(selected, codebook_size)

            if selection == "first_n":
                if num_selected_quantizers is None:
                    raise ValueError(
                        "num_selected_quantizers must be provided when "
                        "code_selection='first_n'"
                    )
                keep = max(1, min(int(num_selected_quantizers), tensor.shape[0]))
                selected = tensor[:keep]
                if selected.shape[0] == 1:
                    return selected[0].contiguous()
                return MimiEncoder._interleave_quantizers(selected, codebook_size)

            if selection == "all_flattened":
                return tensor.reshape(-1)

            raise ValueError(
                "Unsupported Mimi code_selection='{}'; expected one of "
                "['semantic_only', 'first_n', 'all_flattened']".format(
                    code_selection
                )
            )

        if tensor.ndim > 2:
            return tensor.reshape(-1)
        return tensor

    @staticmethod
    def _interleave_quantizers(tensor: torch.Tensor, codebook_size: int) -> torch.Tensor:
        """Convert [Q, T] codes into a 1D stream with quantizer offsets.

        This is a fallback for experiments that keep more than one Mimi codebook.
        The sequence is interleaved in time-major order to preserve local alignment:
        q0_t0, q1_t0, ..., q0_t1, q1_t1, ...
        """

        num_quantizers, num_frames = tensor.shape
        offsets = torch.arange(num_quantizers, dtype=torch.long).unsqueeze(1) * int(
            codebook_size
        )
        shifted = tensor.long() + offsets
        return shifted.transpose(0, 1).reshape(num_frames * num_quantizers).contiguous()


__all__ = [
    "DEFAULT_MIMI_CHECKPOINT_PATH",
    "DEFAULT_MIMI_MODEL_NAME_OR_PATH",
    "MimiEncoder",
]
