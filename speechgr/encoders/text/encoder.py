"""Encoders for text inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from speechgr.encoders.base import ModalityEncoder


class TextEncoder(ModalityEncoder):
    """Tokenizes raw text into model-ready tensors."""

    def __init__(
        self,
        *,
        tokenizer_name: str,
        max_length: Optional[int] = None,
        padding: str | bool = False,
        truncation: bool = True,
        add_special_tokens: bool = True,
        text_field: str = "text",
        sample_id_field: str = "id",
        tokenizer_kwargs: Optional[Mapping[str, object]] = None,
        cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(name="text", cfg=cfg)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_name
        )
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens
        self.text_field = text_field
        self.sample_id_field = sample_id_field
        self.tokenizer_kwargs = dict(tokenizer_kwargs or {})

    def encode(
        self,
        text: str,
        *,
        max_length: Optional[int] = None,
        padding: Optional[str | bool] = None,
        truncation: Optional[bool] = None,
        add_special_tokens: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize `text` and return tensorized features."""

        kwargs: Dict[str, object] = dict(self.tokenizer_kwargs)
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", padding if padding is not None else self.padding)
        kwargs.setdefault("truncation", truncation if truncation is not None else self.truncation)
        kwargs.setdefault(
            "max_length", max_length if max_length is not None else self.max_length
        )
        kwargs.setdefault(
            "add_special_tokens",
            add_special_tokens if add_special_tokens is not None else self.add_special_tokens,
        )
        encoded = self.tokenizer(text, **kwargs)
        return {key: value.squeeze(0) for key, value in encoded.items()}

    def encode_batch(
        self,
        texts: Iterable[str],
        *,
        max_length: Optional[int] = None,
        padding: Optional[str | bool] = "longest",
        truncation: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Batch-tokenize `texts` and return tensors."""

        kwargs: Dict[str, object] = dict(self.tokenizer_kwargs)
        kwargs.setdefault("return_tensors", "pt")
        kwargs["padding"] = padding if padding is not None else self.padding
        kwargs["truncation"] = truncation if truncation is not None else self.truncation
        kwargs["max_length"] = max_length if max_length is not None else self.max_length
        kwargs["add_special_tokens"] = self.add_special_tokens
        return self.tokenizer(list(texts), **kwargs)

    def supports_precompute(self) -> bool:
        return True

    def precompute(
        self,
        dataset_split: str,
        output_dir: str,
        samples: Iterable[Mapping[str, object]],
    ) -> None:
        """Pre-tokenize text fields and persist tensors to a consolidated cache."""

        cache: Dict[str, Dict[str, torch.Tensor]] = {}

        for sample in samples:
            if self.text_field not in sample:
                raise KeyError(
                    f"Expected text field '{self.text_field}' in sample during precompute"
                )
            text_value = sample[self.text_field]
            if not isinstance(text_value, str):
                raise TypeError(
                    f"Expected text field '{self.text_field}' to be str, got {type(text_value)!r}"
                )

            sample_id = sample.get(self.sample_id_field)
            if sample_id is None:
                raise KeyError(
                    f"Expected id field '{self.sample_id_field}' in sample during precompute"
                )

            encoded = self.encode(text_value)
            cache[str(sample_id)] = {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in encoded.items()
            }

        cache_path = self.cache_path(dataset_split, output_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        self._loaded_cache = cache
        self._loaded_cache_key = (dataset_split, output_dir)

    def _load_cache(self, path: Path) -> Dict[str, Any]:
        loaded = torch.load(path, map_location="cpu")
        return {
            str(sample_id): {
                key: value if isinstance(value, torch.Tensor) else torch.tensor(value)
                for key, value in feature_dict.items()
            }
            for sample_id, feature_dict in loaded.items()
        }


__all__ = ["TextEncoder"]
