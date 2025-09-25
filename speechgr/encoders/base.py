"""Abstract base classes for modality encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from omegaconf import DictConfig


class ModalityEncoder(ABC):
    """Base interface for modality-specific encoders."""

    name: str

    def __init__(self, name: str, cfg: Optional[DictConfig] = None) -> None:
        self.name = name
        self.cfg = cfg or DictConfig({})
        self._loaded_cache: Optional[Dict[str, Any]] = None
        self._loaded_cache_key: Optional[Tuple[str, str]] = None

    def supports_precompute(self) -> bool:
        """Whether this encoder can precompute features independently."""

        return False

    def cache_path(self, dataset_split: str, output_dir: str) -> Path:
        """Return the canonical cache path for ``dataset_split`` in ``output_dir``."""

        return Path(output_dir) / f"{dataset_split}_{self.name}.pt"

    def has_precomputed(self, dataset_split: str, output_dir: str) -> bool:
        """Check whether a cache file already exists for the given split."""

        return self.cache_path(dataset_split, output_dir).exists()

    def load_feature(
        self, dataset_split: str, output_dir: str, sample_id: str
    ) -> Any:
        """Load a cached feature for ``sample_id`` (and memoise results).

        This is the public entry point that datasets should use; it lazily
        materialises the underlying cache via :meth:`_load_cache` when first
        accessed and keeps it in memory for subsequent lookups.
        """

        cache = self._ensure_cache(dataset_split, output_dir)
        try:
            return cache[sample_id]
        except KeyError as exc:  # pragma: no cover - surfaced to caller
            raise KeyError(
                f"Sample '{sample_id}' missing from cache for split "
                f"'{dataset_split}'"
            ) from exc

    def clear_cache(self) -> None:
        """Forget any in-memory cache copy."""

        self._loaded_cache = None
        self._loaded_cache_key = None

    def _ensure_cache(self, dataset_split: str, output_dir: str) -> Dict[str, Any]:
        key = (dataset_split, output_dir)
        if self._loaded_cache_key != key:
            path = self.cache_path(dataset_split, output_dir)
            if not path.exists():
                raise FileNotFoundError(
                    f"Cache file '{path}' not found for encoder '{self.name}'"
                )
            self._loaded_cache = self._load_cache(path)
            self._loaded_cache_key = key
        return self._loaded_cache or {}

    def _load_cache(self, path: Path) -> Dict[str, Any]:
        """Load a cache file from ``path`` (override in subclasses with caches)."""

        raise NotImplementedError(
            f"Encoder '{self.name}' does not provide cache loading"
        )

    def precompute(
        self,
        dataset_split: str,
        output_dir: str,
        samples: Iterable[Dict[str, Any]],
    ) -> None:
        """Optional helper to precompute features for a dataset split."""

        raise NotImplementedError("precompute is not implemented for this encoder")


__all__ = ["ModalityEncoder"]
