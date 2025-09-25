"""Precompute modality features via registered encoders."""

from __future__ import annotations

from pathlib import Path

from hydra import main as hydra_main
from omegaconf import DictConfig
from datasets import load_dataset

from speechgr.encoders.registry import get_encoder_class, list_encoders


@hydra_main(version_base=None, config_path="../../configs", config_name="precompute")
def main(cfg: DictConfig) -> None:
    encoder_name = cfg.encoder.name
    try:
        encoder_cls = get_encoder_class(encoder_name)
    except KeyError as exc:
        available = ", ".join(sorted(list_encoders()))
        raise ValueError(
            f"Unsupported encoder '{encoder_name}'. Available: {available}"
        ) from exc
    encoder = encoder_cls(**cfg.encoder.params)
    if not encoder.supports_precompute():
        raise ValueError(f"Encoder '{encoder_name}' does not support precompute")

    dataset = load_dataset(cfg.dataset.name, cfg.dataset.config)
    split = cfg.dataset.split
    if split not in dataset:
        raise ValueError(f"Split '{split}' not present in dataset")

    target_dir = Path(cfg.output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    cache_path = encoder.cache_path(split, str(target_dir))
    encoder.precompute(split, str(target_dir), dataset[split])
    print(f"Saved {split} cache to {cache_path}")


if __name__ == "__main__":
    main()
