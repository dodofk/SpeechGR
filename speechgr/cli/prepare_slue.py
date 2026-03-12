"""Utilities to materialize SLUE-SQA5 CSVs and encoder precomputations."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping

from datasets import Audio, load_dataset
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

import torch

from speechgr.encoders.registry import get_encoder_class

_SPLITS = ["train", "validation", "test", "verified_test"]

logger = logging.getLogger(__name__)


def _iter_with_progress(rows, *, desc: str):
    total = None
    try:
        total = len(rows)
    except TypeError:
        total = None
    return tqdm(rows, desc=desc, total=total, unit="row")


def _load_slue_dataset(cfg: DictConfig):
    dataset_name = cfg.get("dataset_name", "asapp/slue-phase-2")
    dataset_config = cfg.get("dataset_config", "sqa5")
    streaming = bool(cfg.get("streaming", False))
    decode_audio = bool(cfg.get("decode_audio", True))

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        streaming=streaming,
    )

    logger.info(
        "Loaded dataset name=%s config=%s streaming=%s decode_audio=%s",
        dataset_name,
        dataset_config,
        streaming,
        decode_audio,
    )

    audio_feature = Audio(decode=decode_audio)
    dataset = dataset.cast_column("question_audio", audio_feature)
    dataset = dataset.cast_column("document_audio", audio_feature)
    logger.info("Casted question_audio and document_audio with decode=%s", decode_audio)
    return dataset


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_split_csv(split: str, data, csv_dir: Path, doc_texts: Dict[str, str]) -> None:
    path = csv_dir / f"{split}.csv"
    fieldnames = [
        "question_id",
        "question_text",
        "document_id",
        "document_text",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in _iter_with_progress(data, desc=f"write csv:{split}"):
            question_id = row["question_id"]
            document_id = row["document_id"]
            question_text = row.get("normalized_question_text") or row.get(
                "raw_question_text", ""
            )
            document_text = row.get("normalized_document_text") or row.get(
                "raw_document_text", ""
            )
            writer.writerow(
                {
                    "question_id": question_id,
                    "question_text": question_text,
                    "document_id": document_id,
                    "document_text": document_text,
                }
            )
            if document_id not in doc_texts:
                doc_texts[document_id] = document_text
    logger.info("Wrote split CSV for %s to %s", split, path)


def _write_corpus_csv(doc_texts: Dict[str, str], csv_dir: Path) -> None:
    path = csv_dir / "corpus.csv"
    fieldnames = ["document_id", "document_text"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for doc_id, doc_text in tqdm(doc_texts.items(), desc="write csv:corpus", unit="doc"):
            writer.writerow({"document_id": doc_id, "document_text": doc_text})
    logger.info("Wrote corpus CSV to %s", path)


def _corpus_iterator(dataset) -> Iterator[Mapping[str, object]]:
    seen = set()
    for split in _SPLITS:
        logger.info("Scanning corpus candidates from split=%s", split)
        for row in _iter_with_progress(dataset[split], desc=f"corpus scan:{split}"):
            doc_id = row["document_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            yield {
                "document_id": doc_id,
                "document_audio": row["document_audio"],
                "normalized_document_text": row.get("normalized_document_text"),
            }


def _precompute_split(
    encoder,
    split: str,
    dataset_split,
    output_dir: Path,
    *,
    skip_existing: bool,
    force_recompute: bool,
) -> bool:
    logger.info("Starting question precompute split=%s output_dir=%s", split, output_dir)
    cache_path = encoder.cache_path(split, str(output_dir))
    if skip_existing and not force_recompute and cache_path.exists():
        logger.info(
            "Skipping question precompute split=%s because cache already exists at %s",
            split,
            cache_path,
        )
        return False
    encoder.precompute(split, str(output_dir), dataset_split)
    logger.info("Finished question precompute split=%s", split)
    return True


def _precompute_corpus(
    encoder,
    dataset,
    output_dir: Path,
    *,
    skip_existing: bool,
    force_recompute: bool,
) -> bool:
    logger.info("Starting corpus precompute output_dir=%s", output_dir)
    cache_path = encoder.cache_path("corpus", str(output_dir))
    if skip_existing and not force_recompute and cache_path.exists():
        logger.info(
            "Skipping corpus precompute because cache already exists at %s",
            cache_path,
        )
        return False
    encoder.precompute("corpus", str(output_dir), _corpus_iterator(dataset))
    logger.info("Finished corpus precompute")
    return True


def _instantiate_encoder(name: str, params: Dict[str, object]):
    encoder_cls = get_encoder_class(name)
    logger.info("Instantiating encoder name=%s with params keys=%s", name, sorted(params.keys()))
    return encoder_cls(**params)


def _length_statistics(cache_path: Path, value_key: str = "codes") -> Dict[str, float]:
    cache = torch.load(cache_path, map_location="cpu")
    lengths = []
    for entry in cache.values():
        payload = entry
        if isinstance(payload, dict):
            payload = payload[value_key]
        tensor = payload if isinstance(payload, torch.Tensor) else torch.tensor(payload)
        lengths.append(int(tensor.numel()))

    if not lengths:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0, "max": 0}

    arr = torch.tensor(lengths, dtype=torch.float32)
    return {
        "count": int(arr.numel()),
        "mean": float(arr.mean().item()),
        "std": float(arr.std(unbiased=False).item()),
        "min": int(arr.min().item()),
        "max": int(arr.max().item()),
    }


@hydra_main(version_base=None, config_path="../../configs/prepare", config_name="slue_sqa5")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    dataset = _load_slue_dataset(cfg)

    output_root = Path(cfg.output_root).resolve()
    csv_dir = _ensure_dir(output_root / "csv")
    precompute_dir = _ensure_dir(output_root / "precomputed")
    skip_existing = bool(cfg.get("skip_existing_precompute", True))
    force_recompute = bool(cfg.get("force_recompute", False))
    logger.info("Output root: %s", output_root)
    logger.info(
        "Precompute policy: skip_existing=%s force_recompute=%s",
        skip_existing,
        force_recompute,
    )

    # 1. Write CSV manifests and gather corpus text.
    doc_texts: Dict[str, str] = {}
    for split in _SPLITS:
        logger.info("Writing CSV manifest for split=%s", split)
        _write_split_csv(split, dataset[split], csv_dir, doc_texts)
    _write_corpus_csv(doc_texts, csv_dir)
    logger.info("Finished CSV materialization")

    # 2. Precompute question encodings.
    question_cfg = OmegaConf.to_container(cfg.encoder.question, resolve=True) or {}
    document_cfg = OmegaConf.to_container(cfg.encoder.document, resolve=True) or {}

    question_params = question_cfg.get("params", {})
    document_params = document_cfg.get("params", {})

    question_encoder = _instantiate_encoder(cfg.encoder.name, question_params)
    document_encoder = _instantiate_encoder(cfg.encoder.name, document_params)
    logger.info("Question encoder ready")
    logger.info("Document encoder ready")

    stats: Dict[str, Dict[str, Dict[str, float]]] = {"question": {}, "corpus": {}}

    for split in _SPLITS:
        split_dir = _ensure_dir(precompute_dir / split)
        _precompute_split(
            question_encoder,
            split,
            dataset[split],
            split_dir,
            skip_existing=skip_existing,
            force_recompute=force_recompute,
        )
        question_encoder.clear_cache()
        cache_path = split_dir / f"{split}_{cfg.encoder.name}.pt"
        stats["question"][split] = _length_statistics(cache_path)
        logger.info("Length stats for question split=%s: %s", split, stats["question"][split])

    corpus_dir = _ensure_dir(precompute_dir / "corpus")
    _precompute_corpus(
        document_encoder,
        dataset,
        corpus_dir,
        skip_existing=skip_existing,
        force_recompute=force_recompute,
    )
    document_encoder.clear_cache()
    stats["corpus"] = _length_statistics(corpus_dir / f"corpus_{cfg.encoder.name}.pt")
    logger.info("Length stats for corpus: %s", stats["corpus"])

    stats_path = output_root / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("Wrote code length statistics to %s", stats_path)


if __name__ == "__main__":
    main()
