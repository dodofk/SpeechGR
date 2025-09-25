"""Utilities to materialize SLUE-SQA5 CSVs and encoder precomputations."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping

from datasets import load_dataset
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

from speechgr.encoders.registry import get_encoder_class

_SPLITS = ["train", "validation", "test", "verified_test"]


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
        for row in data:
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


def _write_corpus_csv(doc_texts: Dict[str, str], csv_dir: Path) -> None:
    path = csv_dir / "corpus.csv"
    fieldnames = ["document_id", "document_text"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for doc_id, doc_text in doc_texts.items():
            writer.writerow({"document_id": doc_id, "document_text": doc_text})


def _corpus_iterator(dataset) -> Iterator[Mapping[str, object]]:
    seen = set()
    for split in _SPLITS:
        for row in dataset[split]:
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
    encoder, split: str, dataset_split, output_dir: Path
) -> None:
    encoder.precompute(split, str(output_dir), dataset_split)


def _precompute_corpus(
    encoder, dataset, output_dir: Path
) -> None:
    encoder.precompute("corpus", str(output_dir), _corpus_iterator(dataset))


def _instantiate_encoder(name: str, params: Dict[str, object]):
    encoder_cls = get_encoder_class(name)
    return encoder_cls(**params)


@hydra_main(version_base=None, config_path="../../configs", config_name="prepare/slue_sqa5")
def main(cfg: DictConfig) -> None:
    dataset = load_dataset("asapp/slue-phase-2", "sqa5")

    output_root = Path(cfg.output_root).resolve()
    csv_dir = _ensure_dir(output_root / "csv")
    precompute_dir = _ensure_dir(output_root / "precomputed")

    # 1. Write CSV manifests and gather corpus text.
    doc_texts: Dict[str, str] = {}
    for split in _SPLITS:
        _write_split_csv(split, dataset[split], csv_dir, doc_texts)
    _write_corpus_csv(doc_texts, csv_dir)

    # 2. Precompute question encodings.
    question_cfg = OmegaConf.to_container(cfg.encoder.question, resolve=True) or {}
    document_cfg = OmegaConf.to_container(cfg.encoder.document, resolve=True) or {}

    question_params = question_cfg.get("params", {})
    document_params = document_cfg.get("params", {})

    question_encoder = _instantiate_encoder(cfg.encoder.name, question_params)
    document_encoder = _instantiate_encoder(cfg.encoder.name, document_params)

    for split in _SPLITS:
        split_dir = _ensure_dir(precompute_dir / split)
        _precompute_split(question_encoder, split, dataset[split], split_dir)
        question_encoder.clear_cache()

    corpus_dir = _ensure_dir(precompute_dir / "corpus")
    _precompute_corpus(document_encoder, dataset, corpus_dir)
    document_encoder.clear_cache()


if __name__ == "__main__":
    main()
