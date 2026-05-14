#!/usr/bin/env python3
"""
Smoke-test the packed-unit training dataloaders with a tiny synthetic SLUE layout.

This validates the same code paths used by DSI training and QG training:
  - SlueSQA5DatasetV2 + IndexingCollator
  - QueryGenDataset + DataCollatorForSeq2Seq
  - packed .npz query/document stores

It does not train a model and does not require the full SLUE-SQA5 CSVs.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data import IndexingCollator, SlueSQA5DatasetV2  # noqa: E402
from qg import QueryGenDataset  # noqa: E402
from transformers import DataCollatorForSeq2Seq  # noqa: E402
from unit_store import PackedUnitStore, PackedUnitWriter  # noqa: E402


def write_csvs(dataset_dir: Path) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [
        {"question_id": "q1", "document_id": "docA", "passage": "alpha"},
        {"question_id": "q2", "document_id": "docB", "passage": "beta"},
    ]
    val_rows = [
        {"question_id": "q3", "document_id": "docA", "passage": "alpha"},
    ]
    pd.DataFrame(train_rows).to_csv(dataset_dir / "train.csv", index=False)
    pd.DataFrame(val_rows).to_csv(dataset_dir / "validation.csv", index=False)
    pd.DataFrame(val_rows).to_csv(dataset_dir / "test.csv", index=False)
    pd.DataFrame(val_rows).to_csv(dataset_dir / "verified_test.csv", index=False)
    pd.DataFrame(
        [
            {"document_id": "docA", "passage": "alpha document"},
            {"document_id": "docB", "passage": "beta document"},
        ]
    ).to_csv(dataset_dir / "slue_sqa5_corpus.csv", index=False)
    pd.DataFrame(
        [
            {"idx": "0", "document_id": "docA", "post_query": "what is alpha"},
            {"idx": "1", "document_id": "docB", "post_query": "what is beta"},
        ]
    ).to_csv(dataset_dir / "slue_sqa5_pq10_llama32_3b_clean.csv", index=False)

    lookup_path = dataset_dir / "flan-t5-base-unused_tokens.txt"
    np.savetxt(lookup_path, np.arange(100, 600, dtype=np.int32), fmt="%d")
    return lookup_path


def write_csvs_from_store(
    dataset_dir: Path,
    code_dir: Path,
    discrete_code_num: int,
) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_store = PackedUnitStore(code_dir / "train.npz")
    validation_store = PackedUnitStore(code_dir / "validation.npz")
    document_store = PackedUnitStore(code_dir / "documents.npz")

    doc_ids = document_store.ids[:2]
    train_qids = train_store.ids[:2]
    validation_qid = validation_store.ids[0]
    if len(doc_ids) < 2 or len(train_qids) < 2:
        raise AssertionError("Real packed store needs at least two train questions and documents")

    train_rows = [
        {"question_id": train_qids[0], "document_id": doc_ids[0], "passage": "alpha"},
        {"question_id": train_qids[1], "document_id": doc_ids[1], "passage": "beta"},
    ]
    val_rows = [
        {"question_id": validation_qid, "document_id": doc_ids[0], "passage": "alpha"},
    ]
    pd.DataFrame(train_rows).to_csv(dataset_dir / "train.csv", index=False)
    pd.DataFrame(val_rows).to_csv(dataset_dir / "validation.csv", index=False)
    pd.DataFrame(val_rows).to_csv(dataset_dir / "test.csv", index=False)
    pd.DataFrame(val_rows).to_csv(dataset_dir / "verified_test.csv", index=False)
    pd.DataFrame(
        [
            {"document_id": doc_ids[0], "passage": "alpha document"},
            {"document_id": doc_ids[1], "passage": "beta document"},
        ]
    ).to_csv(dataset_dir / "slue_sqa5_corpus.csv", index=False)
    pd.DataFrame(
        [
            {"idx": "0", "document_id": doc_ids[0], "post_query": "what is alpha"},
            {"idx": "1", "document_id": doc_ids[1], "post_query": "what is beta"},
        ]
    ).to_csv(dataset_dir / "slue_sqa5_pq10_llama32_3b_clean.csv", index=False)

    lookup_path = dataset_dir / "flan-t5-base-unused_tokens.txt"
    np.savetxt(
        lookup_path,
        np.arange(100, 100 + discrete_code_num, dtype=np.int32),
        fmt="%d",
    )
    return lookup_path


def write_packed_codes(code_dir: Path) -> None:
    code_dir.mkdir(parents=True, exist_ok=True)

    docs = PackedUnitWriter()
    docs.add("docA", np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int32))
    docs.add("docB", np.array([4, 3, 2, 1, 0, 4], dtype=np.int32))
    docs.save(code_dir / "documents.npz")

    train = PackedUnitWriter()
    train.add("q1", np.array([0, 1, 2, 3, 4, 0, 1, 2, 3], dtype=np.int32), doc_id="docA")
    train.add("q2", np.array([4, 3, 2], dtype=np.int32), doc_id="docB")
    train.save(code_dir / "train.npz")

    val = PackedUnitWriter()
    val.add("q3", np.array([0, 1, 2, 3, 4, 0], dtype=np.int32), doc_id="docA")
    for split in ["validation", "test", "verified_test"]:
        val.save(code_dir / f"{split}.npz")


def assert_tensor_batch(name: str, batch: dict, expected_keys: set[str]) -> None:
    missing = expected_keys - set(batch)
    if missing:
        raise AssertionError(f"{name} batch missing keys: {sorted(missing)}")
    for key in sorted(expected_keys):
        value = batch[key]
        shape = tuple(value.shape) if hasattr(value, "shape") else "<no-shape>"
        print(f"{name}.{key}: shape={shape}")


def run_smoke(
    root: Path,
    model_name: str,
    code_dir_arg: str | None,
    discrete_code_num: int,
) -> None:
    dataset_dir = root / "slue_sqa5"
    if code_dir_arg:
        code_dir = Path(code_dir_arg)
        lookup_path = write_csvs_from_store(dataset_dir, code_dir, discrete_code_num)
    else:
        code_dir = root / "slue_sqa_code_l22_c500"
        discrete_code_num = 5
        lookup_path = write_csvs(dataset_dir)
        write_packed_codes(code_dir)

    dsi_ds = SlueSQA5DatasetV2(
        split="train",
        max_length=8 if not code_dir_arg else 512,
        dataset_path=str(dataset_dir),
        code_path=str(code_dir),
        model_name_or_path=model_name,
        discrete_code_num=discrete_code_num,
        truncate_offset=2,
        lookup_file_name=str(lookup_path),
    )
    if dsi_ds.query_store is None or dsi_ds.document_store is None:
        raise AssertionError("DSI dataset did not open packed stores")
    query_item = dsi_ds[0]
    corpus_item = dsi_ds[dsi_ds.query_len]
    if len(query_item[0]) > dsi_ds.max_length:
        raise AssertionError(
            f"DSI query item length {len(query_item[0])} exceeds max_length={dsi_ds.max_length}"
        )
    if int(query_item[0][-1]) != 1:
        raise AssertionError("DSI query item should end with EOS token id 1")
    dsi_batch = IndexingCollator(tokenizer=dsi_ds.tokenizer)([query_item, corpus_item])
    assert_tensor_batch(
        "dsi",
        dsi_batch,
        {"input_ids", "attention_mask", "labels", "query_doc_id"},
    )

    val_ds = SlueSQA5DatasetV2(
        split="validation",
        max_length=8 if not code_dir_arg else 512,
        dataset_path=str(dataset_dir),
        code_path=str(code_dir),
        model_name_or_path=model_name,
        discrete_code_num=discrete_code_num,
        truncate_offset=2,
        lookup_file_name=str(lookup_path),
    )
    if len(val_ds) != val_ds.query_len:
        raise AssertionError("Validation DSI dataset should expose query rows only")

    qg_max_length = 8 if not code_dir_arg else 512
    qg_label_max_length = 4
    qg_ds = QueryGenDataset(
        split="train",
        max_length=qg_max_length,
        dataset_path=str(dataset_dir),
        code_path=str(code_dir),
        discrete_code_num=discrete_code_num,
        lookup_file_name=str(lookup_path),
        label_max_length=qg_label_max_length,
    )
    if qg_ds.query_store is None or qg_ds.document_store is None:
        raise AssertionError("QG dataset did not open packed stores")
    qg_item = qg_ds[0]
    if len(qg_item["input_ids"]) > qg_max_length:
        raise AssertionError("QG source length exceeds max_length")
    if len(qg_item["labels"]) > qg_label_max_length + 1:
        raise AssertionError("QG label length exceeds label_max_length + EOS")
    qg_batch = DataCollatorForSeq2Seq(tokenizer=dsi_ds.tokenizer)([qg_item, qg_ds[1]])
    assert_tensor_batch("qg", qg_batch, {"input_ids", "attention_mask", "labels"})

    print("dataloader smoke passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Temporary directory for generated CSV fixtures. Defaults to a unique /tmp directory.",
    )
    parser.add_argument(
        "--model-name",
        default="google/flan-t5-small",
        help="Tokenizer name used by the dataloaders.",
    )
    parser.add_argument(
        "--code-dir",
        default=None,
        help="Optional real packed unit directory with documents.npz/train.npz/validation.npz.",
    )
    parser.add_argument(
        "--discrete-code-num",
        type=int,
        default=500,
        help="Number of unit ids in the real packed store. Synthetic mode uses 5.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the generated smoke-test fixture.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.work_dir or tempfile.mkdtemp(prefix="speechgr_dataloader_smoke_"))
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        run_smoke(root, args.model_name, args.code_dir, args.discrete_code_num)
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
