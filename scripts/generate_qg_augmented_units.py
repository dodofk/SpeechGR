#!/usr/bin/env python3
"""
Generate QG pseudo-query unit codes and build DSI-compatible augmented data.

The DSI loader expects:
  - metadata/<split>.csv with question_id and document_id
  - code_dir/<split>.npz with raw unit ids keyed by question_id
  - code_dir/documents.npz with raw document unit ids

QG checkpoints generate tokenizer ids, so this script converts generated unit
token ids back to raw unit ids before writing the augmented train.npz.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

sys.path.append(str(Path(__file__).resolve().parents[1]))
from unit_store import (  # noqa: E402
    PACKED_SLUE_PATTERNS,
    PackedUnitWriter,
    load_packed_store,
    resolve_unit_code_path,
)
from unit_token_lookup import (  # noqa: E402
    build_lookup_from_csv,
    load_token_lookup,
    lookup_from_model_config,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate QG pseudo queries and write augmented DSI data."
    )
    parser.add_argument("--qg_model_path", required=True)
    parser.add_argument("--model_name_or_path", default="google/flan-t5-base")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--code_path", required=True)
    parser.add_argument("--output_dataset_path", required=True)
    parser.add_argument("--output_code_path", required=True)
    parser.add_argument("--pq_filename", default="slue_sqa5_pq10_llama32_3b_clean.csv")
    parser.add_argument("--corpus_filename", default="slue_sqa5_corpus.csv")
    parser.add_argument(
        "--document_source",
        choices=["corpus", "train"],
        default="corpus",
        help="Generate from all corpus docs or only docs appearing in train.csv.",
    )
    parser.add_argument("--discrete_code_num", type=int, default=500)
    parser.add_argument("--lookup_file_name", default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--offset", type=int, default=30)
    parser.add_argument("--generation_max_length", type=int, default=302)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_documents", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.do_sample and args.num_return_sequences > args.num_beams:
        raise ValueError(
            "num_return_sequences must be <= num_beams when --do_sample is not set"
        )
    if args.max_length < 2:
        raise ValueError(f"max_length must be at least 2, got {args.max_length}")
    if not 0 <= args.offset < args.max_length:
        raise ValueError(
            f"offset must be non-negative and smaller than max_length, got {args.offset}"
        )
    return args


def copy_metadata(dataset_path: Path, output_dataset_path: Path) -> None:
    output_dataset_path.mkdir(parents=True, exist_ok=True)
    for path in dataset_path.iterdir():
        if path.is_file() and path.suffix == ".csv":
            shutil.copy2(path, output_dataset_path / path.name)


def copy_non_train_code_files(base_code_path: Path, output_code_path: Path) -> None:
    output_code_path.mkdir(parents=True, exist_ok=True)
    for name in ["documents.npz", "validation.npz", "test.npz", "verified_test.npz", "README.md"]:
        src = base_code_path / name
        if src.exists():
            shutil.copy2(src, output_code_path / name)


def load_lookup(args: argparse.Namespace, model) -> np.ndarray:
    if args.lookup_file_name:
        lookup = load_token_lookup(
            args.lookup_file_name,
            discrete_code_num=args.discrete_code_num,
        )
    else:
        lookup = lookup_from_model_config(
            model.config,
            discrete_code_num=args.discrete_code_num,
        )
        if lookup is None:
            lookup = build_lookup_from_csv(
                csv_path=str(Path(args.dataset_path) / args.pq_filename),
                tokenizer_name_or_path=args.model_name_or_path,
                discrete_code_num=args.discrete_code_num,
            )
    lookup = np.asarray(lookup, dtype=np.int64)[: args.discrete_code_num]
    if len(lookup) < args.discrete_code_num:
        raise ValueError(
            f"Lookup has {len(lookup)} entries but discrete_code_num={args.discrete_code_num}"
        )
    return lookup


def map_units_to_tokens(raw_units: np.ndarray, unit_to_token: dict[int, int], item_id: str) -> np.ndarray:
    raw_units = np.asarray(raw_units, dtype=int).reshape(-1)
    missing = sorted(set(raw_units.tolist()) - set(unit_to_token))
    if missing:
        raise ValueError(f"{item_id} has unit ids outside lookup: {missing[:10]}")
    return np.asarray([unit_to_token[int(unit)] for unit in raw_units], dtype=np.int64)


def iter_document_chunks(
    document_ids: Iterable[str],
    document_store,
    unit_to_token: dict[int, int],
    max_length: int,
    offset: int,
):
    step = max(1, max_length - offset)
    for doc_id in document_ids:
        raw_code = document_store.get_code(doc_id)
        token_code = map_units_to_tokens(raw_code, unit_to_token, doc_id)
        chunk_idx = 0
        cur_idx = 0
        while cur_idx < len(token_code):
            end_idx = min(cur_idx + max_length - 1, len(token_code))
            input_ids = np.concatenate([token_code[cur_idx:end_idx], [1]])
            yield str(doc_id), chunk_idx, input_ids
            cur_idx += step
            chunk_idx += 1


def decode_generated_units(
    generated_ids: np.ndarray,
    token_to_unit: dict[int, int],
    ignored_token_ids: set[int],
) -> np.ndarray:
    units: list[int] = []
    for token_id in generated_ids.tolist():
        token_id = int(token_id)
        if token_id in ignored_token_ids:
            continue
        if token_id in token_to_unit:
            units.append(token_to_unit[token_id])
    return np.asarray(units, dtype=np.int32)


def select_document_ids(args: argparse.Namespace, dataset_path: Path, document_store) -> list[str]:
    if args.document_source == "train":
        df = pd.read_csv(dataset_path / "train.csv")
        document_ids = df["document_id"].astype(str).drop_duplicates().tolist()
    else:
        df = pd.read_csv(dataset_path / args.corpus_filename)
        document_ids = df["document_id"].astype(str).drop_duplicates().tolist()

    available = set(document_store.ids)
    missing = [doc_id for doc_id in document_ids if doc_id not in available]
    if missing:
        raise ValueError(f"{len(missing)} selected documents are missing codes, e.g. {missing[:5]}")
    if args.max_documents is not None:
        document_ids = document_ids[: args.max_documents]
    return document_ids


def append_original_train_codes(writer: PackedUnitWriter, train_store) -> None:
    for record_id in tqdm(train_store.ids, desc="Copy original train query codes"):
        writer.add(
            record_id,
            train_store.get_code(record_id),
            train_store.get_counts(record_id),
            text=train_store.get_text(record_id),
            doc_id=train_store.get_doc_id(record_id),
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    dataset_path = Path(args.dataset_path)
    output_dataset_path = Path(args.output_dataset_path)
    output_code_path = Path(args.output_code_path)
    if (output_dataset_path.exists() or output_code_path.exists()) and not args.overwrite:
        raise FileExistsError("Output paths exist. Pass --overwrite to replace/update them.")

    base_code_path = Path(
        resolve_unit_code_path(args.code_path, allow_patterns=PACKED_SLUE_PATTERNS)
    )
    document_store = load_packed_store(base_code_path / "documents.npz")
    train_store = load_packed_store(base_code_path / "train.npz")
    if document_store is None or train_store is None:
        raise FileNotFoundError(f"Missing documents.npz or train.npz under {base_code_path}")

    model_kwargs = {}
    if args.bf16 and args.device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = T5ForConditionalGeneration.from_pretrained(args.qg_model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    lookup = load_lookup(args, model)
    unit_to_token = {idx: int(token_id) for idx, token_id in enumerate(lookup)}
    token_to_unit = {int(token_id): idx for idx, token_id in enumerate(lookup)}
    ignored_token_ids = {
        token_id
        for token_id in [tokenizer.pad_token_id, tokenizer.eos_token_id, model.config.decoder_start_token_id]
        if token_id is not None
    }

    document_ids = select_document_ids(args, dataset_path, document_store)
    logger.info("Generating pseudo queries for %d documents", len(document_ids))

    copy_metadata(dataset_path, output_dataset_path)
    copy_non_train_code_files(base_code_path, output_code_path)

    augmented_train = pd.read_csv(dataset_path / "train.csv")
    train_columns = list(augmented_train.columns)
    pseudo_rows = []
    writer = PackedUnitWriter()
    append_original_train_codes(writer, train_store)

    chunk_iter = list(
        iter_document_chunks(
            document_ids,
            document_store,
            unit_to_token,
            max_length=args.max_length,
            offset=args.offset,
        )
    )
    dropped_empty = 0
    generated_count = 0
    batch = []
    batch_meta = []

    def flush_batch() -> None:
        nonlocal dropped_empty, generated_count, batch, batch_meta
        if not batch:
            return
        max_len = max(len(x) for x in batch)
        pad_id = tokenizer.pad_token_id
        input_ids = np.full((len(batch), max_len), pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(batch), max_len), dtype=np.int64)
        for row_idx, ids in enumerate(batch):
            input_ids[row_idx, : len(ids)] = ids
            attention_mask[row_idx, : len(ids)] = 1
        input_ids_t = torch.tensor(input_ids, device=args.device)
        attention_mask_t = torch.tensor(attention_mask, device=args.device)
        generation_kwargs = {
            "input_ids": input_ids_t,
            "attention_mask": attention_mask_t,
            "max_length": args.generation_max_length,
            "num_return_sequences": args.num_return_sequences,
        }
        if args.do_sample:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "top_p": args.top_p,
                    "temperature": args.temperature,
                }
            )
        else:
            generation_kwargs["num_beams"] = args.num_beams
        with torch.no_grad():
            outputs = model.generate(**generation_kwargs).detach().cpu().numpy()

        repeated_meta = [
            meta
            for meta in batch_meta
            for _ in range(args.num_return_sequences)
        ]
        for generated_idx, (doc_id, chunk_idx, output_ids) in enumerate(
            zip(
                [m[0] for m in repeated_meta],
                [m[1] for m in repeated_meta],
                outputs,
            )
        ):
            raw_units = decode_generated_units(output_ids, token_to_unit, ignored_token_ids)
            if len(raw_units) == 0:
                dropped_empty += 1
                continue
            pseudo_id = f"qg_{doc_id}_c{chunk_idx}_n{generated_idx % args.num_return_sequences}"
            writer.add(pseudo_id, raw_units, doc_id=doc_id)
            row = {column: "" for column in train_columns}
            row["question_id"] = pseudo_id
            row["document_id"] = doc_id
            if "post_query" in row:
                row["post_query"] = pseudo_id
            pseudo_rows.append(row)
            generated_count += 1
        batch = []
        batch_meta = []

    for doc_id, chunk_idx, input_ids in tqdm(chunk_iter, desc="Generate pseudo query units"):
        batch.append(input_ids)
        batch_meta.append((doc_id, chunk_idx))
        if len(batch) >= args.batch_size:
            flush_batch()
    flush_batch()

    if pseudo_rows:
        augmented_train = pd.concat(
            [augmented_train, pd.DataFrame(pseudo_rows, columns=train_columns)],
            ignore_index=True,
        )
    augmented_train.to_csv(output_dataset_path / "train.csv", index=False)
    writer.save(output_code_path / "train.npz", compressed=args.compressed)

    summary = {
        "qg_model_path": args.qg_model_path,
        "base_code_path": str(base_code_path),
        "output_dataset_path": str(output_dataset_path),
        "output_code_path": str(output_code_path),
        "document_source": args.document_source,
        "documents": len(document_ids),
        "document_chunks": len(chunk_iter),
        "original_train_queries": len(train_store),
        "pseudo_queries": generated_count,
        "dropped_empty": dropped_empty,
        "num_return_sequences": args.num_return_sequences,
    }
    (output_dataset_path / "qg_augmentation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %s", output_dataset_path / "qg_augmentation_summary.json")
    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
