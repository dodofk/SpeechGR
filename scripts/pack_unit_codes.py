#!/usr/bin/env python3
"""
Pack legacy .code/.cnt discrete-unit directories into the packed .npz layout.

Examples:
  python3 scripts/pack_unit_codes.py \
    --input_dir /path/to/slue_sqa_code_c512 \
    --mode slue \
    --dataset_path /path/to/slue_sqa5

  python3 scripts/pack_unit_codes.py \
    --input_dir /path/to/ll6k_code_l22_c500 \
    --mode librispeech
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Optional

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from unit_store import PackedUnitWriter


def read_int_file(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int32)
    if values.ndim == 0:
        return values.reshape(1)
    return values.reshape(-1)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def read_counts(code_path: Path, code_len: int) -> np.ndarray:
    count_path = code_path.with_suffix(".cnt")
    if count_path.exists():
        counts = read_int_file(count_path)
        if len(counts) == code_len:
            return counts
    return np.ones(code_len, dtype=np.int32)


def load_split_doc_ids(dataset_path: Optional[Path], split: str) -> dict[str, str]:
    if dataset_path is None:
        return {}
    csv_path = dataset_path / f"{split}.csv"
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        return {
            str(row["question_id"]): str(row["document_id"])
            for row in rows
            if "question_id" in row and "document_id" in row
        }


def pack_code_dir(
    code_dir: Path,
    output_path: Path,
    doc_ids: Optional[dict[str, str]] = None,
    compressed: bool = False,
) -> None:
    writer = PackedUnitWriter()
    code_files = sorted(code_dir.glob("*.code"))
    for idx, code_path in enumerate(code_files, start=1):
        if idx == 1 or idx % 1000 == 0 or idx == len(code_files):
            print(f"Packing {code_dir.name}: {idx}/{len(code_files)}")
        record_id = code_path.stem
        codes = read_int_file(code_path)
        writer.add(
            record_id,
            codes,
            read_counts(code_path, len(codes)),
            text=read_text(code_path.with_suffix(".trans.txt")),
            doc_id=(doc_ids or {}).get(record_id, ""),
        )
    writer.save(output_path, compressed=compressed)


def pack_slue(args) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir or args.input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path) if args.dataset_path else None

    document_dir = input_dir / "document_code"
    if document_dir.exists():
        pack_code_dir(
            document_dir,
            output_dir / "documents.npz",
            compressed=args.compressed,
        )

    for split in args.splits:
        split_dir = input_dir / f"{split}_code"
        if not split_dir.exists():
            continue
        pack_code_dir(
            split_dir,
            output_dir / f"{split}.npz",
            doc_ids=load_split_doc_ids(dataset_path, split),
            compressed=args.compressed,
        )


def pack_librispeech(args) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir or args.input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pack_code_dir(
        input_dir,
        output_dir / "librispeech.npz",
        compressed=args.compressed,
    )


def infer_mode(input_dir: Path) -> str:
    if (input_dir / "document_code").exists() or any(input_dir.glob("*_code")):
        return "slue"
    return "librispeech"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Legacy unit-code directory")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for packed archives. Defaults to input_dir.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "slue", "librispeech"],
        default="auto",
        help="Legacy layout to pack.",
    )
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Optional SLUE csv directory for question_id -> document_id metadata.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "validation", "test", "verified_test"],
        help="SLUE question splits to pack.",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Write compressed .npz archives. Smaller but slower.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    mode = infer_mode(input_dir) if args.mode == "auto" else args.mode
    if mode == "slue":
        pack_slue(args)
    else:
        pack_librispeech(args)


if __name__ == "__main__":
    main()
