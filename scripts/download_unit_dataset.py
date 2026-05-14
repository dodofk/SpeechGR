#!/usr/bin/env python3
"""
Download and verify the packed SLUE-SQA5 discrete-unit dataset.

The training code can download the dataset automatically from:
  hf://datasets/dodofk/slue-sqa-code-l22-c500

This script is a preflight helper for new machines or remote servers, so a long
training job does not fail late because the dataset/cache is unavailable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from unit_store import (  # noqa: E402
    DEFAULT_SLUE_UNIT_REPO_ID,
    PACKED_SLUE_PATTERNS,
    PackedUnitStore,
    download_packed_unit_dataset,
)


SPLIT_TO_FILE = {
    "documents": "documents.npz",
    "train": "train.npz",
    "validation": "validation.npz",
    "test": "test.npz",
    "verified_test": "verified_test.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_SLUE_UNIT_REPO_ID,
        help="Hugging Face dataset repo id or hf://datasets/... path.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["all"],
        choices=["all", *SPLIT_TO_FILE.keys()],
        help="Dataset files to download. Use verified_test for a quick smoke test.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hub branch, tag, or commit SHA.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory. HF_HOME also works.",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Optional normal output directory. Without this, files stay in HF cache.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if files already exist in cache.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Download only; skip PackedUnitStore readback checks.",
    )
    return parser.parse_args()


def selected_patterns(splits: list[str]) -> list[str]:
    if "all" in splits:
        return PACKED_SLUE_PATTERNS
    files = ["README.md"]
    files.extend(SPLIT_TO_FILE[split] for split in splits)
    return files


def verify_npz(root: Path, filename: str) -> None:
    path = root / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected downloaded file does not exist: {path}")

    store = PackedUnitStore(path)
    first_id = store.ids[0] if len(store) else "<empty>"
    first_len = len(store.get_code(first_id)) if len(store) else 0
    print(f"{filename}: records={len(store)} first_id={first_id} first_len={first_len}")


def main() -> None:
    args = parse_args()
    patterns = selected_patterns(args.splits)
    local_path = download_packed_unit_dataset(
        repo_id=args.repo_id,
        allow_patterns=patterns,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
        force_download=args.force_download,
    )
    root = Path(local_path)
    print(f"dataset_path={root}")

    if args.no_verify:
        return

    for filename in patterns:
        if filename.endswith(".npz"):
            verify_npz(root, filename)


if __name__ == "__main__":
    main()
