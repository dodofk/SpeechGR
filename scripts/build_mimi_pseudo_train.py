"""Build transcript-free pseudo-queries from Mimi corpus codes and augment train split."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_cache(path: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, dict):
        raise TypeError(f"Cache at '{path}' must be a mapping")
    return cache


def _symlink_or_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Expected source cache at '{src}'")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _span_score(span: torch.Tensor) -> float:
    if span.numel() == 0:
        return -1.0
    unique = torch.unique(span).numel()
    unique_ratio = unique / max(1, span.numel())
    counts = torch.bincount(span.long())
    probs = counts[counts > 0].float()
    probs = probs / probs.sum()
    entropy = float(-(probs * probs.log()).sum().item()) if probs.numel() > 0 else 0.0
    return unique_ratio + 0.1 * entropy


def _mine_spans(
    codes: torch.Tensor,
    *,
    spans_per_doc: int,
    min_span_tokens: int,
    max_span_tokens: int,
    max_attempts_per_span: int,
    rng: random.Random,
) -> list[Tuple[int, int, float]]:
    sequence = codes.long().reshape(-1)
    if sequence.numel() < min_span_tokens:
        return []

    spans: list[Tuple[int, int, float]] = []
    used_regions: list[Tuple[int, int]] = []
    for _ in range(spans_per_doc):
        best: Tuple[int, int, float] | None = None
        for _ in range(max_attempts_per_span):
            span_len = rng.randint(min_span_tokens, min(max_span_tokens, sequence.numel()))
            if span_len >= sequence.numel():
                start = 0
            else:
                start = rng.randint(0, sequence.numel() - span_len)
            end = start + span_len

            # Avoid reusing nearly the same region repeatedly.
            overlaps = False
            for prev_start, prev_end in used_regions:
                overlap = max(0, min(end, prev_end) - max(start, prev_start))
                if overlap / max(1, min(span_len, prev_end - prev_start)) > 0.7:
                    overlaps = True
                    break
            if overlaps:
                continue

            span = sequence[start:end]
            score = _span_score(span)
            if best is None or score > best[2]:
                best = (start, end, score)

        if best is None:
            continue
        spans.append(best)
        used_regions.append((best[0], best[1]))
    return spans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", required=True, help="Existing SLUE csv root")
    parser.add_argument("--precompute-root", required=True, help="Existing Mimi precompute root")
    parser.add_argument("--output-root", required=True, help="Augmented output root")
    parser.add_argument("--encoder-name", default="mimi")
    parser.add_argument("--spans-per-doc", type=int, default=3)
    parser.add_argument("--min-span-seconds", type=float, default=3.0)
    parser.add_argument("--max-span-seconds", type=float, default=8.0)
    parser.add_argument("--token-rate", type=float, default=12.5)
    parser.add_argument("--max-attempts-per-span", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_root = Path(args.csv_root).resolve()
    precompute_root = Path(args.precompute_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_csv = output_root / "csv"
    output_precompute = output_root / "precomputed"

    train_rows = _load_csv(csv_root / "train.csv")
    corpus_rows = _load_csv(csv_root / "corpus.csv")
    train_doc_ids = {str(row["document_id"]) for row in train_rows}
    existing_query_ids = {str(row["question_id"]) for row in train_rows}
    corpus_cache = _load_cache(precompute_root / "corpus" / f"corpus_{args.encoder_name}.pt")
    train_cache = _load_cache(precompute_root / "train" / f"train_{args.encoder_name}.pt")

    rng = random.Random(args.seed)
    min_span_tokens = max(1, int(round(args.min_span_seconds * args.token_rate)))
    max_span_tokens = max(min_span_tokens, int(round(args.max_span_seconds * args.token_rate)))

    pseudo_rows: list[dict[str, str]] = []
    pseudo_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    metadata: list[dict[str, object]] = []

    doc_text_lookup = {str(row["document_id"]): str(row.get("document_text", "")) for row in corpus_rows}

    for doc_id in sorted(train_doc_ids):
        entry = corpus_cache.get(doc_id)
        if entry is None or "codes" not in entry:
            continue
        codes = entry["codes"] if isinstance(entry["codes"], torch.Tensor) else torch.as_tensor(entry["codes"])
        spans = _mine_spans(
            codes,
            spans_per_doc=args.spans_per_doc,
            min_span_tokens=min_span_tokens,
            max_span_tokens=max_span_tokens,
            max_attempts_per_span=args.max_attempts_per_span,
            rng=rng,
        )
        for span_idx, (start, end, score) in enumerate(spans):
            pseudo_id = f"pq_{doc_id}_{span_idx:03d}"
            collision_idx = span_idx
            while pseudo_id in existing_query_ids or pseudo_id in pseudo_cache:
                collision_idx += 1
                pseudo_id = f"pq_{doc_id}_{collision_idx:03d}"
            pseudo_codes = codes.reshape(-1)[start:end].clone().long()
            pseudo_cache[pseudo_id] = {"codes": pseudo_codes}
            pseudo_rows.append(
                {
                    "question_id": pseudo_id,
                    "question_text": "",
                    "document_id": doc_id,
                    "document_text": doc_text_lookup.get(doc_id, ""),
                }
            )
            metadata.append(
                {
                    "question_id": pseudo_id,
                    "document_id": doc_id,
                    "start_token": start,
                    "end_token": end,
                    "score": score,
                }
            )

    # Write CSVs: copy existing files, but augment train.csv.
    output_csv.mkdir(parents=True, exist_ok=True)
    train_fieldnames = ["question_id", "question_text", "document_id", "document_text"]
    _write_csv(output_csv / "train.csv", train_rows + pseudo_rows, train_fieldnames)
    for split_name in ["validation.csv", "test.csv", "verified_test.csv", "corpus.csv"]:
        shutil.copy2(csv_root / split_name, output_csv / split_name)

    # Write precompute caches: augment train cache, link the rest.
    (output_precompute / "train").mkdir(parents=True, exist_ok=True)
    torch.save({**train_cache, **pseudo_cache}, output_precompute / "train" / f"train_{args.encoder_name}.pt")
    for split_name in ["validation", "test", "verified_test", "corpus"]:
        src = precompute_root / split_name / f"{split_name}_{args.encoder_name}.pt"
        if split_name == "corpus":
            src = precompute_root / "corpus" / f"corpus_{args.encoder_name}.pt"
            dst = output_precompute / "corpus" / f"corpus_{args.encoder_name}.pt"
        else:
            dst = output_precompute / split_name / f"{split_name}_{args.encoder_name}.pt"
        _symlink_or_copy(src, dst)

    metadata_path = output_root / "pseudo_queries.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as handle:
        for row in metadata:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "num_source_train_queries": len(train_rows),
        "num_pseudo_queries": len(pseudo_rows),
        "num_augmented_train_queries": len(train_rows) + len(pseudo_rows),
        "spans_per_doc": args.spans_per_doc,
        "min_span_tokens": min_span_tokens,
        "max_span_tokens": max_span_tokens,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
