import csv
from collections import defaultdict

import torch

from speechgr.data.slue_sqa5 import DiscreteUnitDataset


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_cache(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def test_discrete_dataset_chunking(tmp_path):
    csv_root = tmp_path / "csv"
    cache_root = tmp_path / "cache"

    train_rows = [
        {"question_id": "q1", "document_id": "d1"},
        {"question_id": "q2", "document_id": "d2"},
    ]
    val_rows = [
        {"question_id": "v1", "document_id": "d1"},
    ]
    corpus_rows = [
        {"document_id": "d1"},
        {"document_id": "d2"},
    ]

    _write_csv(csv_root / "train.csv", train_rows)
    _write_csv(csv_root / "validation.csv", val_rows)
    _write_csv(csv_root / "corpus.csv", corpus_rows)

    train_cache = {
        "q1": {"codes": torch.arange(0, 10, dtype=torch.long)},
        "q2": {"codes": torch.arange(10, 22, dtype=torch.long)},
    }
    val_cache = {
        "v1": {"codes": torch.arange(3, 14, dtype=torch.long)},
    }
    corpus_cache = {
        "d1": {"codes": torch.arange(0, 20, dtype=torch.long)},
        "d2": {"codes": torch.arange(20, 37, dtype=torch.long)},
    }

    _write_cache(cache_root / "train" / "train_dummy.pt", train_cache)
    _write_cache(cache_root / "validation" / "validation_dummy.pt", val_cache)
    _write_cache(cache_root / "corpus" / "corpus_dummy.pt", corpus_cache)

    dataset = DiscreteUnitDataset(
        split="train",
        dataset_path=str(csv_root),
        precompute_root=str(cache_root),
        encoder_name="dummy",
        include_corpus=True,
        max_length=8,
        query_max_length=8,
        corpus_max_length=8,
        corpus_chunk_size=6,
        corpus_chunk_stride=4,
        special_token=32000,
    )

    assert len(dataset) == dataset.query_len + 9  # 5 segments for d1, 4 for d2
    assert dataset.corpus_segments_per_doc["d1"] == 5
    assert dataset.corpus_segments_per_doc["d2"] == 4

    lengths_by_doc = defaultdict(list)
    for idx in range(dataset.query_len, len(dataset)):
        features, label, doc_idx = dataset[idx]
        assert features.dtype == torch.long
        assert len(features) <= 6
        lengths_by_doc[label].append(len(features))
        assert doc_idx in (0, 1)

    assert lengths_by_doc["d1"] == [6, 6, 6, 6, 4]
    assert lengths_by_doc["d2"] == [6, 6, 6, 5]

    atomic_dataset = DiscreteUnitDataset(
        split="train",
        dataset_path=str(csv_root),
        precompute_root=str(cache_root),
        encoder_name="dummy",
        include_corpus=True,
        max_length=8,
        query_max_length=8,
        corpus_max_length=8,
        train_atomic=True,
        corpus_chunk_size=6,
        corpus_chunk_stride=4,
        special_token=32000,
    )

    _, atomic_label, _ = atomic_dataset[atomic_dataset.query_len]
    assert isinstance(atomic_label, int)
    expected_offset = atomic_dataset._resolve_atomic_offset()
    assert expected_offset > 1
    assert atomic_label == expected_offset
    assert atomic_dataset.valid_ids[:2] == [str(expected_offset), str(expected_offset + 1)]

    val_dataset = DiscreteUnitDataset(
        split="validation",
        dataset_path=str(csv_root),
        precompute_root=str(cache_root),
        encoder_name="dummy",
        include_corpus=True,
        corpus_splits=("train", "validation"),
        special_token=32000,
    )

    assert val_dataset.include_corpus is True
    assert len(val_dataset) == val_dataset.query_len + len(val_dataset.doc_ids)


def test_discrete_dataset_query_corpus_max_length(tmp_path):
    csv_root = tmp_path / "csv"
    cache_root = tmp_path / "cache"

    _write_csv(
        csv_root / "train.csv",
        [{"question_id": "q1", "document_id": "d1"}],
    )
    _write_csv(csv_root / "corpus.csv", [{"document_id": "d1"}])

    query_codes = torch.arange(0, 12, dtype=torch.long)
    corpus_codes = torch.arange(100, 121, dtype=torch.long)

    _write_cache(cache_root / "train" / "train_dummy.pt", {"q1": {"codes": query_codes}})
    _write_cache(cache_root / "corpus" / "corpus_dummy.pt", {"d1": {"codes": corpus_codes}})

    dataset = DiscreteUnitDataset(
        split="train",
        dataset_path=str(csv_root),
        precompute_root=str(cache_root),
        encoder_name="dummy",
        include_corpus=True,
        max_length=None,
        query_max_length=4,
        corpus_max_length=5,
    )

    query_features, _, _ = dataset[0]
    assert query_features.shape[0] == 4

    # After query entries, remaining rows are corpus segments produced via auto chunking
    segments = [dataset[idx][0] for idx in range(dataset.query_len, len(dataset))]
    assert len(segments) == 5  # automatic chunking with max length 5
    assert all(seg.shape[0] <= 5 for seg in segments)
