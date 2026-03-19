import csv
import json
import subprocess
import sys
from pathlib import Path

import torch


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_mimi_pseudo_train_creates_augmented_train(tmp_path):
    csv_root = tmp_path / "csv"
    precompute_root = tmp_path / "precomputed"
    output_root = tmp_path / "aug"

    _write_csv(
        csv_root / "train.csv",
        [{"question_id": "q1", "question_text": "", "document_id": "d1", "document_text": ""}],
    )
    _write_csv(
        csv_root / "validation.csv",
        [{"question_id": "v1", "question_text": "", "document_id": "d1", "document_text": ""}],
    )
    _write_csv(
        csv_root / "test.csv",
        [{"question_id": "t1", "question_text": "", "document_id": "d1", "document_text": ""}],
    )
    _write_csv(
        csv_root / "verified_test.csv",
        [{"question_id": "vt1", "question_text": "", "document_id": "d1", "document_text": ""}],
    )
    _write_csv(
        csv_root / "corpus.csv",
        [{"document_id": "d1", "document_text": ""}],
    )

    (precompute_root / "train").mkdir(parents=True, exist_ok=True)
    torch.save({"q1": {"codes": torch.tensor([1, 2, 3], dtype=torch.long)}}, precompute_root / "train" / "train_mimi.pt")
    for split in ["validation", "test", "verified_test"]:
        (precompute_root / split).mkdir(parents=True, exist_ok=True)
        torch.save({}, precompute_root / split / f"{split}_mimi.pt")
    (precompute_root / "corpus").mkdir(parents=True, exist_ok=True)
    torch.save({"d1": {"codes": torch.tensor(list(range(80)), dtype=torch.long)}}, precompute_root / "corpus" / "corpus_mimi.pt")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_mimi_pseudo_train.py",
            "--csv-root",
            str(csv_root),
            "--precompute-root",
            str(precompute_root),
            "--output-root",
            str(output_root),
            "--spans-per-doc",
            "2",
            "--min-span-seconds",
            "1",
            "--max-span-seconds",
            "2",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((output_root / "summary.json").read_text())
    assert summary["num_pseudo_queries"] == 2

    with (output_root / "csv" / "train.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert any(row["question_id"].startswith("pq_d1_") for row in rows)

    augmented_train = torch.load(output_root / "precomputed" / "train" / "train_mimi.pt", map_location="cpu")
    assert any(key.startswith("pq_d1_") for key in augmented_train)
    assert "num_pseudo_queries" in result.stdout


def test_build_mimi_pseudo_train_avoids_existing_query_id_collision(tmp_path):
    csv_root = tmp_path / "csv"
    precompute_root = tmp_path / "precomputed"
    output_root = tmp_path / "aug"

    _write_csv(
        csv_root / "train.csv",
        [
            {"question_id": "q1", "question_text": "", "document_id": "d1", "document_text": ""},
            {"question_id": "pq_d1_000", "question_text": "", "document_id": "d1", "document_text": ""},
        ],
    )
    _write_csv(csv_root / "validation.csv", [{"question_id": "v1", "question_text": "", "document_id": "d1", "document_text": ""}])
    _write_csv(csv_root / "test.csv", [{"question_id": "t1", "question_text": "", "document_id": "d1", "document_text": ""}])
    _write_csv(csv_root / "verified_test.csv", [{"question_id": "vt1", "question_text": "", "document_id": "d1", "document_text": ""}])
    _write_csv(csv_root / "corpus.csv", [{"document_id": "d1", "document_text": ""}])

    (precompute_root / "train").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "q1": {"codes": torch.tensor([1, 2, 3], dtype=torch.long)},
            "pq_d1_000": {"codes": torch.tensor([4, 5, 6], dtype=torch.long)},
        },
        precompute_root / "train" / "train_mimi.pt",
    )
    for split in ["validation", "test", "verified_test"]:
        (precompute_root / split).mkdir(parents=True, exist_ok=True)
        torch.save({}, precompute_root / split / f"{split}_mimi.pt")
    (precompute_root / "corpus").mkdir(parents=True, exist_ok=True)
    torch.save({"d1": {"codes": torch.tensor(list(range(80)), dtype=torch.long)}}, precompute_root / "corpus" / "corpus_mimi.pt")

    subprocess.run(
        [
            sys.executable,
            "scripts/build_mimi_pseudo_train.py",
            "--csv-root",
            str(csv_root),
            "--precompute-root",
            str(precompute_root),
            "--output-root",
            str(output_root),
            "--spans-per-doc",
            "1",
            "--min-span-seconds",
            "1",
            "--max-span-seconds",
            "2",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        check=True,
        capture_output=True,
        text=True,
    )

    augmented_train = torch.load(output_root / "precomputed" / "train" / "train_mimi.pt", map_location="cpu")
    assert "pq_d1_000" in augmented_train
    assert any(key.startswith("pq_d1_") and key != "pq_d1_000" for key in augmented_train)
