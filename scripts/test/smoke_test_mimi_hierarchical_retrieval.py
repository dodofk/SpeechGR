"""CPU-friendly Stage 3 smoke test for Mimi hierarchical DocID retrieval."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, PreTrainedTokenizerFast, T5Config, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "omegaconf" not in sys.modules:
    omegaconf = types.ModuleType("omegaconf")

    class DictConfig(dict):
        pass

    class OmegaConf:
        @staticmethod
        def to_container(cfg, resolve=True):
            return dict(cfg)

        @staticmethod
        def create(obj):
            return obj

        @staticmethod
        def merge(*configs):
            merged = {}
            for config in configs:
                merged.update(dict(config))
            return merged

    omegaconf.DictConfig = DictConfig
    omegaconf.OmegaConf = OmegaConf
    sys.modules["omegaconf"] = omegaconf

from speechgr.data.collators import IndexingCollator
from speechgr.data.slue.discrete import DiscreteUnitDataset
from speechgr.model import DiscreteInputT5
from speechgr.trainer import DSITrainer
from speechgr.utils_legacy import RestrictDecodeVocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/smoke/mimi_hierarchical_retrieval",
        help="Directory used for generated fixtures, tiny model files, and trainer outputs.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_cache(path: Path, payload: dict[str, dict[str, torch.Tensor]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _build_tiny_tokenizer(model_dir: Path) -> None:
    vocab = {
        "<pad>": 0,
        "</s>": 1,
        "<unk>": 2,
        "<cl_000>": 3,
        "<cl_001>": 4,
        "<lf1_000>": 5,
        "<lf1_001>": 6,
        "<lf2_000>": 7,
        "<lf2_001>": 8,
        "<lf2_002>": 9,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    fast_tokenizer.save_pretrained(model_dir)


def _build_tiny_model(model_dir: Path) -> None:
    codebook_size = 8
    _build_tiny_tokenizer(model_dir)
    config = T5Config(
        vocab_size=10,
        d_model=64,
        d_ff=128,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_kv=32,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    model = DiscreteInputT5(
        config,
        discrete_vocab_size=codebook_size + 1,
        discrete_input_embedding_init="random_text",
    )
    model.initialize_discrete_input_embeddings("random_text")
    model.save_pretrained(model_dir)


def _materialize_smoke_assets(output_root: Path) -> dict[str, str]:
    csv_dir = output_root / "csv"
    precompute_dir = output_root / "precomputed"
    model_dir = output_root / "tiny_t5"
    train_output_dir = output_root / "train_output"
    docid_dir = output_root / "docids"

    _build_tiny_model(model_dir)

    _write_csv(
        csv_dir / "train.csv",
        [
            {"question_id": "q0", "document_id": "doc0"},
            {"question_id": "q1", "document_id": "doc1"},
            {"question_id": "q2", "document_id": "doc2"},
            {"question_id": "q3", "document_id": "doc1"},
        ],
    )
    _write_csv(
        csv_dir / "validation.csv",
        [
            {"question_id": "vq0", "document_id": "doc0"},
            {"question_id": "vq1", "document_id": "doc2"},
        ],
    )
    _write_csv(csv_dir / "test.csv", [{"question_id": "tq0", "document_id": "doc1"}])
    _write_csv(csv_dir / "verified_test.csv", [{"question_id": "vtq0", "document_id": "doc2"}])
    _write_csv(
        csv_dir / "corpus.csv",
        [
            {"document_id": "doc0"},
            {"document_id": "doc1"},
            {"document_id": "doc2"},
        ],
    )

    _write_cache(
        precompute_dir / "train" / "train_mimi.pt",
        {
            "q0": {"codes": torch.tensor([0, 1, 2, 1], dtype=torch.long)},
            "q1": {"codes": torch.tensor([3, 4, 3, 4], dtype=torch.long)},
            "q2": {"codes": torch.tensor([5, 6, 5, 6], dtype=torch.long)},
            "q3": {"codes": torch.tensor([3, 4, 4, 3], dtype=torch.long)},
        },
    )
    _write_cache(
        precompute_dir / "validation" / "validation_mimi.pt",
        {
            "vq0": {"codes": torch.tensor([0, 1, 2], dtype=torch.long)},
            "vq1": {"codes": torch.tensor([5, 6, 6], dtype=torch.long)},
        },
    )
    _write_cache(
        precompute_dir / "test" / "test_mimi.pt",
        {"tq0": {"codes": torch.tensor([3, 4, 3], dtype=torch.long)}},
    )
    _write_cache(
        precompute_dir / "verified_test" / "verified_test_mimi.pt",
        {"vtq0": {"codes": torch.tensor([5, 6, 5], dtype=torch.long)}},
    )
    _write_cache(
        precompute_dir / "corpus" / "corpus_mimi.pt",
        {
            "doc0": {"codes": torch.tensor([0, 1, 2, 1, 0, 1], dtype=torch.long)},
            "doc1": {"codes": torch.tensor([3, 4, 3, 4, 3, 4], dtype=torch.long)},
            "doc2": {"codes": torch.tensor([5, 6, 5, 6, 5, 6], dtype=torch.long)},
        },
    )

    docid_dir.mkdir(parents=True, exist_ok=True)
    docid_map = {
        "doc0": {
            "docid": "<cl_000> <lf1_000> <lf2_000>",
            "tokens": ["<cl_000>", "<lf1_000>", "<lf2_000>"],
        },
        "doc1": {
            "docid": "<cl_000> <lf1_000> <lf2_001>",
            "tokens": ["<cl_000>", "<lf1_000>", "<lf2_001>"],
        },
        "doc2": {
            "docid": "<cl_001> <lf1_001> <lf2_002>",
            "tokens": ["<cl_001>", "<lf1_001>", "<lf2_002>"],
        },
    }
    cluster_members = {
        "<cl_000>": ["doc0", "doc1"],
        "<cl_001>": ["doc2"],
    }
    valid_paths = [
        ["<cl_000>", "<lf1_000>", "<lf2_000>"],
        ["<cl_000>", "<lf1_000>", "<lf2_001>"],
        ["<cl_001>", "<lf1_001>", "<lf2_002>"],
    ]
    (docid_dir / "docid_map.json").write_text(json.dumps(docid_map, indent=2))
    (docid_dir / "cluster_members.json").write_text(json.dumps(cluster_members, indent=2))
    (docid_dir / "valid_paths.json").write_text(json.dumps(valid_paths, indent=2))

    return {
        "csv_dir": str(csv_dir.resolve()),
        "precompute_dir": str(precompute_dir.resolve()),
        "model_dir": str(model_dir.resolve()),
        "train_output_dir": str(train_output_dir.resolve()),
        "docid_map_path": str((docid_dir / "docid_map.json").resolve()),
        "valid_paths_path": str((docid_dir / "valid_paths.json").resolve()),
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    assets = _materialize_smoke_assets(output_root)
    codebook_size = 8
    special_token = codebook_size
    discrete_vocab_size = special_token + 1

    tokenizer = AutoTokenizer.from_pretrained(assets["model_dir"])
    train_dataset = DiscreteUnitDataset(
        split="train",
        dataset_path=assets["csv_dir"],
        precompute_root=assets["precompute_dir"],
        encoder_name="mimi",
        include_corpus=True,
        docid_map_path=assets["docid_map_path"],
        query_max_length=8,
        corpus_max_length=8,
        corpus_chunk_size=8,
        corpus_chunk_stride=8,
        special_token=special_token,
    )
    valid_dataset = DiscreteUnitDataset(
        split="validation",
        dataset_path=assets["csv_dir"],
        precompute_root=assets["precompute_dir"],
        encoder_name="mimi",
        include_corpus=True,
        docid_map_path=assets["docid_map_path"],
        query_max_length=8,
        corpus_max_length=8,
        corpus_chunk_size=8,
        corpus_chunk_stride=8,
        special_token=special_token,
    )

    model = DiscreteInputT5.from_pretrained(
        assets["model_dir"],
        discrete_vocab_size=discrete_vocab_size,
        discrete_input_embedding_init="random_text",
    )
    model.initialize_discrete_input_embeddings("random_text")

    training_args = TrainingArguments(
        output_dir=assets["train_output_dir"],
        num_train_epochs=args.epochs,
        max_steps=8,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=0.01,
        warmup_steps=0,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        disable_tqdm=True,
    )

    trainer = DSITrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=IndexingCollator(tokenizer=tokenizer, padding="longest"),
        processing_class=tokenizer,
        compute_metrics=None,
        restrict_decode_vocab=RestrictDecodeVocab(train_dataset.valid_ids, tokenizer),
        id_max_length=4,
    )
    trainer.max_length = 8
    trainer.num_return_sequences = 2
    trainer.top_k = 2
    trainer.generation_max_length = 4

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    summary = {
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "artifacts": assets,
        "output_root": str(output_root),
        "epochs": args.epochs,
        "codebook_size": codebook_size,
        "special_token": special_token,
        "discrete_vocab_size": discrete_vocab_size,
        "valid_ids": train_dataset.valid_ids,
    }
    summary_path = output_root / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Hierarchical smoke test succeeded. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
