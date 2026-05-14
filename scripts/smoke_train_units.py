#!/usr/bin/env python3
"""
Run a tiny Trainer smoke test for packed-unit DSI and QG training.

This is intentionally not a real experiment. It uses a tiny randomly
initialized T5 config, a one- or two-example SLUE CSV fixture, and the same
dataset/collator/trainer code paths as the normal training scripts.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    T5Config,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from data import IndexingCollator, SlueSQA5DatasetV2  # noqa: E402
from hub_checkpoint import HubCheckpointArguments, build_hub_checkpoint_callbacks  # noqa: E402
from qg import QGTrainer, QueryGenDataset  # noqa: E402
from scripts.smoke_test_dataloaders import (  # noqa: E402
    write_csvs,
    write_csvs_from_store,
    write_packed_codes,
)
from t5_pretrain import DataCollatorForT5SpanCorruption, DiscreteCodeDataset  # noqa: E402
from trainer import DSITrainer  # noqa: E402


def make_tiny_t5(tokenizer, special_token: int) -> T5ForConditionalGeneration:
    vocab_size = max(len(tokenizer), tokenizer.vocab_size, special_token + 1)
    config = T5Config(
        vocab_size=vocab_size,
        d_model=32,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )
    return T5ForConditionalGeneration(config)


def prepare_fixture(root: Path, code_dir_arg: str | None, discrete_code_num: int):
    dataset_dir = root / "slue_sqa5"
    if code_dir_arg:
        code_dir = Path(code_dir_arg)
        lookup_path = write_csvs_from_store(dataset_dir, code_dir, discrete_code_num)
        return dataset_dir, code_dir, lookup_path, discrete_code_num

    code_dir = root / "slue_sqa_code_l22_c500"
    lookup_path = write_csvs(dataset_dir)
    write_packed_codes(code_dir)
    return dataset_dir, code_dir, lookup_path, 5


def make_training_args(args, output_dir: Path) -> TrainingArguments:
    use_hub_smoke = bool(args.hf_checkpoint_repo_id)
    training_kwargs = dict(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        logging_first_step=True,
        report_to=[],
        disable_tqdm=True,
        use_cpu=not args.cuda,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    if use_hub_smoke:
        training_kwargs.update(
            eval_strategy="steps",
            eval_steps=1,
            save_strategy="steps",
            save_steps=1,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_safetensors=True,
        )
    else:
        training_kwargs["save_strategy"] = "no"
    return TrainingArguments(**training_kwargs)


def run_dsi(
    args,
    dataset_dir: Path,
    code_dir: Path,
    lookup_path: Path,
    discrete_code_num: int,
):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = make_tiny_t5(tokenizer, args.special_token)
    train_ds = SlueSQA5DatasetV2(
        split="train",
        max_length=args.max_length,
        dataset_path=str(dataset_dir),
        code_path=str(code_dir),
        model_name_or_path=args.tokenizer_name,
        discrete_code_num=discrete_code_num,
        truncate_offset=args.truncate_offset,
        special_token=args.special_token,
        lookup_file_name=str(lookup_path),
    )
    trainer = DSITrainer(
        model=model,
        tokenizer=tokenizer,
        args=make_training_args(args, Path(args.output_dir) / "dsi"),
        train_dataset=train_ds,
        data_collator=IndexingCollator(tokenizer=tokenizer, padding="longest"),
        restrict_decode_vocab=None,
        id_max_length=16,
    )
    result = trainer.train()
    print(
        f"dsi smoke passed: global_step={trainer.state.global_step} "
        f"metrics={result.metrics}"
    )


def run_qg(
    args,
    dataset_dir: Path,
    code_dir: Path,
    lookup_path: Path,
    discrete_code_num: int,
):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = make_tiny_t5(tokenizer, args.special_token)
    train_ds = QueryGenDataset(
        split="train",
        max_length=args.max_length,
        dataset_path=str(dataset_dir),
        code_path=str(code_dir),
        discrete_code_num=discrete_code_num,
        special_token=args.special_token,
        lookup_file_name=str(lookup_path),
        label_max_length=args.label_max_length,
    )
    trainer = QGTrainer(
        model=model,
        args=make_training_args(args, Path(args.output_dir) / "qg"),
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
        ),
        generation_max_length=args.label_max_length + 2,
    )
    result = trainer.train()
    print(
        f"qg smoke passed: global_step={trainer.state.global_step} "
        f"metrics={result.metrics}"
    )


def run_hub_checkpoint_smoke(
    args,
    dataset_dir: Path,
    code_dir: Path,
    lookup_path: Path,
    discrete_code_num: int,
):
    if not args.hf_checkpoint_repo_id:
        raise ValueError("--mode hub requires --hf-checkpoint-repo-id")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = make_tiny_t5(tokenizer, args.special_token)
    train_ds = QueryGenDataset(
        split="train",
        max_length=args.max_length,
        dataset_path=str(dataset_dir),
        code_path=str(code_dir),
        discrete_code_num=discrete_code_num,
        special_token=args.special_token,
        lookup_file_name=str(lookup_path),
        label_max_length=args.label_max_length,
    )
    hub_args = HubCheckpointArguments(
        hf_checkpoint_repo_id=args.hf_checkpoint_repo_id,
        hf_checkpoint_private=args.hf_checkpoint_private,
        hf_checkpoint_mode=args.hf_checkpoint_mode,
        hf_checkpoint_latest_path=args.hf_checkpoint_latest_path,
        hf_checkpoint_best_path=args.hf_checkpoint_best_path,
        hf_checkpoint_prune_old=True,
        hf_checkpoint_fail_on_error=True,
    )
    trainer = Trainer(
        model=model,
        args=make_training_args(args, Path(args.output_dir) / "hub"),
        train_dataset=train_ds,
        eval_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
        ),
        callbacks=build_hub_checkpoint_callbacks(hub_args, tokenizer=tokenizer),
    )
    result = trainer.train()
    print(
        f"hub checkpoint smoke passed: repo={args.hf_checkpoint_repo_id} "
        f"global_step={trainer.state.global_step} metrics={result.metrics}"
    )


def run_t5_pretrain_smoke(
    args,
    code_dir: Path,
    discrete_code_num: int,
):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = make_tiny_t5(tokenizer, args.special_token)
    token_file = Path(args.output_dir) / "t5pt_unit_token_lookup.txt"
    train_ds = DiscreteCodeDataset(
        max_length=args.max_length,
        chunk_offset=args.truncate_offset,
        code_dir=str(code_dir),
        discrete_code_num=discrete_code_num,
        split="train",
        token_file=str(token_file),
        tokenizer_name_or_path=args.tokenizer_name,
        lookup_source_csv=None,
        validation_fraction=0.5,
        min_chunk_length=2,
        sentinel_start_id=32099,
        sentinel_direction=-1,
    )
    trainer = Trainer(
        model=model,
        args=make_training_args(args, Path(args.output_dir) / "t5pt"),
        train_dataset=train_ds,
        data_collator=DataCollatorForT5SpanCorruption(
            tokenizer=tokenizer,
            model=model,
            mask_prob=0.2,
            mean_span_length=3,
            sentinel_start_id=32099,
            sentinel_direction=-1,
        ),
    )
    result = trainer.train()
    print(
        f"t5 pretrain smoke passed: global_step={trainer.state.global_step} "
        f"metrics={result.metrics}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["dsi", "qg", "both", "hub", "t5pt"],
        default="both",
    )
    parser.add_argument("--code-dir", default=None, help="Packed unit directory to test.")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Fixture directory. Defaults to /tmp.",
    )
    parser.add_argument("--output-dir", default="/tmp/speechgr_unit_train_smoke")
    parser.add_argument("--tokenizer-name", default="google/flan-t5-small")
    parser.add_argument("--discrete-code-num", type=int, default=500)
    parser.add_argument("--special-token", type=int, default=32000)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--label-max-length", type=int, default=32)
    parser.add_argument("--truncate-offset", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available.")
    parser.add_argument("--keep", action="store_true", help="Keep generated fixture files.")
    parser.add_argument(
        "--hf-checkpoint-repo-id",
        default=None,
        help="Optional HF model repo id. Used by --mode hub to verify latest/best uploads.",
    )
    parser.add_argument(
        "--hf-checkpoint-private",
        action="store_true",
        help="Create the smoke HF model repo as private.",
    )
    parser.add_argument(
        "--hf-checkpoint-mode",
        choices=["model", "trainer"],
        default="model",
        help="Hub checkpoint upload mode for --mode hub.",
    )
    parser.add_argument("--hf-checkpoint-latest-path", default="latest")
    parser.add_argument("--hf-checkpoint-best-path", default="best")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.truncate_offset < args.max_length:
        raise ValueError(
            "--truncate-offset must be non-negative and smaller than --max-length"
        )

    os.environ.setdefault("WANDB_MODE", "disabled")
    root = Path(args.work_dir or tempfile.mkdtemp(prefix="speechgr_unit_train_smoke_"))
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    try:
        dataset_dir, code_dir, lookup_path, discrete_code_num = prepare_fixture(
            root,
            args.code_dir,
            args.discrete_code_num,
        )
        print(
            "fixture:",
            f"dataset_dir={dataset_dir}",
            f"code_dir={code_dir}",
            f"discrete_code_num={discrete_code_num}",
        )
        if args.mode in {"dsi", "both"}:
            run_dsi(args, dataset_dir, code_dir, lookup_path, discrete_code_num)
        if args.mode in {"qg", "both"}:
            run_qg(args, dataset_dir, code_dir, lookup_path, discrete_code_num)
        if args.mode == "hub":
            run_hub_checkpoint_smoke(
                args,
                dataset_dir,
                code_dir,
                lookup_path,
                discrete_code_num,
            )
        if args.mode == "t5pt":
            run_t5_pretrain_smoke(args, code_dir, discrete_code_num)
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
