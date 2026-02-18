#!/usr/bin/env bash

set -euo pipefail

# Simple CPU-only smoke test for the wavtokenizer retrieval setup.
#
# Usage:
#   bash scripts/run_retrieval_smoke_cpu.sh
#
# Requirements:
#   - uv-installed environment with project dependencies (uv sync)
#   - wavtokenizer CSV + precompute under outputs/slue_wavtok/{csv,precomputed}

export CUDA_VISIBLE_DEVICES="3"
export DISABLE_WANDB=1

uv run python -m speechgr.cli.train \
  task.data.train_atomic=true \
  task.training.training_args.max_steps=1 \
  task.training.training_args.per_device_train_batch_size=1 \
  task.training.training_args.gradient_accumulation_steps=1 \
  task.training.training_args.dataloader_num_workers=0 \
  task.training.training_args.evaluation_strategy=no \
  task.training.training_args.save_strategy=no \
  task.training.training_args.logging_steps=1 \
  task.training.training_args.output_dir=outputs/debug_retrieval \
  task.training.training_args.run_name=debug_retrieval_cpu_smoke \
  task.training.training_args.report_to=[]
