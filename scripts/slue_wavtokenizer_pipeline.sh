#!/usr/bin/env bash
set -euo pipefail

# SLUE SQA5 pipeline using WavTokenizer discrete units.
# Update CONFIG_PATH and OUTPUT_ROOT to match your environment.

OUTPUT_ROOT="outputs/slue_wavtok"
UV_PYTHON_VERSION="3.12"

# 1. Generate CSV manifests and precompute discrete units (uses GPU by default).
UV_PYTHON=$UV_PYTHON_VERSION uv run python -m speechgr.cli.prepare_slue \
  output_root="$OUTPUT_ROOT"

# 2. Train retrieval on the cached discrete units.
UV_PYTHON=$UV_PYTHON_VERSION uv run python -m speechgr.cli.train \
  task=retrieval \
  data.modality=discrete_precomputed \
  data.dataset_path="$OUTPUT_ROOT/csv" \
  data.precompute_root="$OUTPUT_ROOT/precomputed" \
  data.encoder_name=wavtokenizer \
  data.train_atomic=true \
  model.model_name=google/flan-t5-base \
  training.training_args.output_dir="$OUTPUT_ROOT/models" \
  training.training_args.per_device_train_batch_size=4 \
  training.training_args.per_device_eval_batch_size=4 \
  training.training_args.fp16=true \
  training.training_args.max_steps=1000
