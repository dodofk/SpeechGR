#!/usr/bin/env bash
set -euo pipefail

# SLUE SQA5 pipeline using WavTokenizer discrete units.
# Update CONFIG_PATH, MODEL_PATH, and OUTPUT_ROOT to match your environment.

CONFIG_PATH="inventory/WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
MODEL_PATH="inventory/WavTokenizer/wavtokenizer_large_speech_320_v2.ckpt"
OUTPUT_ROOT="outputs/slue_wavtok"

CSV_ROOT="$OUTPUT_ROOT/csv"
CACHE_ROOT="$OUTPUT_ROOT/precomputed"

# 1. Generate CSV manifests and precompute discrete units (uses GPU by default).
uv run python -m speechgr.cli.prepare_slue \
  encoder.name=wavtokenizer \
  encoder.question.params.config_path="$CONFIG_PATH" \
  encoder.question.params.model_path="$MODEL_PATH" \
  encoder.question.params.device=cuda \
  encoder.document.params.config_path="$CONFIG_PATH" \
  encoder.document.params.model_path="$MODEL_PATH" \
  encoder.document.params.device=cuda \
  output_root="$OUTPUT_ROOT"

# 2. Train retrieval on the cached discrete units.
uv run python -m speechgr.cli.train \
  task=retrieval \
  data.modality=discrete_precomputed \
  data.dataset_path="$CSV_ROOT" \
  data.precompute_root="$CACHE_ROOT" \
  data.encoder_name=wavtokenizer \
  data.train_atomic=true \
  model.model_name=google/flan-t5-base \
  training.training_args.output_dir="$OUTPUT_ROOT/models" \
  training.training_args.per_device_train_batch_size=4 \
  training.training_args.per_device_eval_batch_size=4 \
  training.training_args.fp16=true \
  training.training_args.max_steps=1000
