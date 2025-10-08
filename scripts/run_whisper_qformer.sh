#!/usr/bin/env bash

# Smoke + full-run helper for Whisper-feature Q-Former training.
#
# Usage:
#   chmod +x scripts/run_whisper_qformer.sh
#   ./scripts/run_whisper_qformer.sh \
#       CSV_ROOT=/mnt/slue_sqa5_whisper/csv \
#       CACHE_ROOT=/mnt/slue_sqa5_whisper/precomputed \
#       DEVICE=cuda TRAIN_BS=4 EVAL_BS=4 GRAD_ACC=2 MAX_STEPS=50000
#
# Override defaults by exporting the variables above or by passing them inline
# (VAR=value ./scripts/run_whisper_qformer.sh). The script will:
#   1. Materialize SLUE-SQA5 CSV manifests and Whisper caches via Hydra preset.
#   2. Launch Q-Former training using the cached Whisper features.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables or inline VAR=value pairs)
# ---------------------------------------------------------------------------
CSV_ROOT=${CSV_ROOT:-/abs/path/to/slue_sqa5_whisper/csv}
CACHE_ROOT=${CACHE_ROOT:-/abs/path/to/slue_sqa5_whisper/precomputed}
DEVICE=${DEVICE:-cuda}
TRAIN_BS=${TRAIN_BS:-4}
EVAL_BS=${EVAL_BS:-4}
GRAD_ACC=${GRAD_ACC:-2}
MAX_STEPS=${MAX_STEPS:-250000}
RUN_NAME=${RUN_NAME:-slue_sqa5-whisper-full}

# Derive prepare_slue output root from CSV directory (parent directory).
OUTPUT_ROOT=$(dirname "$CSV_ROOT")

echo "[run_whisper_qformer] Writing manifes ts & caches to $OUTPUT_ROOT"

uv run python -m speechgr.cli.prepare_slue \
  --config-name=slue_sqa5_whisper \
  output_root="$OUTPUT_ROOT" \
  encoder.params.device="$DEVICE"

echo "[run_whisper_qformer] Launching Q-Former training"

uv run python -m speechgr.cli.qformer \
  +experiment=qformer/slue_sqa5_whisper \
  data.dataset_path="$CSV_ROOT" \
  data.precompute_root="$CACHE_ROOT" \
  model.device="$DEVICE" \
  training.training_args.per_device_train_batch_size="$TRAIN_BS" \
  training.training_args.per_device_eval_batch_size="$EVAL_BS" \
  training.training_args.gradient_accumulation_steps="$GRAD_ACC" \
  training.training_args.max_steps="$MAX_STEPS" \
  training.training_args.run_name="$RUN_NAME"

echo "[run_whisper_qformer] Done"
