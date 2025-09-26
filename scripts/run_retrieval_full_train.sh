#!/usr/bin/env bash

set -euo pipefail

# Full wavtokenizer retrieval fine-tuning run on CPU with atomic document ids
# and WandB logging.
#
# Usage:
#   bash scripts/run_retrieval_full_train.sh
#
# Environment variables (optional):
#   WANDB_ENTITY  - WandB entity (user or org)
#   WANDB_PROJECT - overrides project name (default: speechgr)
#   WANDB_RUN_NAME - explicit run name (defaults to Hydrated config value)

export CUDA_VISIBLE_DEVICES=""
unset DISABLE_WANDB

: "${WANDB_PROJECT:=speechgr}"

cmd=(
  uv run python -m speechgr.cli.train
  task.data.train_atomic=true
  task.logging.project="${WANDB_PROJECT}"
)

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  cmd+=(task.logging.entity="${WANDB_ENTITY}")
fi

if [[ -n "${WANDB_RUN_NAME:-}" ]]; then
  cmd+=(task.training.training_args.run_name="${WANDB_RUN_NAME}")
fi

"${cmd[@]}"
