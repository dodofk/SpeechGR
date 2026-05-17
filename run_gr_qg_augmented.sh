#!/usr/bin/env bash
set -euo pipefail

export DATASET_PATH="${DATASET_PATH:-data/slue_sqa5_qg_aug_unitpt50k_ckpt10000}"
export CODE_DIR="${CODE_DIR:-data/slue_sqa5_qg_aug_unitpt50k_ckpt10000_codes}"
export MODEL_PATH="${MODEL_PATH:-ckpts/audio-t5-pt-flant5-base-c500-l22/checkpoint-50000}"
export RUN_NAME="${RUN_NAME:-slue_sqa5-flan-t5-base-GR-qgaug-unitpt50k-ckpt10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-models/slue_sqa5-flan-t5-base-GR-qgaug-unitpt50k-ckpt10000}"
export REPORT_TO="${REPORT_TO:-none}"

bash run.sh
