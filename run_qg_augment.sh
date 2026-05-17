#!/usr/bin/env bash
set -euo pipefail

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

DATASET_PATH="${DATASET_PATH:-}"
CODE_DIR="${CODE_DIR:-hf://datasets/dodofk/slue-sqa-code-l22-c500}"
QG_MODEL_PATH="${QG_MODEL_PATH:-ckpts/flan-t5-QG-unitpt50k-b12/checkpoint-10000}"
OUTPUT_DATASET_PATH="${OUTPUT_DATASET_PATH:-data/slue_sqa5_qg_aug_unitpt50k_ckpt10000}"
OUTPUT_CODE_DIR="${OUTPUT_CODE_DIR:-data/slue_sqa5_qg_aug_unitpt50k_ckpt10000_codes}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-google/flan-t5-base}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OVERWRITE="${OVERWRITE:-True}"

if [[ -z "${DATASET_PATH}" ]]; then
  echo "Set DATASET_PATH to the SLUE-SQA5 metadata directory before QG augmentation." >&2
  exit 1
fi

is_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

EXTRA_ARGS=()
if is_true "${DO_SAMPLE:-False}"; then
  EXTRA_ARGS+=(--do_sample --top_p "${TOP_P:-0.95}" --temperature "${TEMPERATURE:-1.0}")
fi
if [[ -n "${MAX_DOCUMENTS:-}" ]]; then
  EXTRA_ARGS+=(--max_documents "${MAX_DOCUMENTS}")
fi
if is_true "${COMPRESSED:-False}"; then
  EXTRA_ARGS+=(--compressed)
fi
if is_true "${OVERWRITE}"; then
  EXTRA_ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" scripts/generate_qg_augmented_units.py \
  --qg_model_path "${QG_MODEL_PATH}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --code_path "${CODE_DIR}" \
  --output_dataset_path "${OUTPUT_DATASET_PATH}" \
  --output_code_path "${OUTPUT_CODE_DIR}" \
  --document_source "${DOCUMENT_SOURCE:-corpus}" \
  --max_length "${MAX_LENGTH:-512}" \
  --offset "${OFFSET:-30}" \
  --generation_max_length "${GENERATION_MAX_LENGTH:-302}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --num_return_sequences "${NUM_RETURN_SEQUENCES:-1}" \
  --num_beams "${NUM_BEAMS:-1}" \
  --discrete_code_num "${DISCRETE_CODE_NUM:-500}" \
  --bf16 \
  "${EXTRA_ARGS[@]}"
