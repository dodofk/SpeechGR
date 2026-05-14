#!/usr/bin/env bash
set -euo pipefail

is_false() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off) return 0 ;;
    *) return 1 ;;
  esac
}
DATASET_PATH="${DATASET_PATH:-/home/ricky/dodofk/dataset/slue_sqa5}"
CODE_DIR="${CODE_DIR:-hf://datasets/dodofk/slue-sqa-code-l22-c500}"
TOKEN_FILE="${TOKEN_FILE:-ckpts/token_lookups/flan-t5-base-c500-l22-token-lookup.txt}"

CHECKPOINT_ARGS=()
if is_false "${SAVE_CHECKPOINTS:-True}"; then
  CHECKPOINT_ARGS=(
    --save_strategy no
    --load_best_model_at_end False
  )
else
  CHECKPOINT_ARGS=(
    --save_steps 10000
    --save_strategy "steps"
    --save_total_limit 2
    --load_best_model_at_end
    --metric_for_best_model eval_loss
    --greater_is_better False
    --save_safetensors True
  )
fi
FINAL_SAVE_ARGS=()
if is_false "${SAVE_FINAL_MODEL:-True}"; then
  FINAL_SAVE_ARGS=(--save_final_model False)
fi
HF_CHECKPOINT_ARGS=()
if [[ -n "${HF_CHECKPOINT_REPO_ID:-}" ]] && ! is_false "${SAVE_CHECKPOINTS:-True}"; then
  HF_CHECKPOINT_ARGS=(
    --hf_checkpoint_repo_id "${HF_CHECKPOINT_REPO_ID}"
    --hf_checkpoint_private "${HF_CHECKPOINT_PRIVATE:-False}"
    --hf_checkpoint_mode "${HF_CHECKPOINT_MODE:-model}"
    --hf_checkpoint_latest_path "${HF_CHECKPOINT_LATEST_PATH:-latest}"
    --hf_checkpoint_best_path "${HF_CHECKPOINT_BEST_PATH:-best}"
    --hf_checkpoint_prune_old "${HF_CHECKPOINT_PRUNE_OLD:-True}"
  )
  if [[ -n "${HF_CHECKPOINT_REVISION:-}" ]]; then
    HF_CHECKPOINT_ARGS+=(--hf_checkpoint_revision "${HF_CHECKPOINT_REVISION}")
  fi
fi

python3 t5_pretrain.py \
  --model_name_or_path "google/flan-t5-base" \
  --learning_rate 0.0001 \
  --lr_scheduler_type linear \
  --warmup_steps 10000 \
  --max_grad_norm 1.0 \
  --sentinel_start_id 32099 \
  --sentinel_direction -1 \
  --final_model_dir "ckpts/flan-t5-base-c500-l22-final" \
  --seq_length 512 \
  --mask_prob 0.2 \
  --mean_span_length 7 \
  --code_dir "${CODE_DIR}" \
  --token_file "${TOKEN_FILE}" \
  --lookup_source_csv "${DATASET_PATH}/slue_sqa5_pq10_llama32_3b_clean.csv" \
  --validation_fraction 0.08 \
  --min_chunk_length 64 \
  --evaluation_strategy steps \
  --eval_steps 10000 \
  --max_steps 500000 \
  --project "audio-t5-pretrain" \
  --output_dir "ckpts/audio-t5-pt-flant5-base-c500-l22" \
  --per_device_train_batch_size 6 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --logging_steps 100 \
  --bf16 \
  --report_to "wandb" \
  --description "Pretraining T5 on spoken discrete units with canonical span corruption, 500 clusters on layer 22" \
  "${CHECKPOINT_ARGS[@]}" \
  "${FINAL_SAVE_ARGS[@]}" \
  "${HF_CHECKPOINT_ARGS[@]}"
