#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-google/flan-t5-base}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_PATH_ARGS=()
if [[ -n "${MODEL_PATH}" ]]; then
  MODEL_PATH_ARGS=(--model_path "${MODEL_PATH}")
fi
is_false() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off) return 0 ;;
    *) return 1 ;;
  esac
}
CHECKPOINT_ARGS=()
if is_false "${SAVE_CHECKPOINTS:-True}"; then
  CHECKPOINT_ARGS=(
    --save_strategy no
    --load_best_model_at_end False
  )
else
  CHECKPOINT_ARGS=(
    --save_strategy steps
    --save_steps 2500
    --save_total_limit 2
    --load_best_model_at_end
    --metric_for_best_model unit_f1
    --greater_is_better True
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

python qg.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  "${MODEL_PATH_ARGS[@]}" \
  --final_model_dir "ckpts/flan-t5-querygen" \
  --special_token 32000 \
  --dataset_path "/home/ricky/dodofk/dataset/slue_sqa5/" \
  --code_path "hf://datasets/dodofk/slue-sqa-code-l22-c500" \
  --split train \
  --max_length 512 \
  --label_max_length 300 \
  --discrete_code_num 500 \
  --project "Audio-QG" \
  --description "Query generation on SLUE-SQA5 with discrete units from flan-t5-base" \
  --output_dir "ckpts/flan-t5-QG" \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --evaluation_strategy steps \
  --eval_steps 2500 \
  --max_steps 100000 \
  --logging_steps 100 \
  --bf16 \
  --report_to "wandb" \
  "${CHECKPOINT_ARGS[@]}" \
  "${FINAL_SAVE_ARGS[@]}" \
  "${HF_CHECKPOINT_ARGS[@]}"
