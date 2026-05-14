#!/usr/bin/env bash
set -euo pipefail

is_false() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off) return 0 ;;
    *) return 1 ;;
  esac
}
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

if ! is_false "${REQUIRE_CUDA:-True}"; then
  python3 - <<'PY'
import os
import sys

import torch

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
accelerate_cpu = os.environ.get("ACCELERATE_USE_CPU", "")
print(f"[cuda-preflight] CUDA_VISIBLE_DEVICES={visible}")
print(f"[cuda-preflight] ACCELERATE_USE_CPU={accelerate_cpu or '<unset>'}")
print(f"[cuda-preflight] torch={torch.__version__} torch.version.cuda={torch.version.cuda}")
print(f"[cuda-preflight] torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"[cuda-preflight] torch.cuda.device_count()={torch.cuda.device_count()}")
if accelerate_cpu.lower() in {"1", "true", "yes", "on"}:
    sys.exit(
        "[cuda-preflight] ACCELERATE_USE_CPU is forcing CPU. "
        "Run: unset ACCELERATE_USE_CPU"
    )
if not torch.cuda.is_available():
    sys.exit(
        "[cuda-preflight] CUDA is not available. Select a GPU with "
        "CUDA_VISIBLE_DEVICES=0 or CUDA_VISIBLE_DEVICES=1. "
        "Do not use CUDA_VISIBLE_DEVICES=-1 unless you want CPU. "
        "For an intentional CPU debug run, set REQUIRE_CUDA=False."
    )
print(f"[cuda-preflight] selected_device={torch.cuda.get_device_name(0)}")
PY
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
  --validation_fraction 0.08 \
  --min_chunk_length 64 \
  --evaluation_strategy steps \
  --eval_steps 10000 \
  --max_steps 500000 \
  --use_cpu False \
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
