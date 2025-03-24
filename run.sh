#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Define common directories
BASE_DIR=$(pwd)
OUTPUT_DIR="/home/ricky/dodofk/dataset/slue_sqa_code_l22_c500"
# KM_MODEL_PATH="${BASE_DIR}/models/kmeans_model.bin"  # Path to your kmeans model
KM_MODEL_PATH="/home/ricky/dodofk/DUAL-textless-SQA/speech-content-encoder/km_100h_c500/km_feat_layer_22"

echo "Step 1: Running Speech Content Encoding with S2U.py"
python speech-content-encoder/S2U.py \
  --km_path ${KM_MODEL_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --layer 22 \
  --sample_rate 16000 \
  --chunk_length 250000 \
  --device cuda

# echo "Step 2: Running DSI model training with run.py"
# python DSI-QG/run.py \
#   --model_name "google/mt5-small" \
#   --max_length 32 \
#   --id_max_length 64 \
#   --task "DSI" \
#   --dataset_name "slue_sqa5" \
#   --special_token 32000 \
#   --run_notes "Speech retrieval training run" \
#   --output_dir "${BASE_DIR}/model_outputs" \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --gradient_accumulation_steps 4 \
#   --learning_rate 1e-4 \
#   --num_train_epochs 3 \
#   --warmup_ratio 0.1 \
#   --logging_steps 100 \
#   --evaluation_strategy "steps" \
#   --eval_steps 1000 \
#   --save_strategy "steps" \
#   --save_steps 1000 \
#   --save_total_limit 3 \
#   --load_best_model_at_end \
#   --metric_for_best_model "Hits@10" \
#   --fp16 \
#   --report_to "wandb" \
#   --run_name "speech_dsi_training" \
#   --code_path "${OUTPUT_DIR}

# echo "Execution completed successfully!"