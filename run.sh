#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Define common directories
# BASE_DIR=$(pwd)
# OUTPUT_DIR="/home/ricky/dodofk/dataset/slue_sqa_code_l22_c500_vad"
# # KM_MODEL_PATH="${BASE_DIR}/models/kmeans_model.bin"  # Path to your kmeans model
# KM_MODEL_PATH="speech-content-encoder/km_100h_c500/km_feat_layer_22"

# echo "Step 1: Running Speech Content Encoding with S2U.py"
# python speech-content-encoder/S2U.py \
#   --km_path ${KM_MODEL_PATH} \
#   --output_dir ${OUTPUT_DIR} \
#   --layer 22 \
#   --sample_rate 16000 \
#   --chunk_length 250000 \
#   --device cuda \
#   --save_format code \
#   --use_vad

echo "Step 2: Running DSI model training with run.py"
python3 run.py \
    --model_name "google/flan-t5-base" \
    --run_name "slue_sqa5-flan-t5-base-DSI-QG-q&d-both-du-l22-c500-wpt-d128" \
    --max_length 128 \
    --output_dir "models/slue_sqa5-flan-t5-base-DSI-QG-q&d-both-du-l22-c500-wpt-d128" \
    --learning_rate 0.0001 \
    --warmup_steps 10000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size  8 \
    --evaluation_strategy steps \
    --eval_steps 2500 \
    --max_steps 100000 \
    --save_strategy steps \
    --dataloader_num_workers 0 \
    --save_steps 2500 \
    --save_total_limit 4 \
    --load_best_model_at_end \
    --gradient_accumulation_steps 8 \
    --report_to wandb \
    --logging_steps 200 \
    --dataloader_drop_last False \
    --metric_for_best_model Hits@20 \
    --greater_is_better True \
    --save_safetensors True \
    --run_note "fine-tune on flan-t5 with 500 cluster discrete unit on layer 22 with pretrain checkpoint with document max length 128 " \
    --code_path "/home/ricky/dodofk/dataset/slue_sqa_code_l22_c500" \
    --discrete_code_num 500 \
    --bf16 True \
    --model_path ckpts/audio-t5-pt-flant5-base-c500-l22/checkpoint-219000
echo "Execution completed successfully!"