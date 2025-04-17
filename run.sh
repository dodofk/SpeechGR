#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Define common directories
# BASE_DIR=$(pwd)
# OUTPUT_DIR="/home/ricky/dodofk/dataset/slue_sqa_code_l22_c2000"
# # KM_MODEL_PATH="${BASE_DIR}/models/kmeans_model.bin"  # Path to your kmeans model
# KM_MODEL_PATH="speech-content-encoder/hubert_22_km_2000"

# echo "Step 1: Running Speech Content Encoding with S2U.py"
# python speech-content-encoder/S2U.py \
#   --km_path ${KM_MODEL_PATH} \
#   --output_dir ${OUTPUT_DIR} \
#   --layer 22 \
#   --sample_rate 16000 \
#   --chunk_length 250000 \
#   --device cuda \
#   --save_format both

# echo "Step 2: Running DSI model training with run.py"
python3 run.py \
    --model_name "google/flan-t5-base" \
    --run_name "slue_sqa5-flan-t5-base-DSI-QG-q&d-both-du-l22-c2000-bpe6000" \
    --max_length 512 \
    --output_dir "models/slue_sqa5-flan-t5-base-DSI-QG-q&d-both-du-l22-c2000-bpe6000" \
    --learning_rate 0.0001 \
    --warmup_steps 10000 \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size  8 \
    --evaluation_strategy steps \
    --eval_steps 2500 \
    --max_steps 100000 \
    --save_strategy steps \
    --dataloader_num_workers 0 \
    --save_steps 2500 \
    --save_total_limit 4 \
    --load_best_model_at_end \
    --gradient_accumulation_steps 12 \
    --report_to wandb \
    --logging_steps 100 \
    --dataloader_drop_last False \
    --metric_for_best_model Hits@20 \
    --greater_is_better True \
    --save_safetensors True \
    --run_note "fine-tune on flan t5 with 2000 cluster discrete unit and subword modeling vocab size 6000 on layer 22" \
    --code_path "/home/ricky/dodofk/dataset/slue_sqa_code_l22_c2000_bpe" \
    --discrete_code_num 6000 \
    --bf16 True
# echo "Execution completed successfully!"