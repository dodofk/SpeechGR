#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# This script is used to extract the feature from the LibriSpeech dataset
# Define common directories
BASE_DIR=$(pwd)
# OUTPUT_DIR="/home/ricky/dodofk/dataset/ls100_code_l22_c500"
OUTPUT_DIR="/home/ricky/dodofk/dataset/ll6k_code_l22_c500"
# KM_MODEL_PATH="${BASE_DIR}/models/kmeans_model.bin"  # Path to your kmeans model
KM_MODEL_PATH="/home/ricky/dodofk/DUAL-textless-SQA/speech-content-encoder/km_100h_c500/km_feat_layer_22"

echo "Step 1: Running Speech Content Encoding with S2U.py"
python speech-content-encoder/S2U.py \
  --km_path ${KM_MODEL_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --layer 22 \
  --sample_rate 16000 \
  --chunk_length 250000 \
  --device cuda \
  --task librispeech
