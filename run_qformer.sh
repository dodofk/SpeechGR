#!/bin/bash
# run_qformer.sh — launch Q-Former-T5 on SLUE-SQA-5
# Usage: edit the flags below, then:
#   bash run_qformer.sh
python3 run_qformer.py \
  --model_name "google/flan-t5-large" \
  --run_name "slue_sqa5-flan-t5-large-qformer-N1-D2-L17-S17-du-l22-c500-fte" \
  --output_dir "models/slue_sqa5-flan-t5-large-qformer-N1-D2-L17-S17-du-l22-c500-fte" \
  --max_length 255 \
  --learning_rate 1e-5 \
  --warmup_steps 5000 \
  --per_device_train_batch_size 24 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --max_steps 250000 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 4 \
  --load_best_model_at_end \
  --metric_for_best_model Hits@20 \
  --greater_is_better True \
  --save_safetensors True \
  --logging_steps 50 \
  --dataloader_num_workers 0 \
  --dataloader_drop_last False \
  --dataset_path "/home/ricky/dodofk/dataset/slue_sqa5/" \
  --code_path "/home/ricky/dodofk/dataset/slue_sqa_code_l22_c500" \
  --discrete_code_num 500 \
  --special_token 32000 \
  --d_model_front 768 \
  --n_queries 1 \
  --qformer_depth 2 \
  --win_size_f 17 \
  --win_stride_f 17 \
  --freeze_t5_encoder True \
  --save_safetensors False \
  --run_note "Train flant5-large + qformer with 500 discrete unit on layer 22 with 1 query per window and 17 window size and 17 window stride"

echo "✅ run_qformer.sh completed."
