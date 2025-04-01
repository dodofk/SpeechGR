#!/bin/bash
# run.sh: Script to launch T5 pretraining with custom arguments.
python pretrain_t5.py \
  --model_name_or_path "google/flan-t5-base" \
  --learning_rate 0.0001 \
  --sentinel_start_id 32000 \
  --final_model_dir "my_flan_t5_final" \
  --seq_length 512 \
  --mask_prob 0.15 \
  --mean_span_length 3 \
  --evaluation_strategy steps \
  --eval_steps 2500 \
  --max_steps 100000 \
  --project "audio-t5-pretrain" \
  --description "Pretraining T5 on spoken discrete units" \
  --output_dir "audio-t5-pt-flant5-base-c500-l22" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --logging_steps 2500 \
  --save_steps 2500 \
  --save_strategy "steps" \
  --save_total_limit 4 \
  --report_to "wandb"
