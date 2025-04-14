#!/bin/bash
# run.sh: Script to launch T5 pretraining with custom arguments.
python t5_pretrain.py \
  --model_name_or_path "google/flan-t5-base" \
  --learning_rate 0.0001 \
  --sentinel_start_id 32001 \
  --final_model_dir "ckpts/flan-t5-base-c500-l22-final" \
  --seq_length 512 \
  --mask_prob 0.2 \
  --mean_span_length 7 \
  --evaluation_strategy steps \
  --eval_steps 10000 \
  --max_steps 500000 \
  --project "audio-t5-pretrain" \
  --description "Pretraining T5 on spoken discrete units" \
  --output_dir "ckpts/audio-t5-pt-flant5-base-c500-l22" \
  --per_device_train_batch_size 6 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --logging_steps 100 \
  --save_steps 3000 \
  --save_strategy "steps" \
  --save_total_limit 4 \
  --bf16 \
  --report_to "wandb" \
  --description "Pretraining T5 on spoken discrete units with 500 clusters on layer 22" \
  --model_path "ckpts/audio-t5-pt-flant5-base-c500-l22"
