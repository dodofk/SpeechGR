#!/bin/bash
# run_qformer_whisper.sh — launch Q-Former-T5 on SLUE-SQA-5 using Whisper features
# Usage: edit the flags below, then:
#   bash run_qformer_whisper.sh
python3 run_qformer.py \
  --model_name "google/flan-t5-base" \
  --run_name "slue_sqa5-flan-t5-base-qformer-whisper-N1-D2-L17-S17" \
  --output_dir "models/slue_sqa5-flan-t5-base-qformer-whisper-N1-D2-L17-S17" \
  --learning_rate 1e-4 \
  --warmup_steps 5000 \
  --per_device_train_batch_size 36 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
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
  --d_model_front 768 \
  --n_queries 1 \
  --qformer_depth 2 \
  --win_size_f 17 \
  --win_stride_f 17 \
  --freeze_t5_encoder False \
  --model_type "qformer" \
  --use_whisper_features True \
  --whisper_model_name "openai/whisper-base" \
  --device "cuda" \
  --apply_spec_augment True \
  --time_warp_param 80 \
  --freq_mask_param 27 \
  --time_mask_param 100 \
  --debug_max_samples 500 \
  --run_note "Train flant5-base + qformer with Whisper features, 1 query per window, 17 window size and 17 window stride"

echo "✅ run_qformer_whisper.sh completed." 