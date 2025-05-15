#!/bin/bash
python3 run_ranking.py \
    --model_name "google/flan-t5-base" \
    --run_name "slue_sqa5-flan-t5-base-DSI-Ranking-q&d-both-du-l22-c500" \
    --max_length 512 \
    --output_dir "models/slue_sqa5-flan-t5-base-DSI-Ranking-q&d-both-du-l22-c500" \
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
    --gradient_accumulation_steps 8 \
    --logging_steps 100 \
    --dataloader_drop_last False \
    --metric_for_best_model Hits@20 \
    --greater_is_better True \
    --save_safetensors True \
    --run_note "fine-tune on flan-t5-base with 500 cluster discrete unit on layer 22 with new Ranking loss (in batch)" \
    --code_path "/home/ricky/dodofk/dataset/slue_sqa_code_l22_c500" \
    --discrete_code_num 500 \
    --bf16 True 
echo "Execution completed successfully!"