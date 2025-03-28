#!/bin/bash

deepspeed --include=localhost:0,1,2,3,4,5,6,7 /home/lx/LYX903_balance/llava/train/train_mem.py \
    --deepspeed /home/lx/LYX903_balance/scripts/zero3.json \
    --model_name_or_path /data/model_weights/vicuna/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /data/training_data/LLaVA-v1.5/instruction_tuning_665k/llava_v1_5_mix665k_AvgEmbed.json \
    --sentence_embed_folder /data/training_data/LLaVA-v1.5/instruction_tuning_665k/sentence_embed_avg \
    --image_folder /data/training_data/LLaVA-v1.5/instruction_tuning_665k \
    --vision_tower /data/model_weights/clip/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /data/proj903/lx-5-pt/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --vit_base_layer -2 \
    --vit_aggregate_groups "[[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24]]" \
    --aggregate_by_average False \
    --aggregator_num_transformers 4 \
    --aggregator_num_heads 4 \
    --aggregator_hidden_dim 1024 \
    --sentence_embedder /data/model_weights/all-mpnet-base-v2 \
    --sentence_embed_dim 768 \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data/proj903/lx-AvgEmbed-balance \
    --lambda_balance 0.02 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    #--max_steps 2
