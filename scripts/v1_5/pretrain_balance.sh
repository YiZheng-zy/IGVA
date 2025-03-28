#!/bin/bash
# make sure that per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 256

deepspeed --include=localhost:0,1,2,3,4,5,6,7 /home/lx/LYX903_balance/llava/train/train_mem.py \
    --deepspeed /home/lx/LYX903_balance/scripts/zero2.json \
    --model_name_or_path /data/model_weights/vicuna/vicuna-7b-v1.5 \
    --version plain \
    --data_path /data/training_data/LLaVA-v1.5/pre_train_558k/blip_laion_cc_sbu_558k.json \
    --image_folder /data/training_data/LLaVA-v1.5/pre_train_558k/images \
    --vision_tower /data/model_weights/clip/clip-vit-large-patch14-336 \
    --image_aspect_ratio "pad" \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --vit_base_layer -2 \
    --vit_aggregate_groups "[[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24]]" \
    --aggregate_by_average True \
    --bf16 True \
    --output_dir /data/proj903/lx-5-pt \
    --lambda_balance 0 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 6 \
    --lazy_preprocess True \
    --report_to wandb \
    #--max_steps 1 \