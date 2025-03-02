#!/bin/bash
# make sure that per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
# make sure that aggregator_hidden_dim % aggregator_num_heads == 0
# export WANDB_MODE=disabled

deepspeed --include=localhost:0,4,5,6,7 --master_port 10085 llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path /data/model_weights/vicuna/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /data/training_data/LLaVA-v1.5/instruction_tuning_665k/llava_v1_5_mix665k_MultiModalOneRound_with_SentenceEmbed_v1.json \
    --image_folder /data/training_data/LLaVA-v1.5/instruction_tuning_665k \
    --vision_tower /data/model_weights/clip/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter llava-v1.5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --vit_aggregate_layers "[2, 5, 11, 17]" \
    --aggregate_by_average False \
    --aggregator_num_transformers 4 \
    --aggregator_num_heads 4 \
    --aggregator_hidden_dim 1024 \
    --num_vit_aggregate_group 4 \
    --vit_aggregate_group "[[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24]]" \
    --sentence_embedder /data/model_weights/all-mpnet-base-v2 \
    --sentence_embed_dim 768 \
    --image_aspect_ratio square \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir llava-v1.5-7b-custom-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4000 \
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
    # --max_steps 1 \
   
