#!/bin/bash
# make sure that per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
export WANDB_MODE=disabled
# export TOKENIZERS_PARALLELISM=false
deepspeed --include=localhost:0,1,2,3 --master_port 29501 llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --output_dir llava-v1.5-7b-finetune \
   
