#!/bin/bash

# ======== 定义GPU列表 =========
# 从环境变量CUDA_VISIBLE_DEVICES中获取GPU列表，如果未定义，则默认为0（即使用第一个GPU）
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# 将GPU列表拆分成一个数组，GPULIST，每个元素代表一个GPU
IFS=',' read -ra GPULIST <<< "$gpu_list"
# GPU列表中的元素个数，即并行任务的数量，赋值给CHUNKS
CHUNKS=${#GPULIST[@]}

# ========= 定义检查点和数据集分割 ============
CKPT="llava-v1.5-13b"
SPLIT="llava_vqav2_mscoco_test-dev2015"

# ========== 启动多GPU并行评估 ============
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path liuhaotian/llava-v1.5-13b \
        --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

