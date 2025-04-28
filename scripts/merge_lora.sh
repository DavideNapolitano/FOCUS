#!/bin/bash

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /data2/dnapolitano/MM/scripts/Qwen2-VL-Finetune/output/lora_vision_test \
    --model-base $MODEL_NAME  \
    --save-model-path /data2/dnapolitano/MM/scripts/Qwen2-VL-Finetune/merged/merge_test \
    --safe-serialization