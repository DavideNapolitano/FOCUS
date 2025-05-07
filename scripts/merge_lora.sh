#!/bin/bash

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /data1/dnapolitano/MM/scripts/FOCUS/output/multitask_span_ner_lora \
    --model-base $MODEL_NAME  \
    --save-model-path /data1/dnapolitano/MM/scripts/FOCUS/merged \
    --safe-serialization