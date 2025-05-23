#!/bin/bash

# finetune_multitask_lora_span_ner.sh - Multi-task training script with span-based NER
# 
# This script trains a vision-language model with two auxiliary tasks:
# 1. Object detection with percentage-based coordinates (for visual understanding)
# 2. Span-based named entity recognition (for textual understanding)
# 
# The span-based NER approach allows for identifying multi-token entities like
# "New York City" or "The White House" as complete units, improving entity recognition.

# Select model to fine-tune
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# Set up environment and paths
export PYTHONPATH=src:$PYTHONPATH

# Calculate gradient accumulation steps based on available GPUs
GLOBAL_BATCH_SIZE=4
BATCH_PER_DEVICE=4
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# Run the multi-task training
deepspeed src/training/train_multitask.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /data1/dnapolitano/MM/scripts/FOCUS/data/train_sample_2000.json \
    --image_folder /data1/dnapolitano/MM/data/VQA/Images/mscoco/train2014 \
    --include_detection True \
    --detection_annotation_path /data1/dnapolitano/MM/scripts/FOCUS/data/train_annotation.json \
    --num_object_classes 46294 \
    --object_detection_layer 11 \
    --detection_loss_weight 0.5 \
    --include_ner True \
    --ner_annotation_path  /data1/dnapolitano/MM/scripts/FOCUS/data/train_NER.json\
    --num_entity_types 97 \
    --ner_layer 11 \
    --ner_loss_weight 0.5 \
    --vision_layer_percentage 0.1 \
    --language_layer_percentage 0.1 \
    --max_span_length 8 \
    --use_boundary_labels True \
    --contrastive_loss_weight 0 \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/multitask_span_ner_lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing False \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4