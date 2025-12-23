#!/bin/bash

LLM_VERSION=qizekun/ShapeLLM_7B_gapartnet_v1.0
MODEL_VERSION=shapellm-7bs-scaffold2
PRETRAIN_TAG=gapartnet-v1.0
TAG=lorascaffold-test2

type=scaffold

if [ $type = "general" ]; then
    meta_path="./playground/data/shapellm/cap3d_objaverse_sft_45k.json"
    pcs_path="./playground/data/shapellm/cap3d_pcs"
elif [ $type = "gapartnet" ]; then
    meta_path="./playground/data/shapellm/gapartnet_sft_27k_openai.json"
    pcs_path="./playground/data/shapellm/gapartnet_pcs"
elif [ $type = "test" ]; then
    meta_path="./playground/data/mini_test/mini_test.json"
    pcs_path="./playground/data/mini_test/pcs"
elif [ $type = "scaffold" ]; then  # ← 추가
    meta_path="./playground/data/shapellm/scaffold_sft/instructions_train.json"
    pcs_path="./playground/data/shapellm/scaffold_sft/pcs"
else
    echo "Unknown type"
    exit 1
fi

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $LLM_VERSION \
    --version v1 \
    --data_path $meta_path \
    --point_folder $pcs_path \
    --vision_tower ReConV2/cfgs/pretrain/large/openshape.yaml \
    --vision_tower_path ./checkpoints/recon/large.pth \
    --sample_points_num 10000 \
    --with_color True \
    --occlusion False \
    --prompt_token_num 32 \
    --with_ape True \
    --with_local True \
    --with_global True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_pt_start_end False \
    --mm_use_pt_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/$MODEL_VERSION-$type-$TAG \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to none