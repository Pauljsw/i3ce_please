#!/bin/bash

# Scaffold Validation Script
# 훈련된 모델로 validation 데이터셋 평가

MODEL_VERSION=shapellm-7bs-scaffold2
TAG=lorascaffold-test2
type=scaffold

# 체크포인트 경로 설정 (가장 최근 체크포인트 또는 best 체크포인트)
MODEL_PATH="./checkpoints/$MODEL_VERSION-$type-$TAG"
MODEL_BASE="qizekun/ShapeLLM_7B_gapartnet_v1.0"

# Validation 데이터 경로
VAL_ANNO_PATH="./playground/data/shapellm/scaffold_sft/instructions_val.json"
VAL_PCS_PATH="./playground/data/shapellm/scaffold_sft/pcs"

# 출력 디렉토리
OUTPUT_DIR="./eval_results/$MODEL_VERSION-$type-$TAG"
mkdir -p $OUTPUT_DIR

# 출력 파일명
OUTPUT_FILE="scaffold_val_results.json"

echo "================================================"
echo "Starting Scaffold Validation"
echo "Model Path: $MODEL_PATH"
echo "Validation Annotations: $VAL_ANNO_PATH"
echo "Validation Point Clouds: $VAL_PCS_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "================================================"

# Validation 실행
CUDA_VISIBLE_DEVICES=0,1,2,3 python llava/eval/eval_scaffold_val.py \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --data_path $VAL_PCS_PATH \
    --anno_path $VAL_ANNO_PATH \
    --pointnum 10000 \
    --use_color \
    --output_dir $OUTPUT_DIR \
    --output_file $OUTPUT_FILE \
    --temperature 0.2 \
    --top_k 1 \
    --num_beams 1 \
    --shuffle False \
    --num_workers 4

echo "================================================"
echo "Validation completed!"
echo "Results saved to: $OUTPUT_DIR/$OUTPUT_FILE"
echo "================================================"