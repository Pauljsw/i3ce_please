#!/bin/bash
# ==============================================================================
# Stage 2: Instruction Tuning for Scaffold Missing Detection
# ==============================================================================
#
# Purpose:
#   Fine-tune the full model for scaffold missing detection task.
#   Uses the mm_projector weights from Stage 1.
#
# What is trained:
#   - LLaMA backbone (via LoRA or full fine-tuning)
#   - mm_projector (continued training)
#
# What is frozen:
#   - ReCon++ point cloud encoder (vision_tower)
#
# Data:
#   - Instruction-following format with missing detection QA pairs
#   - UNIFORM question templates (no data leakage)
#
# Expected time: ~7-14 hours on 8x A100
#
# ==============================================================================

set -e

# Configuration
LLM_VERSION="qizekun/ShapeLLM_7B_gapartnet_v1.0"
OUTPUT_NAME="scaffold-stage2-instruction-tuning_mmproj_tune"

# Stage 1 checkpoint (mm_projector)
STAGE1_CHECKPOINT="./checkpoints/scaffold-stage1-feature-alignment"
MM_PROJECTOR_PATH="${STAGE1_CHECKPOINT}/mm_projector.bin"

# Data paths (Stage 2 SFT data)
DATA_PATH="./playground/data/shapellm/scaffold_v2/stage2_sft_train.json"
PCS_PATH="./playground/data/shapellm/scaffold_v2/pcs"

# Output directory
OUTPUT_DIR="./checkpoints/${OUTPUT_NAME}"
LOG_DIR="./logs/${OUTPUT_NAME}"

# Create log directory for tensorboard
mkdir -p $LOG_DIR

echo "=============================================================="
echo "Stage 2: Instruction Tuning"
echo "=============================================================="
echo "Stage 1 checkpoint: ${STAGE1_CHECKPOINT}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================================="

# Check prerequisites
if [ ! -d "$STAGE1_CHECKPOINT" ]; then
    echo "ERROR: Stage 1 checkpoint not found at ${STAGE1_CHECKPOINT}"
    echo "Please run Stage 1 training first:"
    echo "  bash scripts/pretrain_scaffold_stage1.sh"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Training data not found at ${DATA_PATH}"
    echo "Please run the data generation pipeline first:"
    echo "  python -m tools.scaffold_data_pipeline.main --num-scenes 3000"
    exit 1
fi

# Find mm_projector.bin
if [ -f "$MM_PROJECTOR_PATH" ]; then
    echo "Using mm_projector from: ${MM_PROJECTOR_PATH}"
elif [ -f "${STAGE1_CHECKPOINT}/mm_projector/checkpoint-500.bin" ]; then
    MM_PROJECTOR_PATH="${STAGE1_CHECKPOINT}/mm_projector/checkpoint-500.bin"
    echo "Using mm_projector from: ${MM_PROJECTOR_PATH}"
else
    # Find latest checkpoint
    LATEST=$(ls -t ${STAGE1_CHECKPOINT}/mm_projector/*.bin 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        MM_PROJECTOR_PATH="$LATEST"
        echo "Using mm_projector from: ${MM_PROJECTOR_PATH}"
    else
        echo "ERROR: Cannot find mm_projector.bin in ${STAGE1_CHECKPOINT}"
        exit 1
    fi
fi

# Run training
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --model_name_or_path $LLM_VERSION \
    --version v1 \
    --data_path $DATA_PATH \
    --point_folder $PCS_PATH \
    --vision_tower ReConV2/cfgs/pretrain/large/openshape.yaml \
    --vision_tower_path ./checkpoints/recon/large.pth \
    --pretrain_mm_mlp_adapter $MM_PROJECTOR_PATH \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --freeze_mm_mlp_adapter False \
    --mm_vision_select_layer -2 \
    --mm_use_pt_start_end False \
    --mm_use_pt_patch_token False \
    --sample_points_num 10000 \
    --with_color True \
    --with_ape True \
    --with_local True \
    --with_global True \
    --prompt_token_num 32 \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --logging_dir $LOG_DIR

echo "=============================================================="
echo "Stage 2 Training Complete!"
echo "=============================================================="
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Next step: Run evaluation with:"
echo "  python -m llava.eval.model_vqa \\"
echo "    --model-path ${OUTPUT_DIR} \\"
echo "    --model-base ${LLM_VERSION} \\"
echo "    --question-file ./playground/data/shapellm/scaffold_v2/test_questions.jsonl \\"
echo "    --point-folder ./playground/data/shapellm/scaffold_v2/pcs \\"
echo "    --answers-file ${OUTPUT_DIR}/test_answers.jsonl"
echo "=============================================================="
