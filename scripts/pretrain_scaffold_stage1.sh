#!/bin/bash
# ==============================================================================
# Stage 1: Feature Alignment for Scaffold Missing Detection
# ==============================================================================
#
# Purpose:
#   Train the mm_projector (multimodal projector) to align ReCon++ point cloud
#   features with the LLM's text embedding space.
#
# What is trained:
#   - mm_projector (ReConProjector with 3 MLPs for pos/local/global features)
#   - Learnable prompt tokens (32 tokens)
#
# What is frozen:
#   - ReCon++ point cloud encoder (vision_tower)
#   - LLaMA backbone (all weights)
#
# Data:
#   - Simple captions describing scaffold structures
#   - Format: <point> -> "A 3-bay, 2-floor scaffold with 8 vertical posts..."
#
# Environment: 4x RTX 3090 (24GB each)
# Effective batch size: 2 * 8 * 4 = 64
#
# Epoch strategy for small dataset (2,100 samples):
#   - Original ShapeLLM: 785K samples × 1 epoch
#   - Our data: 2,100 samples × 15 epochs = ~500 effective steps
#   - This allows proper convergence without severe overfitting
#   - Checkpoints saved every 100 steps for manual selection
#
# ==============================================================================

set -e

# Configuration
LLM_VERSION="qizekun/ShapeLLM_7B_gapartnet_v1.0"
OUTPUT_NAME="scaffold-stage1-feature-alignment"

# Data paths (Stage 1 caption data)
DATA_PATH="./playground/data/shapellm/scaffold_v2/stage1_caption_train.json"
PCS_PATH="./playground/data/shapellm/scaffold_v2/pcs"

# Output directory
OUTPUT_DIR="./checkpoints/${OUTPUT_NAME}"
LOG_DIR="./logs/${OUTPUT_NAME}"

# Create log directory for tensorboard
mkdir -p $LOG_DIR

echo "=============================================================="
echo "Stage 1: Feature Alignment Training (3090 x 4)"
echo "=============================================================="
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Batch: 2 x 8 x 4GPU = 64 effective"
echo "=============================================================="

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Training data not found at ${DATA_PATH}"
    echo "Please run the data generation pipeline first:"
    echo "  python -m tools.scaffold_data_pipeline.main --num-scenes 1000"
    exit 1
fi

# Run training (optimized for 3090 x 4)
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $LLM_VERSION \
    --version plain \
    --data_path $DATA_PATH \
    --point_folder $PCS_PATH \
    --vision_tower ReConV2/cfgs/pretrain/large/openshape.yaml \
    --vision_tower_path ./checkpoints/recon/large.pth \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
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
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --logging_dir $LOG_DIR

echo "=============================================================="
echo "Stage 1 Training Complete!"
echo "=============================================================="
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Next step: Run Stage 2 training with:"
echo "  bash scripts/finetune_scaffold_stage2.sh"
echo "=============================================================="
