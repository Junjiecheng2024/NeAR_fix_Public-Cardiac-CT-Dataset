#!/bin/bash
# ==============================================================================
# NeAR v2.0 Coronary Tier2 Pipeline
# ==============================================================================
# Complete pipeline for Coronary PoC:
# 1. Data preparation (class-specific crop)
# 2. Training (Shape + Appearance)
# 3. Inference
# 4. Coordinate mapping (Phase 1.5)
# 5. Integration with Phase 2 & 3
# ==============================================================================

set -e

# Configuration
PROJECT_ROOT="/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset"
DATA_ROOT="/scratch/project_2016517/junjie/dataset"

# Original data paths
ORIGINAL_IMAGES="${DATA_ROOT}/original/images"
ORIGINAL_LABELS="${DATA_ROOT}/original/segmentations"

# Tier2 data output
TIER2_DATA="${DATA_ROOT}/coronary_tier2"

# Training output
CHECKPOINT_DIR="${PROJECT_ROOT}/repairing/phase1/checkpoints"

# Final output (global space predictions)
GLOBAL_OUTPUT="${DATA_ROOT}/coronary_tier2_global"

# ==============================================================================
# Step 1: Data Preparation
# ==============================================================================
echo "=============================================="
echo "Step 1: Preparing Coronary Tier2 Data"
echo "=============================================="

cd ${PROJECT_ROOT}/data_prepare

python prepare_coronary_tier2.py \
    --images_dir ${ORIGINAL_IMAGES} \
    --labels_dir ${ORIGINAL_LABELS} \
    --output_dir ${TIER2_DATA} \
    --margin 20 \
    --target_resolution 256 \
    --n_workers 16

echo "Data preparation complete!"

# ==============================================================================
# Step 2: Training
# ==============================================================================
echo "=============================================="
echo "Step 2: Training NeAR v2.0 Tier2 Model"
echo "=============================================="

cd ${PROJECT_ROOT}/repairing/phase1

# Single GPU training
python train.py \
    --config config.py \
    --devices 1

# For multi-GPU training, use:
# python train.py \
#     --config config.py \
#     --devices 4 \
#     --strategy ddp

echo "Training complete!"

# ==============================================================================
# Step 3: Inference
# ==============================================================================
echo "=============================================="
echo "Step 3: Running Inference"
echo "=============================================="

# Find the latest checkpoint
LATEST_CKPT=$(ls -td ${CHECKPOINT_DIR}/Coronary_Tier2_v2_* | head -1)/best.ckpt

python inference.py \
    --config config.py \
    --checkpoint ${LATEST_CKPT} \
    --output_dir ${TIER2_DATA}/predictions \
    --chunk_size 128

echo "Inference complete!"

# ==============================================================================
# Step 4: Map to Global Space (Phase 1.5)
# ==============================================================================
echo "=============================================="
echo "Step 4: Mapping to Global 256³ Space"
echo "=============================================="

python map_tier2_to_global.py \
    --tier2_dir ${TIER2_DATA}/predictions \
    --crop_params_dir ${TIER2_DATA} \
    --output_dir ${GLOBAL_OUTPUT} \
    --global_shape 256 \
    --n_workers 16

echo "Coordinate mapping complete!"

# ==============================================================================
# Step 5: Integration with Phase 2 & 3 (TODO)
# ==============================================================================
echo "=============================================="
echo "Next Steps:"
echo "=============================================="
echo "1. Run Phase 2 morphological processing on ${GLOBAL_OUTPUT}"
echo "2. Integrate with other classes in Phase 3 fusion"
echo "3. Run verification: python verify_all.py"
echo "=============================================="
echo "Pipeline complete!"
