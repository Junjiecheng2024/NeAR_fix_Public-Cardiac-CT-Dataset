#!/bin/bash
#SBATCH --job-name=near_inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

# ============================================================================
# NeAR v2.0 Phase1 Inference Script
# ============================================================================
# Run inference for a trained model to generate probability maps
#
# Usage:
#   sbatch scripts/hpc/phase1/run_inference_sbatch.sh coronary
#   sbatch scripts/hpc/phase1/run_inference_sbatch.sh aorta
#   sbatch scripts/hpc/phase1/run_inference_sbatch.sh la
# ============================================================================

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
near_setup_env

# Get class name
CLASS_NAME=${1:-coronary}
echo "Running inference for class: $CLASS_NAME"

if [[ -n "${NEAR_CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${NEAR_CUDA_VISIBLE_DEVICES}"
fi

CONFIG="${NEAR_CONFIG_PATH:-$NEAR_REPO_ROOT/repairing/phase1/config.py}"
CHECKPOINT_BASE="${NEAR_PHASE1_CHECKPOINT_ROOT}"
DATA_BASE="${NEAR_DATA_ROOT}"
INFERENCE_RESOLUTION="${NEAR_INFERENCE_RESOLUTION:-128}"
GLOBAL_SHAPE="${NEAR_GLOBAL_SHAPE:-256}"
CHUNK_SIZE="${NEAR_CHUNK_SIZE:-128}"
OUTPUT_DIR="${NEAR_PHASE1_OUTPUT_DIR:-${DATA_BASE}/${CLASS_NAME}_global}"

# Convert class name to match checkpoint folder name
# Abbreviations (LA, LV, RA, RV, PA, LAA, PV) are ALL CAPS
# Others (Coronary, Aorta, Myocardium) are Title Case
case $CLASS_NAME in
    la|lv|ra|rv|pa|laa|pv)
        CLASS_UPPER=$(echo "$CLASS_NAME" | tr '[:lower:]' '[:upper:]')
        ;;
    *)
        CLASS_UPPER=$(echo "$CLASS_NAME" | sed 's/.*/\u&/')
        ;;
esac

# Find the latest checkpoint for this class
CHECKPOINT=$(ls -td ${CHECKPOINT_BASE}/${CLASS_UPPER}_Tier2_* 2>/dev/null | head -1)/best.ckpt

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found for class $CLASS_NAME"
    echo "Searched in: ${CHECKPOINT_BASE}/${CLASS_UPPER}_Tier2_*/best.ckpt"
    exit 1
fi

echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Class: $CLASS_NAME"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Container: ${NEAR_CONTAINER:-<disabled>}"
echo "=============================================="

# Run inference at 128³ resolution (same as training) then map to global 256³ space
# NOTE: --no_sliding_window is required because sliding window breaks the
# appearance-to-grid correspondence that the model learned during training
# NOTE: Using 128³ inference to avoid GPU OOM (256³ needs ~20GB VRAM)
near_run_python_gpu \
    "$NEAR_REPO_ROOT/repairing/phase1/inference.py" \
    --config "$CONFIG" \
    --class_name "$CLASS_NAME" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --chunk_size "$CHUNK_SIZE" \
    --no_sliding_window \
    --inference_resolution "$INFERENCE_RESOLUTION" \
    --global_shape "$GLOBAL_SHAPE"

echo "Inference complete for class: $CLASS_NAME"
echo "Results saved to: $OUTPUT_DIR (256³ global space)"
