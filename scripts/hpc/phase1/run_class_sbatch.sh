#!/bin/bash
#SBATCH --job-name=near_phase1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:4
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

# ============================================================================
# NeAR v2.0 Phase1 Training Script - Multi-Class Support
# ============================================================================
# Uses Lightning's native multi-GPU support (no torchrun)
#
# Usage:
#   sbatch scripts/hpc/phase1/run_class_sbatch.sh coronary    # Train coronary
#   sbatch scripts/hpc/phase1/run_class_sbatch.sh aorta       # Train aorta
# ============================================================================

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
near_setup_env

# Get class name from command line argument or environment variable
CLASS_NAME=${1:-${CLASS_NAME:-coronary}}
PHASE1_DEVICES="${NEAR_PHASE1_DEVICES:-4}"

if [[ -n "${NEAR_CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${NEAR_CUDA_VISIBLE_DEVICES}"
fi

echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Class: $CLASS_NAME"
echo "Repo Root: $NEAR_REPO_ROOT"
echo "Data Root: $NEAR_DATA_ROOT"
echo "Checkpoint Root: $NEAR_PHASE1_CHECKPOINT_ROOT"
echo "Logger: ${NEAR_LOGGER:-csv}"
echo "Container: ${NEAR_CONTAINER:-<disabled>}"
echo "Devices: ${PHASE1_DEVICES}"
echo "=============================================="

# Run training - Lightning handles multi-GPU via devices=4
near_run_python_gpu \
    "$NEAR_REPO_ROOT/repairing/phase1/train.py" \
    --config "$NEAR_REPO_ROOT/repairing/phase1/config.py" \
    --class_name "$CLASS_NAME" \
    --devices "$PHASE1_DEVICES" \
    --logger "${NEAR_LOGGER:-csv}"

echo "Training complete for class: $CLASS_NAME"
