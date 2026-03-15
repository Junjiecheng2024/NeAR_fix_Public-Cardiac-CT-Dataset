#!/bin/bash
#SBATCH --job-name=near_phase2_morph
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=1-10
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -e slurm-%x-%A_%a.err

# ============================================================================
# NeAR Phase 2: Morphological Processing (SLURM Array Job)
# ============================================================================
# Runs perform_morphology_v2.py for each class independently.
# ============================================================================

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
near_setup_env

# SLURM_ARRAY_TASK_ID corresponds to the class ID (1-10)
CLASS_ID=$SLURM_ARRAY_TASK_ID

# Define Class Name based on ID
case $CLASS_ID in
    1) CLASS_NAME="myocardium" ;;
    2) CLASS_NAME="la" ;;
    3) CLASS_NAME="lv" ;;
    4) CLASS_NAME="ra" ;;
    5) CLASS_NAME="rv" ;;
    6) CLASS_NAME="aorta" ;;
    7) CLASS_NAME="pa" ;;
    8) CLASS_NAME="laa" ;;
    9) CLASS_NAME="coronary" ;;
    10) CLASS_NAME="pv" ;;
    *) echo "Unknown Class ID: $CLASS_ID"; exit 1 ;;
esac

echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID:-local}, Array Task ID: ${SLURM_ARRAY_TASK_ID:-$CLASS_ID}"
echo "Processing Class: $CLASS_NAME (ID: $CLASS_ID)"
echo "Repo Root: $NEAR_REPO_ROOT"
echo "Data Root: $NEAR_DATA_ROOT"
echo "Container: ${NEAR_CONTAINER:-<disabled>}"
echo "=================================================="

# Paths
# Input: The global output from Phase 1 inference
INPUT_DIR="${NEAR_DATA_ROOT}/${CLASS_NAME}_global"
# Output: The morphologically processed directory
OUTPUT_DIR="${NEAR_DATA_ROOT}/${CLASS_NAME}_morph"
# Ref: Original dataset for NIfTI headers
REF_DIR="${NEAR_REF_ROOT:-${NEAR_DATA_ROOT}/original}"

echo "Input Dir: $INPUT_DIR"
echo "Output Dir: $OUTPUT_DIR"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory does not exist! ($INPUT_DIR)"
    echo "Make sure Phase 1 inference has finished."
    exit 1
fi

near_run_python \
    "$NEAR_REPO_ROOT/repairing/phase2/perform_morphology_v2.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --target_class "$CLASS_ID" \
    --ref_dir "$REF_DIR"

echo "=================================================="
echo "Class $CLASS_NAME processing complete."
echo "=================================================="
