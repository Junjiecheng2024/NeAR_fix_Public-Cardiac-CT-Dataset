#!/bin/bash
#SBATCH --job-name=prep_full_tier2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

# ============================================================================
# Re-generate FULL Tier2 Data (CT + Masks + CropParams)
# ============================================================================
# The inference model (Shape+Appearance) requires CT data ("appearance").
# This script regenerates the complete dataset for all 10 classes
# and writes it under the configured NEAR data root.
# ============================================================================

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
near_setup_env

IMAGES_DIR="${NEAR_IMAGES_DIR:-${NEAR_DATA_ROOT}/original/images}"
LABELS_DIR="${NEAR_LABELS_DIR:-${NEAR_DATA_ROOT}/original/segmentations}"
OUTPUT_DIR="${NEAR_PREP_OUTPUT_DIR:-${NEAR_DATA_ROOT}}"
TARGET_RESOLUTION="${NEAR_TARGET_RESOLUTION:-128}"
N_WORKERS="${NEAR_N_WORKERS:-32}"

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Generating FULL Tier2 data for ALL 10 classes"
echo "Images Dir: ${IMAGES_DIR}"
echo "Labels Dir: ${LABELS_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

near_run_python \
    "${NEAR_REPO_ROOT}/data_prepare/prepare_all_classes_tier2.py" \
    --all \
    --images_dir "${IMAGES_DIR}" \
    --labels_dir "${LABELS_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --target_resolution "${TARGET_RESOLUTION}" \
    --n_workers "${N_WORKERS}"

echo "=============================================="
echo "Done! Full Tier2 data generated."
echo "=============================================="
