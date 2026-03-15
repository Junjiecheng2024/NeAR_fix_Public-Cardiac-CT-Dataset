#!/bin/bash
#SBATCH --job-name=prepare_all_classes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# ============================================================================
# NeAR v2.0 - Prepare All Cardiac Classes (Excluding Coronary)
# ============================================================================
# This script processes all 9 remaining cardiac classes:
# Myocardium, LA, LV, RA, RV, Aorta, PA, LAA, PV
#
# Uses Apptainer container for consistent environment.
# ============================================================================

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
near_setup_env

ORIGINAL_IMAGES="${NEAR_IMAGES_DIR:-${NEAR_DATA_ROOT}/original/images}"
ORIGINAL_LABELS="${NEAR_LABELS_DIR:-${NEAR_DATA_ROOT}/original/segmentations}"
OUTPUT_DIR="${NEAR_PREP_OUTPUT_DIR:-${NEAR_DATA_ROOT}}"
TARGET_RESOLUTION="${NEAR_TARGET_RESOLUTION:-256}"
N_WORKERS="${NEAR_N_WORKERS:-20}"

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-$N_WORKERS}"
echo "=============================================="
echo ""
echo "Repo Root: ${NEAR_REPO_ROOT}"
echo "Images Dir: ${ORIGINAL_IMAGES}"
echo "Labels Dir: ${ORIGINAL_LABELS}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Container: ${NEAR_CONTAINER:-<disabled>}"
echo "=============================================="

# ==============================================================================
# Run data preparation through the shared runtime wrapper
# ==============================================================================
echo ""
echo "Starting multi-class data preparation..."
echo "Classes to process: Myocardium, LA, LV, RA, RV, Aorta, PA, LAA, PV"
echo ""

near_run_python \
  "${NEAR_REPO_ROOT}/data_prepare/prepare_all_classes_tier2.py" \
  --images_dir "${ORIGINAL_IMAGES}" \
  --labels_dir "${ORIGINAL_LABELS}" \
  --output_dir "${OUTPUT_DIR}" \
  --target_resolution "${TARGET_RESOLUTION}" \
  --n_workers "${N_WORKERS}" \
  --all \
  --skip_coronary

echo ""
echo "=============================================="
echo "Data preparation complete!"
echo "=============================================="
echo ""
echo "Output directories created:"
echo "  - ${OUTPUT_DIR}/myocardium_tier2/"
echo "  - ${OUTPUT_DIR}/la_tier2/"
echo "  - ${OUTPUT_DIR}/lv_tier2/"
echo "  - ${OUTPUT_DIR}/ra_tier2/"
echo "  - ${OUTPUT_DIR}/rv_tier2/"
echo "  - ${OUTPUT_DIR}/aorta_tier2/"
echo "  - ${OUTPUT_DIR}/pa_tier2/"
echo "  - ${OUTPUT_DIR}/laa_tier2/"
echo "  - ${OUTPUT_DIR}/pv_tier2/"
echo ""
echo "Global summary: ${OUTPUT_DIR}/all_classes_summary.json"
