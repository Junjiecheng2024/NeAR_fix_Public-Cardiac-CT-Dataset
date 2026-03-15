#!/bin/bash
#SBATCH --job-name=gen_crop_params
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

# ============================================================================
# Generate crop_params.json for all 10 cardiac classes
# ============================================================================
# Output: ${NEAR_DATA_ROOT}/{class}_tier2/
# ============================================================================

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
near_setup_env

LABELS_DIR="${NEAR_LABELS_DIR:-${NEAR_DATA_ROOT}/original/segmentations}"
OUTPUT_BASE="${NEAR_CROP_PARAMS_OUTPUT_BASE:-${NEAR_DATA_ROOT}}"
TARGET_RESOLUTION="${NEAR_TARGET_RESOLUTION:-128}"
N_WORKERS="${NEAR_N_WORKERS:-32}"

echo "=============================================="
echo "Generating crop_params.json for ALL 10 classes"
echo "Labels Dir: ${LABELS_DIR}"
echo "Output: ${OUTPUT_BASE}"
echo "=============================================="

near_run_python \
    "${NEAR_REPO_ROOT}/data_prepare/generate_crop_params.py" \
    --all \
    --labels_dir "${LABELS_DIR}" \
    --output_base "${OUTPUT_BASE}" \
    --target_resolution "${TARGET_RESOLUTION}" \
    --n_workers "${N_WORKERS}"

echo "Done! crop_params.json files saved to ${OUTPUT_BASE}"
