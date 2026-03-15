#!/bin/bash
#SBATCH --job-name=near_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# ============================================================================
# NeAR v2.0 Evaluation: Verify Repair Quality
# ============================================================================

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
near_setup_env

DATA_ROOT="${NEAR_DATA_ROOT}"
GT_ROOT="${NEAR_GT_ROOT:-${DATA_ROOT}/original/segmentations}"
OUTPUT_CSV="${NEAR_PHASE3_EVAL_CSV}"

EXTRA_ARGS=()
if [[ "${NEAR_SKIP_HD95:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--skip_hd95)
fi

echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Data Root: $DATA_ROOT"
echo "GT Root: $GT_ROOT"
echo "Output CSV: $OUTPUT_CSV"
echo "Container: ${NEAR_CONTAINER:-<disabled>}"
echo "=============================================="

near_run_python \
    "$NEAR_REPO_ROOT/repairing/phase3/evaluate_repair_quality.py" \
    --data_root "$DATA_ROOT" \
    --gt_root "$GT_ROOT" \
    --output_csv "$OUTPUT_CSV" \
    "${EXTRA_ARGS[@]}"

echo "Evaluation Complete."
