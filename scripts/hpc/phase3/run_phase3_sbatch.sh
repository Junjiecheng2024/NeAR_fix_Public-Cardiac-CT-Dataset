#!/bin/bash
#SBATCH --job-name=near_phase3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# ============================================================================
# NeAR v2.0 Phase 3: Multi-class Fusion & Correction
# ============================================================================
# Fuses 10 single-class masks into one final segmentation
# and applies anatomical constraints.
# Input: dataset/{class}_morph (or _global)
# Output: dataset/repaired_phase3
# ============================================================================

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
near_setup_env

DATA_ROOT="${NEAR_DATA_ROOT}"
OUTPUT_DIR="${NEAR_PHASE3_OUTPUT_DIR:-${DATA_ROOT}/repaired_phase3}"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Phase 3: Fusion and Anatomical Correction"
echo "Data Root: $DATA_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Container: ${NEAR_CONTAINER:-<disabled>}"
echo "=============================================="

near_run_python \
    "$NEAR_REPO_ROOT/repairing/phase3/phase3.py" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR"

echo "=============================================="
echo "Phase 3 Done!"
echo "=============================================="
