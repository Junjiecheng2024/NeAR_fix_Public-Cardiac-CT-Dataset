#!/bin/bash
# ==============================================================================
# submit_remaining_training.sh
# ==============================================================================
# Submit training jobs for the 5 remaining classes (data in project_2016526)
# Classes: LA, LV, RA, RV, PA
#
# Uses config_2016526.py and run_class_2016526_sbatch.sh
#
# Usage:
#   ./submit_remaining_training.sh           # Submit all 5 classes
#   ./submit_remaining_training.sh --dry-run # Show what would be submitted
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_class_2016526_sbatch.sh"

# Remaining 5 classes (data in project_2016526)
CLASSES=(
    "la"
    "lv"
    "ra"
    "rv"
    "pa"
)

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

echo "=============================================="
echo "NeAR Phase1 - Remaining Classes Training"
echo "=============================================="
echo ""
echo "Data location: /scratch/project_2016526/JunjieCheng/dataset/"
echo "Classes to submit: ${CLASSES[*]}"
echo "Total: ${#CLASSES[@]} jobs"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY RUN] Would submit:"
    for class in "${CLASSES[@]}"; do
        echo "  sbatch ${SBATCH_SCRIPT} ${class}"
    done
    echo ""
    echo "Run without --dry-run to actually submit."
    exit 0
fi

echo "Submitting jobs..."
echo ""

for class in "${CLASSES[@]}"; do
    echo -n "Submitting ${class}... "
    JOB_ID=$(sbatch "${SBATCH_SCRIPT}" "${class}" | awk '{print $4}')
    echo "Job ID: ${JOB_ID}"
done

echo ""
echo "=============================================="
echo "All ${#CLASSES[@]} jobs submitted!"
echo "=============================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all:   scancel -u \$USER"
