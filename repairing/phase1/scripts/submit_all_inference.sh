#!/bin/bash
# ==============================================================================
# submit_all_inference.sh
# ==============================================================================
# Submit inference jobs for all 10 cardiac classes
#
# Usage:
#   ./submit_all_inference.sh           # Submit all classes
#   ./submit_all_inference.sh --dry-run # Show what would be submitted
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_inference_sbatch.sh"

# All 10 classes
CLASSES=(
    "coronary"
    "aorta"
    "myocardium"
    "la"
    "lv"
    "ra"
    "rv"
    "pa"
    "pv"
    "laa"
)

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

echo "=============================================="
echo "NeAR Phase1 - Inference Submission"
echo "=============================================="
echo ""
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
echo "All ${#CLASSES[@]} inference jobs submitted!"
echo "=============================================="
echo ""
echo "Monitor with: squeue -u \$USER"
