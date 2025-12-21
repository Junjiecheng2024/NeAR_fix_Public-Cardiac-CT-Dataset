#!/bin/bash
#SBATCH -A project_2016517
#SBATCH --job-name=prepare_all_classes
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/project_2016517/JunjieCheng/logs/prepare_all_%j.out
#SBATCH --error=/scratch/project_2016517/JunjieCheng/logs/prepare_all_%j.err

# ============================================================================
# NeAR v2.0 - Prepare All Cardiac Classes (Excluding Coronary)
# ============================================================================
# This script processes all 9 remaining cardiac classes:
# Myocardium, LA, LV, RA, RV, Aorta, PA, LAA, PV
#
# Coronary is skipped since it's already processed.
# ============================================================================

# Load environment
module load python-data/3.12-25.09
source /projappl/project_2016517/JunjieCheng/junjieenv/bin/activate

# Setup paths
export PYTHONPATH=$PYTHONPATH:$(pwd)
export OMP_NUM_THREADS=1

set -e

# Configuration
PROJECT_ROOT="/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset"
DATA_ROOT="/scratch/project_2016517/JunjieCheng/dataset"

# Original data paths (use correct JunjieCheng path)
ORIGINAL_IMAGES="${DATA_ROOT}/original/images"
ORIGINAL_LABELS="${DATA_ROOT}/original/segmentations"

# Output base directory
OUTPUT_DIR="${DATA_ROOT}"

# Create logs directory
mkdir -p /scratch/project_2016517/JunjieCheng/logs

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=============================================="
echo ""
echo "Project Root: ${PROJECT_ROOT}"
echo "Images Dir: ${ORIGINAL_IMAGES}"
echo "Labels Dir: ${ORIGINAL_LABELS}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=============================================="

# ==============================================================================
# Process All Classes (Skip Coronary)
# ==============================================================================
echo ""
echo "Starting multi-class data preparation..."
echo "Classes to process: Myocardium, LA, LV, RA, RV, Aorta, PA, LAA, PV"
echo ""

cd ${PROJECT_ROOT}

python data_prepare/prepare_all_classes_tier2.py \
    --images_dir ${ORIGINAL_IMAGES} \
    --labels_dir ${ORIGINAL_LABELS} \
    --output_dir ${OUTPUT_DIR} \
    --target_resolution 128 \
    --n_workers 20 \
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
