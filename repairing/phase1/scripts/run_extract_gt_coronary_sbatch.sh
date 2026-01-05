#!/bin/bash
#SBATCH --job-name=extract_gt_coronary
#SBATCH -A project_2016526
#SBATCH --ntasks=1
#SBATCH -p medium
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/extract_gt_%j.out
#SBATCH --error=/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/extract_gt_%j.err

# ============================================================================
# Extract GT Coronary for Phase 2/3 Bypass Experiment
# ============================================================================
# Extracts Coronary (class_id=9) from original GT segmentations,
# resizes to 256³, and saves in Phase 1 output format for Phase 2/3.
# ============================================================================

WORKDIR=/scratch/project_2016517/JunjieCheng
PROJDIR=/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset
CONTAINER=$WORKDIR/pytorch.sif
DATA_ROOT=$WORKDIR/dataset

export PYTHONUSERBASE=$WORKDIR/pyuser
export HOME=$WORKDIR

# Create log directory
mkdir -p $WORKDIR/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs

# Paths
INPUT_DIR=$DATA_ROOT/original/segmentations
OUTPUT_DIR=$DATA_ROOT/coronary_global_gt

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Task: Extract GT Coronary (Bypass Phase 1)"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

cd $PROJDIR

srun apptainer exec \
    -B /scratch:/scratch \
    -B /projappl:/projappl \
    $CONTAINER \
    python repairing/phase1/extract_original_coronary.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --target_size 256

echo "=============================================="
echo "GT Coronary Extraction Done!"
echo "Next: Run Phase 2 morphology on $OUTPUT_DIR"
echo "=============================================="
