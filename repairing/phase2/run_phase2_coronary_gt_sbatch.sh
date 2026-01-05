#!/bin/bash
#SBATCH --job-name=phase2_coronary_gt
#SBATCH -A project_2016526
#SBATCH --ntasks=1
#SBATCH -p medium
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/phase2_coronary_gt_%j.out
#SBATCH --error=/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/phase2_coronary_gt_%j.err

# ============================================================================
# Phase 2 Morphology for GT Coronary Bypass Experiment
# ============================================================================
# Applies morphological processing to extracted GT Coronary masks.
# Input: coronary_global_gt (from extract_original_coronary.py)
# Output: coronary_morph_gt
# ============================================================================

WORKDIR=/scratch/project_2016517/JunjieCheng
PROJDIR=/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset
CONTAINER=$WORKDIR/pytorch.sif
DATA_ROOT=$WORKDIR/dataset

export PYTHONUSERBASE=$WORKDIR/pyuser
export HOME=$WORKDIR

# Create directories
mkdir -p $WORKDIR/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs
mkdir -p $DATA_ROOT/coronary_morph_gt

# Paths for GT Coronary
INPUT_DIR=$DATA_ROOT/coronary_global_gt
OUTPUT_DIR=$DATA_ROOT/coronary_morph_gt
REF_DIR=$DATA_ROOT/original

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Phase 2: Morphology for GT Coronary"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

cd $PROJDIR/repairing/phase2

srun apptainer exec \
    -B /scratch:/scratch \
    -B /projappl:/projappl \
    $CONTAINER \
    python perform_morphology_v2.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --target_class 9 \
    --ref_dir $REF_DIR

echo "=============================================="
echo "Phase 2 Coronary GT Done!"
echo ""
echo "Next steps:"
echo "  1. mv $DATA_ROOT/coronary_morph $DATA_ROOT/coronary_morph_backup"
echo "  2. mv $DATA_ROOT/coronary_morph_gt $DATA_ROOT/coronary_morph"
echo "  3. Run Phase 3"
echo "  4. Restore: mv coronary_morph coronary_morph_gt && mv coronary_morph_backup coronary_morph"
echo "=============================================="
