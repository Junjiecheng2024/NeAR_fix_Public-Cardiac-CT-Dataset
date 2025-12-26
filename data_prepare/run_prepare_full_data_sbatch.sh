#!/bin/bash
#SBATCH -A project_2016526
#SBATCH --job-name=prep_full_tier2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --partition=medium
#SBATCH --mem=128G
#SBATCH -o /scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/prep_tier2_%j.out
#SBATCH -e /scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/prep_tier2_%j.err

# ============================================================================
# Re-generate FULL Tier2 Data (CT + Masks + CropParams)
# ============================================================================
# The inference model (Shape+Appearance) requires CT data ("appearance").
# This script regenerates the complete dataset for all 10 classes
# and saves it to project_2016517.
# ============================================================================

WORKDIR=/scratch/project_2016517/JunjieCheng
PROJDIR=/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset
CONTAINER=$WORKDIR/pytorch.sif

export PYTHONUSERBASE=$WORKDIR/pyuser
export HOME=$WORKDIR

mkdir -p $WORKDIR/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Generating FULL Tier2 data for ALL 10 classes"
echo "Output: /scratch/project_2016517/JunjieCheng/dataset/"
echo "=============================================="

cd $PROJDIR

srun apptainer exec \
    -B /scratch:/scratch \
    -B /projappl:/projappl \
    $CONTAINER \
    python data_prepare/prepare_all_classes_tier2.py \
    --all \
    --images_dir /scratch/project_2016517/JunjieCheng/dataset/original/images \
    --labels_dir /scratch/project_2016517/JunjieCheng/dataset/original/segmentations \
    --output_dir /scratch/project_2016517/JunjieCheng/dataset \
    --target_resolution 128 \
    --n_workers 32

echo "=============================================="
echo "Done! Full Tier2 data generated."
echo "=============================================="
