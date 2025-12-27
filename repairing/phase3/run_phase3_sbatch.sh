#!/bin/bash
#SBATCH --job-name=near_phase3
#SBATCH --account=project_2016526
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p medium
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase3/logs/phase3_%j.out
#SBATCH --error=/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase3/logs/phase3_%j.err

# ============================================================================
# NeAR v2.0 Phase 3: Multi-class Fusion & Correction
# ============================================================================
# Fuses 10 single-class masks into one final segmentation
# and applies anatomical constraints.
# Input: dataset/{class}_morph (or _global)
# Output: dataset/repaired_phase3
# ============================================================================

WORKDIR=/scratch/project_2016517/JunjieCheng
PROJDIR=/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset
CONTAINER=$WORKDIR/pytorch.sif
DATA_ROOT=$WORKDIR/dataset

export PYTHONUSERBASE=$WORKDIR/pyuser
export HOME=$WORKDIR

mkdir -p $PROJDIR/phase3/logs
mkdir -p $DATA_ROOT/repaired_phase3

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Phase 3: Fusion and Anatomical Correction"
echo "Data Root: $DATA_ROOT"
echo "Output: $DATA_ROOT/repaired_phase3"
echo "=============================================="

cd $PROJDIR

srun apptainer exec \
    -B /scratch:/scratch \
    -B /projappl:/projappl \
    $CONTAINER \
    python repairing/phase3/phase3.py \
    --data_root $DATA_ROOT \
    --output_dir $DATA_ROOT/repaired_phase3

echo "=============================================="
echo "Phase 3 Done!"
echo "=============================================="
