#!/bin/bash
#SBATCH --job-name=phase3_gt_coronary
#SBATCH -A project_2016526
#SBATCH --ntasks=1
#SBATCH -p medium
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase3/logs/phase3_gt_coronary_%j.out
#SBATCH --error=/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase3/logs/phase3_gt_coronary_%j.err

# ============================================================================
# Phase 3 with GT Coronary Bypass Experiment
# ============================================================================
# Uses coronary_morph_gt (from GT extraction) instead of coronary_morph
# All other classes use their original Phase 2 outputs.
# 
# Strategy: Temporarily swap coronary_morph with coronary_morph_gt,
# run Phase 3, then restore.
# ============================================================================

WORKDIR=/scratch/project_2016517/JunjieCheng
PROJDIR=/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset
CONTAINER=$WORKDIR/pytorch.sif
DATA_ROOT=$WORKDIR/dataset

export PYTHONUSERBASE=$WORKDIR/pyuser
export HOME=$WORKDIR

mkdir -p $DATA_ROOT/repaired_phase3_gt_coronary
mkdir -p $WORKDIR/NeAR_fix_Public-Cardiac-CT-Dataset/phase3/logs

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Phase 3: Fusion with GT Coronary"
echo "Data Root: $DATA_ROOT"
echo "Output: $DATA_ROOT/repaired_phase3_gt_coronary"
echo "=============================================="

cd $DATA_ROOT

# ========== Step 1: Backup and Swap Coronary ==========
echo "Swapping coronary_morph with coronary_morph_gt..."

if [ -d "coronary_morph" ]; then
    mv coronary_morph coronary_morph_backup_$$
    echo "  Backed up coronary_morph -> coronary_morph_backup_$$"
fi

if [ -d "coronary_morph_gt" ]; then
    # Create symlink so we don't have to copy files
    ln -s coronary_morph_gt coronary_morph
    echo "  Created symlink coronary_morph -> coronary_morph_gt"
else
    echo "ERROR: coronary_morph_gt not found! Run Phase 2 GT coronary first."
    # Restore backup if swap failed
    if [ -d "coronary_morph_backup_$$" ]; then
        mv coronary_morph_backup_$$ coronary_morph
    fi
    exit 1
fi

# ========== Step 2: Run Phase 3 ==========
cd $PROJDIR

srun apptainer exec \
    -B /scratch:/scratch \
    -B /projappl:/projappl \
    $CONTAINER \
    python repairing/phase3/phase3.py \
    --data_root $DATA_ROOT \
    --output_dir $DATA_ROOT/repaired_phase3_gt_coronary

# ========== Step 3: Restore Original Coronary ==========
echo "Restoring original coronary_morph..."
cd $DATA_ROOT

# Remove symlink
rm -f coronary_morph

# Restore backup
if [ -d "coronary_morph_backup_$$" ]; then
    mv coronary_morph_backup_$$ coronary_morph
    echo "  Restored coronary_morph from backup"
fi

echo "=============================================="
echo "Phase 3 with GT Coronary Done!"
echo "Output: $DATA_ROOT/repaired_phase3_gt_coronary"
echo ""
echo "Next: Run evaluation to compare results"
echo "=============================================="
