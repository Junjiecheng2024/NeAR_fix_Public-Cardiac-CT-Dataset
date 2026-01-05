#!/bin/bash
#SBATCH --job-name=eval_gt_coronary
#SBATCH -A project_2016526
#SBATCH -p medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --output=/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase3/logs/eval_gt_coronary_%j.out
#SBATCH --error=/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase3/logs/eval_gt_coronary_%j.err

# ============================================================================
# Evaluation for GT Coronary Bypass Experiment
# ============================================================================
# Evaluates repaired_phase3_gt_coronary vs Ground Truth
# ============================================================================

WORKDIR=/scratch/project_2016517/JunjieCheng
PROJDIR=/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset
CONTAINER=$WORKDIR/pytorch.sif
DATA_ROOT=$WORKDIR/dataset
GT_ROOT=$DATA_ROOT/original/segmentations

export PYTHONUSERBASE=$WORKDIR/pyuser
export HOME=$WORKDIR

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Evaluation: GT Coronary Experiment"
echo "=============================================="

cd $PROJDIR

# Create temporary symlink so evaluation script finds the right Phase 3 output
cd $DATA_ROOT
if [ -d "repaired_phase3_gt_coronary" ]; then
    # Backup original repaired_phase3 if exists
    if [ -d "repaired_phase3" ]; then
        mv repaired_phase3 repaired_phase3_backup_$$
    fi
    ln -s repaired_phase3_gt_coronary repaired_phase3
    echo "Symlinked repaired_phase3 -> repaired_phase3_gt_coronary"
else
    echo "ERROR: repaired_phase3_gt_coronary not found!"
    exit 1
fi

cd $PROJDIR

srun apptainer exec \
    -B /scratch:/scratch \
    -B /projappl:/projappl \
    $CONTAINER \
    python repairing/phase3/evaluate_repair_quality.py \
    --data_root $DATA_ROOT \
    --gt_root $GT_ROOT \
    --output_csv $PROJDIR/repairing/phase3/evaluation_gt_coronary.csv \
    --use_gt_coronary

# Restore original
cd $DATA_ROOT
rm -f repaired_phase3
if [ -d "repaired_phase3_backup_$$" ]; then
    mv repaired_phase3_backup_$$ repaired_phase3
    echo "Restored original repaired_phase3"
fi

echo "=============================================="
echo "Evaluation Complete!"
echo "Results: $PROJDIR/repairing/phase3/evaluation_gt_coronary.csv"
echo "=============================================="
