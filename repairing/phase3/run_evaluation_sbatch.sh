#!/bin/bash
#SBATCH --job-name=near_eval
#SBATCH -A project_2016526
#SBATCH -p medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --output=/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase3/logs/eval_%j.out
#SBATCH --error=/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase3/logs/eval_%j.err

# ============================================================================
# NeAR v2.0 Evaluation: Verify Repair Quality
# ============================================================================

WORKDIR=/scratch/project_2016517/JunjieCheng
PROJDIR=/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset
CONTAINER=$WORKDIR/pytorch.sif
DATA_ROOT=$WORKDIR/dataset
GT_ROOT=$DATA_ROOT/original/segmentations

export PYTHONUSERBASE=$WORKDIR/pyuser
export HOME=$WORKDIR

cd $PROJDIR

srun apptainer exec \
    -B /scratch:/scratch \
    -B /projappl:/projappl \
    $CONTAINER \
    python repairing/phase3/evaluate_repair_quality.py \
    --data_root $DATA_ROOT \
    --gt_root $GT_ROOT \
    --output_csv $PROJDIR/repairing/phase3/evaluation_results_full.csv

echo "Evaluation Complete."
