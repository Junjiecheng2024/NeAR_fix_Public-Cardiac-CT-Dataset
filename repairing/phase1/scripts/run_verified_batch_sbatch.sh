#!/bin/bash
#SBATCH --job-name=near_verified
#SBATCH -A project_2016526
#SBATCH -p gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-9
#SBATCH -o /scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/verified_%A_%a.out
#SBATCH -e /scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/verified_%A_%a.err

# ============================================================================
# Verified Batch Inference - 使用验证过的逻辑推理所有类
# Usage: sbatch run_verified_batch_sbatch.sh
# ============================================================================

WORKDIR=/scratch/project_2016517/JunjieCheng
PROJDIR=/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset
CONTAINER=$WORKDIR/pytorch.sif
DATA_ROOT=$WORKDIR/dataset

export PYTHONUSERBASE=$WORKDIR/pyuser
export HOME=$WORKDIR

# 10 个类
CLASSES=("myocardium" "la" "lv" "ra" "rv" "aorta" "pa" "laa" "coronary" "pv")
CLASS_NAME=${CLASSES[$SLURM_ARRAY_TASK_ID]}

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Class: $CLASS_NAME"
echo "=============================================="

mkdir -p $WORKDIR/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs

cd $PROJDIR

# 先删除旧的输出
rm -rf $DATA_ROOT/${CLASS_NAME}_global

srun apptainer exec --nv \
    -B /scratch:/scratch \
    -B /projappl:/projappl \
    $CONTAINER \
    python repairing/phase1/inference_batch_verified.py \
    --class_name $CLASS_NAME \
    --data_root $DATA_ROOT

echo "=============================================="
echo "Done: $CLASS_NAME"
echo "=============================================="
