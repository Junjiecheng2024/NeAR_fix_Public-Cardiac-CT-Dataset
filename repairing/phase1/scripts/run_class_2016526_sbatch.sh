#!/bin/bash
#SBATCH -A project_2016526
#SBATCH --job-name=near_phase1_2016526
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=gpumedium
#SBATCH -o /scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/%x_%j.out
#SBATCH -e /scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/%x_%j.err

# ============================================================================
# NeAR v2.0 Phase1 Training - Classes in project_2016526
# ============================================================================
# For: LA, LV, RA, RV, PA (data in /scratch/project_2016526/)
# Uses config_2016526.py instead of config.py
# ============================================================================

# Get class name from command line argument
CLASS_NAME=${1:-la}
echo "Training class: $CLASS_NAME (using config_2016526.py)"

# Directories
WORKDIR=/scratch/project_2016517/JunjieCheng
PROJECTDIR=/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset

# Environment setup
export PYTHONUSERBASE=$WORKDIR/pyuser
export PIP_CACHE_DIR=$WORKDIR/pip-cache
export TMPDIR=$WORKDIR/tmp
export XDG_CACHE_HOME=$WORKDIR/.cache
export HOME=$WORKDIR

# WandB setup
export WANDB_DIR=$WORKDIR/wandb
export WANDB_CONFIG_DIR=$WORKDIR/.config/wandb
export NETRC=$WORKDIR/.netrc
export MPLCONFIGDIR=$WORKDIR/.config/matplotlib

mkdir -p "$WORKDIR/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs"
mkdir -p "$WORKDIR/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/checkpoints"
mkdir -p "$PYTHONUSERBASE" "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME"
mkdir -p "$WANDB_DIR" "$WANDB_CONFIG_DIR" "$MPLCONFIGDIR"

export PATH="$PYTHONUSERBASE/bin:$PATH"
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Container path
CONTAINER=$WORKDIR/pytorch.sif

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Class: $CLASS_NAME"
echo "Config: config_2016526.py"
echo "Data: /scratch/project_2016526/JunjieCheng/dataset/"
echo "=============================================="

cd "$PROJECTDIR"

# Run training with the separate config file
srun apptainer exec --nv \
  -B /scratch:/scratch \
  -B /projappl:/projappl \
  "$CONTAINER" \
  bash -lc "
    set -e
    echo 'Using python:' \$(which python)
    python -c 'import torch; print(\"torch:\", torch.__version__, \"cuda:\", torch.cuda.is_available())'

    python repairing/phase1/train.py \
      --devices 4 \
      --strategy ddp \
      --config repairing/phase1/config_2016526.py \
      --class_name $CLASS_NAME
  "

echo "Training complete for class: $CLASS_NAME"
