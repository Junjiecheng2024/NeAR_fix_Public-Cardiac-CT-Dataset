#!/bin/bash
#SBATCH -A project_2016526
#SBATCH --job-name=near_phase1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=gpumedium
#SBATCH -o /scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/%x_%j.out
#SBATCH -e /scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/%x_%j.err

# ============================================================================
# NeAR v2.0 Phase1 Training Script - Multi-Class Support
# ============================================================================
# Uses Lightning's native multi-GPU support (no torchrun)
#
# Usage:
#   sbatch run_class_sbatch.sh coronary    # Train coronary
#   sbatch run_class_sbatch.sh aorta       # Train aorta
# ============================================================================

# Get class name from command line argument or environment variable
CLASS_NAME=${1:-${CLASS_NAME:-coronary}}
echo "Training class: $CLASS_NAME"

# Directories
WORKDIR=/scratch/project_2016517/JunjieCheng
PROJDIR=/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset

# Environment setup
export PYTHONUSERBASE=$WORKDIR/pyuser
export PIP_CACHE_DIR=$WORKDIR/pip-cache
export TMPDIR=$WORKDIR/tmp
export XDG_CACHE_HOME=$WORKDIR/.cache
export HOME=$WORKDIR

# WandB setup (avoid write to read-only home)
export WANDB_DIR=$WORKDIR/wandb
export WANDB_CONFIG_DIR=$WORKDIR/.config/wandb
export NETRC=$WORKDIR/.netrc
export MPLCONFIGDIR=$WORKDIR/.config/matplotlib

# Create directories
mkdir -p $PYTHONUSERBASE $PIP_CACHE_DIR $TMPDIR
mkdir -p $XDG_CACHE_HOME $WANDB_DIR $WANDB_CONFIG_DIR $MPLCONFIGDIR
mkdir -p $WORKDIR/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs
mkdir -p $WORKDIR/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/checkpoints

# Make all 4 GPUs visible
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Class: $CLASS_NAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# Container path
CONTAINER=$WORKDIR/pytorch.sif

# Run training - Lightning handles multi-GPU via devices=4
srun apptainer exec --nv \
    -B /scratch:/scratch \
    -B /projappl:/projappl \
    $CONTAINER \
    python $PROJDIR/repairing/phase1/train.py \
    --config $PROJDIR/repairing/phase1/config.py \
    --class_name $CLASS_NAME \
    --devices 4

echo "Training complete for class: $CLASS_NAME"
