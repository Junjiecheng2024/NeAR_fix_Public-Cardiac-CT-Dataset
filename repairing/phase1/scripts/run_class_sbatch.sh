#!/bin/bash
#SBATCH -A project_2016526
#SBATCH --job-name=near_phase1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=47:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=gpumedium
#SBATCH -o /scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/%x_%j.out
#SBATCH -e /scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/%x_%j.err

# ============================================================================
# NeAR v2.0 Phase1 Training Script - Multi-Class Support
# ============================================================================
# Usage:
#   sbatch run_class_sbatch.sh coronary    # Train coronary
#   sbatch run_class_sbatch.sh aorta       # Train aorta
#   sbatch run_class_sbatch.sh la          # Train left atrium
#   sbatch --export=CLASS_NAME=aorta run_class_sbatch.sh   # Alternative
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

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Class: $CLASS_NAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "=============================================="

# Container path
CONTAINER=$WORKDIR/pytorch.sif

# Run training with torchrun for DDP
srun apptainer exec --nv \
    -B /scratch:/scratch \
    -B /projappl:/projappl \
    $CONTAINER \
    python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=1 \
    --master_port=29500 \
    $PROJDIR/repairing/phase1/train.py \
    --config $PROJDIR/repairing/phase1/config.py \
    --class_name $CLASS_NAME \
    --devices 4 \
    --strategy ddp

echo "Training complete for class: $CLASS_NAME"
