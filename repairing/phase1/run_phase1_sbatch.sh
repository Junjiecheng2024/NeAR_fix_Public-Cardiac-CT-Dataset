#!/bin/bash
#SBATCH -A project_2016526
#SBATCH --job-name=near_tier2_fusion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -p gpumedium
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Load necessary modules
module load python-data/3.12-25.09
source /projappl/project_2016517/JunjieCheng/junjieenv/bin/activate

# Define project root
PROJECT_ROOT="/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset"

export PYTHONPATH=$PYTHONPATH:${PROJECT_ROOT}
export OMP_NUM_THREADS=8

# Create logs directory
mkdir -p logs

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Run training with Fusion model
cd ${PROJECT_ROOT}/repairing/phase1

python train_tier2.py \
    --config configs/coronary_tier2_fusion.py \
    --devices 4 \
    --strategy ddp
