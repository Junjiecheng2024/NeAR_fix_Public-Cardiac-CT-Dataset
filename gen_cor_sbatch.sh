#!/bin/bash
#SBATCH -A project_2016517
#SBATCH --job-name=near_tier2
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/near_tier2_%j.out
#SBATCH --error=logs/near_tier2_%j.err

# Load environment
module load python-data/3.12-25.09
source /projappl/project_2016517/JunjieCheng/junjieenv/bin/activate

# Setup paths
export PYTHONPATH=$PYTHONPATH:$(pwd)
export OMP_NUM_THREADS=1

# Create logs directory
mkdir -p logs

set -e

# Configuration
PROJECT_ROOT="/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset"
DATA_ROOT="/scratch/project_2016517/junjie/dataset"

# Original data paths
ORIGINAL_IMAGES="${DATA_ROOT}/original/images"
ORIGINAL_LABELS="${DATA_ROOT}/original/segmentations"

# Tier2 data output
TIER2_DATA="${DATA_ROOT}/coronary_tier2"

# ==============================================================================
# Step 1: Data Preparation
# ==============================================================================
echo "=============================================="
echo "Step 1: Preparing Coronary Tier2 Data"
echo "=============================================="

cd ${PROJECT_ROOT}/data_prepare

python prepare_coronary_tier2.py \
    --images_dir ${ORIGINAL_IMAGES} \
    --labels_dir ${ORIGINAL_LABELS} \
    --output_dir ${TIER2_DATA} \
    --margin 20 \
    --target_resolution 256 \
    --n_workers 20

echo "Data preparation complete!"