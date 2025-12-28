#!/bin/bash
#SBATCH -A project_2016526
#SBATCH --job-name=near_inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -p gpusmall
#SBATCH -o /scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/inference_%x_%j.out
#SBATCH -e /scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/inference_%x_%j.err

# ============================================================================
# NeAR v2.0 Phase1 Inference Script
# ============================================================================
# Run inference for a trained model to generate probability maps
#
# Usage:
#   sbatch run_inference_sbatch.sh coronary
#   sbatch run_inference_sbatch.sh aorta
#   sbatch run_inference_sbatch.sh la
# ============================================================================

# Get class name
CLASS_NAME=${1:-coronary}
echo "Running inference for class: $CLASS_NAME"

# Directories
WORKDIR=/scratch/project_2016517/JunjieCheng
PROJDIR=/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset
CHECKPOINT_BASE=$WORKDIR/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/checkpoints

# Environment setup
export PYTHONUSERBASE=$WORKDIR/pyuser
export PIP_CACHE_DIR=$WORKDIR/pip-cache
export TMPDIR=$WORKDIR/tmp
export XDG_CACHE_HOME=$WORKDIR/.cache
export HOME=$WORKDIR

mkdir -p $PYTHONUSERBASE $PIP_CACHE_DIR $TMPDIR $XDG_CACHE_HOME
mkdir -p $WORKDIR/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs

export CUDA_VISIBLE_DEVICES=0

# Container path
CONTAINER=$WORKDIR/pytorch.sif

# Determine config file based on class
# Note: All data now in project_2016517
case $CLASS_NAME in
    la|lv|ra|rv|pa)
        CONFIG=$PROJDIR/repairing/phase1/config_2016526.py
        ;;
    *)
        CONFIG=$PROJDIR/repairing/phase1/config.py
        ;;
esac

# All classes use project_2016517 for data
DATA_BASE=/scratch/project_2016517/JunjieCheng/dataset

# Convert class name to match checkpoint folder name
# Abbreviations (LA, LV, RA, RV, PA, LAA, PV) are ALL CAPS
# Others (Coronary, Aorta, Myocardium) are Title Case
case $CLASS_NAME in
    la|lv|ra|rv|pa|laa|pv)
        CLASS_UPPER=$(echo "$CLASS_NAME" | tr '[:lower:]' '[:upper:]')
        ;;
    *)
        CLASS_UPPER=$(echo "$CLASS_NAME" | sed 's/.*/\u&/')
        ;;
esac

# Find the latest checkpoint for this class
CHECKPOINT=$(ls -td ${CHECKPOINT_BASE}/${CLASS_UPPER}_Tier2_* 2>/dev/null | head -1)/best.ckpt

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found for class $CLASS_NAME"
    echo "Searched in: ${CHECKPOINT_BASE}/${CLASS_UPPER}_Tier2_*/best.ckpt"
    exit 1
fi

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Class: $CLASS_NAME"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "=============================================="

# Output directory for global-space predictions (256³)
OUTPUT_DIR=${DATA_BASE}/${CLASS_NAME}_global

# Run inference at 256³ resolution with coordinate mapping to global space
# NOTE: --no_sliding_window is required because sliding window breaks the
# appearance-to-grid correspondence that the model learned during training
srun apptainer exec --nv \
    -B /scratch:/scratch \
    -B /projappl:/projappl \
    $CONTAINER \
    python $PROJDIR/repairing/phase1/inference.py \
    --config $CONFIG \
    --checkpoint "$CHECKPOINT" \
    --output_dir $OUTPUT_DIR \
    --no_sliding_window \
    --inference_resolution 256 \
    --global_shape 256

echo "Inference complete for class: $CLASS_NAME"
echo "Results saved to: $OUTPUT_DIR (256³ global space)"
