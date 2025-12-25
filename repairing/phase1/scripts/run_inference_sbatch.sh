#!/bin/bash
#SBATCH -A project_2016526
#SBATCH --job-name=near_inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -p gpumedium
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
# Classes in project_2016526: la, lv, ra, rv, pa
# Classes in project_2016517: coronary, aorta, myocardium, pv, laa
case $CLASS_NAME in
    la|lv|ra|rv|pa)
        CONFIG=$PROJDIR/repairing/phase1/config_2016526.py
        DATA_BASE=/scratch/project_2016526/JunjieCheng/dataset
        ;;
    *)
        CONFIG=$PROJDIR/repairing/phase1/config.py
        DATA_BASE=/scratch/project_2016517/JunjieCheng/dataset
        ;;
esac

# Convert class name to match checkpoint folder name (capitalize first letter)
CLASS_UPPER=$(echo "$CLASS_NAME" | sed 's/.*/\u&/')

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

# Output directory for predictions
OUTPUT_DIR=${DATA_BASE}/${CLASS_NAME}_tier2/predictions

# Run inference at 256³ resolution
srun apptainer exec --nv \
    -B /scratch:/scratch \
    -B /projappl:/projappl \
    $CONTAINER \
    python $PROJDIR/repairing/phase1/inference.py \
    --config $CONFIG \
    --checkpoint "$CHECKPOINT" \
    --output_dir $OUTPUT_DIR \
    --chunk_size 128 \
    --inference_resolution 256

echo "Inference complete for class: $CLASS_NAME"
echo "Results saved to: $OUTPUT_DIR"
