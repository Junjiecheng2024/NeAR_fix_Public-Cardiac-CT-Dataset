#!/bin/bash
#SBATCH --job-name=near_phase2_morph
#SBATCH --account=project_2016517
#SBATCH --partition=medium
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=1-10
#SBATCH -o /scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/morph_%a_%j.out
#SBATCH -e /scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/morph_%a_%j.err

# ============================================================================
# NeAR Phase 2: Morphological Processing (SLURM Array Job)
# ============================================================================
# Runs perform_morphology_v2.py for each class independently.
# ============================================================================

WORKDIR=/scratch/project_2016517/JunjieCheng
PROJDIR=/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset
CONTAINER=$WORKDIR/pytorch.sif

export PYTHONUSERBASE=$WORKDIR/pyuser
export HOME=$WORKDIR

# Create logs dir
mkdir -p $WORKDIR/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs

# SLURM_ARRAY_TASK_ID corresponds to the class ID (1-10)
CLASS_ID=$SLURM_ARRAY_TASK_ID

# Define Class Name based on ID
case $CLASS_ID in
    1) CLASS_NAME="myocardium" ;;
    2) CLASS_NAME="la" ;;
    3) CLASS_NAME="lv" ;;
    4) CLASS_NAME="ra" ;;
    5) CLASS_NAME="rv" ;;
    6) CLASS_NAME="aorta" ;;
    7) CLASS_NAME="pa" ;;
    8) CLASS_NAME="laa" ;;
    9) CLASS_NAME="coronary" ;;
    10) CLASS_NAME="pv" ;;
    *) echo "Unknown Class ID: $CLASS_ID"; exit 1 ;;
esac

echo "=================================================="
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing Class: $CLASS_NAME (ID: $CLASS_ID)"
echo "=================================================="

# Paths
# Input: The global output from Phase 1 inference
INPUT_DIR="/scratch/project_2016517/JunjieCheng/dataset/${CLASS_NAME}_global"
# Output: The morphologically processed directory
OUTPUT_DIR="/scratch/project_2016517/JunjieCheng/dataset/${CLASS_NAME}_morph"
# Ref: Original dataset for NIfTI headers
REF_DIR="/scratch/project_2016517/JunjieCheng/dataset/original"

echo "Input Dir: $INPUT_DIR"
echo "Output Dir: $OUTPUT_DIR"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory does not exist! ($INPUT_DIR)"
    echo "Make sure Phase 1 inference has finished."
    exit 1
fi

cd $PROJDIR/repairing/phase2

srun apptainer exec \
    -B /scratch:/scratch \
    -B /projappl:/projappl \
    $CONTAINER \
    python perform_morphology_v2.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --target_class $CLASS_ID \
    --ref_dir "$REF_DIR"

echo "=================================================="
echo "Class $CLASS_NAME processing complete."
echo "=================================================="
