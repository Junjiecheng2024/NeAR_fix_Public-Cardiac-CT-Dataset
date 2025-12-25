#!/bin/bash
#SBATCH -A project_2016526
#SBATCH --job-name=gen_crop_params
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --partition=medium
#SBATCH --mem=64G
#SBATCH -o /scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/crop_params_%j.out
#SBATCH -e /scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/crop_params_%j.err

# ============================================================================
# Generate crop_params.json for all 10 cardiac classes
# ============================================================================
# Output: /scratch/project_2016517/JunjieCheng/dataset/{class}_tier2/
# ============================================================================

WORKDIR=/scratch/project_2016517/JunjieCheng
PROJDIR=/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset
CONTAINER=$WORKDIR/pytorch.sif

export PYTHONUSERBASE=$WORKDIR/pyuser
export HOME=$WORKDIR

mkdir -p $WORKDIR/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs

echo "=============================================="
echo "Generating crop_params.json for ALL 10 classes"
echo "Output: /scratch/project_2016517/JunjieCheng/dataset/"
echo "=============================================="

cd $PROJDIR

srun apptainer exec \
    -B /scratch:/scratch \
    -B /projappl:/projappl \
    $CONTAINER \
    python data_prepare/generate_crop_params.py \
    --all \
    --labels_dir /scratch/project_2016517/JunjieCheng/dataset/original/segmentations \
    --output_base /scratch/project_2016517/JunjieCheng/dataset \
    --target_resolution 128 \
    --n_workers 32

echo "Done! All crop_params.json saved to project_2016517"
