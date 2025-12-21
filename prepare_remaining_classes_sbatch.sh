#!/bin/bash
#SBATCH -A project_2016526
#SBATCH --job-name=prepare_remaining
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/project_2016526/JunjieCheng/logs/prepare_remaining_%j.out
#SBATCH --error=/scratch/project_2016526/JunjieCheng/logs/prepare_remaining_%j.err

# ============================================================================
# NeAR v2.0 - Prepare Remaining Cardiac Classes
# ============================================================================
# This script processes the 5 remaining classes that weren't prepared:
# LA, LV, RA, RV, PA
#
# Output to project_2016526 due to disk space on project_2016517
# ============================================================================

set -euo pipefail

# 目录配置
WORKDIR=/scratch/project_2016517/JunjieCheng
WORKDIR_NEW=/scratch/project_2016526/JunjieCheng
PROJECTDIR=/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset
IMG=$WORKDIR/pytorch.sif

# 外置 pip 安装位置
export PYTHONUSERBASE=$WORKDIR/pyuser
export PIP_CACHE_DIR=$WORKDIR/pip-cache
export TMPDIR=$WORKDIR/pip-tmp
export XDG_CACHE_HOME=$WORKDIR/.cache
export HOME=$WORKDIR

# 原始数据路径（从 project_2016517 读取）
ORIGINAL_IMAGES="${WORKDIR}/dataset/original/images"
ORIGINAL_LABELS="${WORKDIR}/dataset/original/segmentations"

# 输出到 project_2016526
OUTPUT_DIR="${WORKDIR_NEW}/dataset"

# 创建必要目录
mkdir -p "$WORKDIR_NEW/logs" "$OUTPUT_DIR"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$PYTHONUSERBASE"

# 让 pyuser/bin 里的命令可用
export PATH="$PYTHONUSERBASE/bin:$PATH"

# 线程配置
export OMP_NUM_THREADS=1

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=============================================="
echo ""
echo "Project Root: ${PROJECTDIR}"
echo "Images Dir: ${ORIGINAL_IMAGES}"
echo "Labels Dir: ${ORIGINAL_LABELS}"
echo "Output Dir: ${OUTPUT_DIR} (project_2016526)"
echo "Container: ${IMG}"
echo "=============================================="

cd "$PROJECTDIR"

# ==============================================================================
# 处理剩余 5 个类（逐个处理，避免全部跑）
# ==============================================================================
echo ""
echo "Processing remaining 5 classes: LA, LV, RA, RV, PA"
echo ""

for CLASS in LA LV RA RV PA; do
    echo "=============================================="
    echo "Processing: ${CLASS}"
    echo "=============================================="
    
    apptainer exec \
      -B /scratch:/scratch \
      -B /projappl:/projappl \
      "$IMG" \
      python data_prepare/prepare_all_classes_tier2.py \
        --images_dir ${ORIGINAL_IMAGES} \
        --labels_dir ${ORIGINAL_LABELS} \
        --output_dir ${OUTPUT_DIR} \
        --target_resolution 256 \
        --n_workers 20 \
        --class_name ${CLASS}
    
    echo "${CLASS} complete!"
    echo ""
done

echo ""
echo "=============================================="
echo "Data preparation complete!"
echo "=============================================="
echo ""
echo "Output directories created:"
echo "  - ${OUTPUT_DIR}/la_tier2/"
echo "  - ${OUTPUT_DIR}/lv_tier2/"
echo "  - ${OUTPUT_DIR}/ra_tier2/"
echo "  - ${OUTPUT_DIR}/rv_tier2/"
echo "  - ${OUTPUT_DIR}/pa_tier2/"
