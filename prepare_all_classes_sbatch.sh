#!/bin/bash
#SBATCH -A project_2016526
#SBATCH --job-name=prepare_all_classes
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/project_2016517/JunjieCheng/logs/prepare_all_%j.out
#SBATCH --error=/scratch/project_2016517/JunjieCheng/logs/prepare_all_%j.err

# ============================================================================
# NeAR v2.0 - Prepare All Cardiac Classes (Excluding Coronary)
# ============================================================================
# This script processes all 9 remaining cardiac classes:
# Myocardium, LA, LV, RA, RV, Aorta, PA, LAA, PV
#
# Uses Apptainer container for consistent environment.
# ============================================================================

set -euo pipefail

# 目录配置
WORKDIR=/scratch/project_2016517/JunjieCheng
PROJECTDIR=/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset
IMG=$WORKDIR/pytorch.sif

# 外置 pip 安装位置（容器内 --user 会写到这里）
export PYTHONUSERBASE=$WORKDIR/pyuser
export PIP_CACHE_DIR=$WORKDIR/pip-cache
export TMPDIR=$WORKDIR/pip-tmp
export XDG_CACHE_HOME=$WORKDIR/.cache
export HOME=$WORKDIR

# 原始数据路径
ORIGINAL_IMAGES="${WORKDIR}/dataset/original/images"
ORIGINAL_LABELS="${WORKDIR}/dataset/original/segmentations"
OUTPUT_DIR="${WORKDIR}/dataset"

# 创建必要目录
mkdir -p "$WORKDIR/logs" "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$PYTHONUSERBASE"

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
echo "Output Dir: ${OUTPUT_DIR}"
echo "Container: ${IMG}"
echo "=============================================="

cd "$PROJECTDIR"

# ==============================================================================
# 使用容器运行数据准备
# ==============================================================================
echo ""
echo "Starting multi-class data preparation..."
echo "Classes to process: Myocardium, LA, LV, RA, RV, Aorta, PA, LAA, PV"
echo ""

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
    --all \
    --skip_coronary

echo ""
echo "=============================================="
echo "Data preparation complete!"
echo "=============================================="
echo ""
echo "Output directories created:"
echo "  - ${OUTPUT_DIR}/myocardium_tier2/"
echo "  - ${OUTPUT_DIR}/la_tier2/"
echo "  - ${OUTPUT_DIR}/lv_tier2/"
echo "  - ${OUTPUT_DIR}/ra_tier2/"
echo "  - ${OUTPUT_DIR}/rv_tier2/"
echo "  - ${OUTPUT_DIR}/aorta_tier2/"
echo "  - ${OUTPUT_DIR}/pa_tier2/"
echo "  - ${OUTPUT_DIR}/laa_tier2/"
echo "  - ${OUTPUT_DIR}/pv_tier2/"
echo ""
echo "Global summary: ${OUTPUT_DIR}/all_classes_summary.json"
