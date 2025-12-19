#!/bin/bash
#SBATCH -A project_2016526
#SBATCH --job-name=near_tier2_phase1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -p gpumedium
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/%j.out
#SBATCH --error=/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/%j.err

set -euo pipefail

# 目录配置
# - PROJECTDIR: 代码所在位置 (projappl)
# - OUTDIR: 日志和checkpoints输出位置 (scratch)
WORKDIR=/scratch/project_2016517/JunjieCheng
PROJECTDIR=/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset
OUTDIR=/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1
IMG=$WORKDIR/pytorch.sif

# 外置 pip 安装位置（容器内 --user 会写到这里）
export PYTHONUSERBASE=$WORKDIR/pyuser
export PIP_CACHE_DIR=$WORKDIR/pip-cache
export TMPDIR=$WORKDIR/pip-tmp
export XDG_CACHE_HOME=$WORKDIR/.cache

# WandB 和 Matplotlib 配置（避免写入只读的家目录）
export WANDB_DIR=$WORKDIR/wandb
export WANDB_CONFIG_DIR=$WORKDIR/.config/wandb
export NETRC=$WORKDIR/.netrc
export MPLCONFIGDIR=$WORKDIR/.config/matplotlib
export HOME=$WORKDIR  # 让所有工具使用 scratch 作为 HOME

mkdir -p "$OUTDIR/logs" "$OUTDIR/checkpoints" "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$PYTHONUSERBASE"
mkdir -p "$WANDB_DIR" "$WANDB_CONFIG_DIR" "$MPLCONFIGDIR"

# 让 pyuser/bin 里的命令可用（accelerate、wandb 等）
export PATH="$PYTHONUSERBASE/bin:$PATH"

# 线程/显存配置
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$PROJECTDIR"

# 用容器运行（使用 srun 启动多 GPU 训练）
srun apptainer exec --nv \
  -B /scratch:/scratch \
  -B /projappl:/projappl \
  "$IMG" \
  bash -lc "
    set -e
    echo 'Using python:' \$(which python)
    python -c 'import torch; print(\"torch:\", torch.__version__, \"cuda:\", torch.cuda.is_available())'

    python repairing/phase1/train.py \
      --devices 4 \
      --strategy ddp \
      --config repairing/phase1/config.py
  "

