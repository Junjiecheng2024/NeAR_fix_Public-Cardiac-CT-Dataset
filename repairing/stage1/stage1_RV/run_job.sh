#!/bin/bash
#SBATCH -A project_2016517
#SBATCH -p gpumedium
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH -J RV_train

JOB_NAME="RV_train" 
PY_FILE="train_RV_pl.py"        # 你要运行的训练脚本路径



echo "==== SLURM JOB INFO ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "========================"

### === 激活你的环境 ===
module load python-data/3.10-24.04    # 或者你自己使用的 python 模块版本
source /projappl/project_2016517/JunjieCheng/junjieenv/bin/activate

### === 切换到你的代码目录（你需要改成你实际的路径）===
cd /projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/near_repairing/stage1_RV

echo "Current working directory: $(pwd)"
echo "Starting training..."

### === 运行你的 Python 代码 ===
python ${PY_FILE} --devices 4 --strategy ddp --config config_RV

echo "Training finished."
