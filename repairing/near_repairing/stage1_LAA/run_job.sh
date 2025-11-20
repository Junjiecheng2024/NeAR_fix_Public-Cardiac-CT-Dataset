#!/bin/bash
#SBATCH -A project_2016517
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=64G
#SBATCH -t 72:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH -J LAA_train

JOB_NAME="LAA_train" 
PY_FILE="/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/near_repairing/stage1_LAA/train_LAA_pl.py"

echo "==== SLURM JOB INFO ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "========================"

### === 激活你的环境 ===
module load python-data/3.10-24.04
source /projappl/project_2016517/chengjun/junjieenv/bin/activate

### === 切换到你的代码目录 ===
cd /home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/near_repairing/stage1_LAA

echo "Current working directory: $(pwd)"
echo "Starting training..."

### === 运行你的 Python 代码 ===
python ${PY_FILE} --devices 4 --strategy ddp --config config_LAA

echo "Training finished."
