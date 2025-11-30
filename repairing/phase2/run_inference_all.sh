#!/bin/bash
# Phase 2批量推理脚本 (Inference All Classes)

# 基础路径设置
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_PATH="/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset_backup/dataset/near_format_data"
CHECKPOINT_ROOT="/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/phase1"

# Checkpoint 映射表
# 格式: ClassID|ClassName|CheckpointPathRelative
# Note：CheckpointPathRelative 是相对于 CHECKPOINT_ROOT 的路径
# 或者是绝对路径。为了稳妥，我们直接使用 find 找到的绝对路径或者硬编码。
# 根据用户提供的列表：

declare -A CHECKPOINTS
CHECKPOINTS[1]="${CHECKPOINT_ROOT}/phase1_Myocardium/checkpoints/Myocardium_class1_251122_225221/best.ckpt"
CHECKPOINTS[2]="${CHECKPOINT_ROOT}/phase1_LA/checkpoints/LA_class2_251124_155401/best.ckpt"
CHECKPOINTS[3]="${CHECKPOINT_ROOT}/phase1_LV/checkpoints/LV_class3_251124_153327/best.ckpt"
CHECKPOINTS[4]="${CHECKPOINT_ROOT}/phase1_RA/checkpoints/RA_class4_251124_151008/best.ckpt"
CHECKPOINTS[5]="${CHECKPOINT_ROOT}/phase1_RV/checkpoints/RV_class5_shape_only_251121_111501/best.ckpt"
CHECKPOINTS[6]="${CHECKPOINT_ROOT}/phase1_Aorta/checkpoints/Aorta_class6_shape_only_251123_040844/best.ckpt"
CHECKPOINTS[7]="${CHECKPOINT_ROOT}/phase1_PA/checkpoints/PA_class7_251122_015831/best.ckpt"
CHECKPOINTS[8]="${CHECKPOINT_ROOT}/phase1_LAA/checkpoints/LAA_class8_251122_222337/best.ckpt"
CHECKPOINTS[9]="${CHECKPOINT_ROOT}/phase1_coronary/checkpoints/Coronary_class9_251126_101417/best.ckpt"
CHECKPOINTS[10]="${CHECKPOINT_ROOT}/phase1_PV/checkpoints/PV_class10_251124_223846/best.ckpt"

# 循环处理所有类 (1-10)
for class_id in {1..10}
do
    # 定义类名
    case $class_id in
        1) CLASS_NAME="Myocardium" ;;
        2) CLASS_NAME="LA" ;;
        3) CLASS_NAME="LV" ;;
        4) CLASS_NAME="RA" ;;
        5) CLASS_NAME="RV" ;;
        6) CLASS_NAME="Aorta" ;;
        7) CLASS_NAME="PA" ;;
        8) CLASS_NAME="LAA" ;;
        9) CLASS_NAME="Coronary" ;;
        10) CLASS_NAME="PV" ;;
    esac

    # 如果是 LV (Class 3)，且已经跑过了，可以选择跳过
    # 但为了保险，或者用户可能想覆盖，这里默认执行。
    # 如果想跳过，取消下面注释
    # if [ "$class_id" -eq 3 ]; then
    #     echo "Skipping Class 3 (LV) as it is already processed."
    #     continue
    # fi

    CKPT_PATH="${CHECKPOINTS[$class_id]}"
    OUTPUT_DIR="${BASE_DIR}/class${class_id}_${CLASS_NAME}/class${class_id}_${CLASS_NAME}_results_256"
    
    echo "=================================================="
    echo "Starting Inference for Class ${class_id}: ${CLASS_NAME}"
    echo "Checkpoint: ${CKPT_PATH}"
    echo "Output: ${OUTPUT_DIR}"
    
    if [ ! -f "$CKPT_PATH" ]; then
        echo "ERROR: Checkpoint not found at ${CKPT_PATH}"
        continue
    fi

    # 运行 Python 脚本
    # Note：batch_size 设为 131072 以加快速度 (根据显存调整)
    python inference_and_evaluate.py \
        --checkpoint "$CKPT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --resolution 256 \
        --target_class $class_id \
        --all_samples \
        --data_path "$DATA_PATH" \
        --batch_size 131072

    echo "Finished Inference for Class ${class_id}"
    echo "=================================================="
done

echo "All Inferences Done!"
