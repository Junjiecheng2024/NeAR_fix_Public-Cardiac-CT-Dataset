#!/bin/bash
# Phase 2批量形态学处理脚本

# 基础路径设置
# 使用当前脚本所在目录作为基准
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_BACKUP_DIR="/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset_backup"
REF_DIR="${DATA_BACKUP_DIR}/dataset/original" # 包含 segmentations/*.nii.gz

echo "Base Dir: ${BASE_DIR}"
echo "Ref Dir: ${REF_DIR}"

# 循环处理所有类 (1-10)
# 对应关系: 1=Myo, 2=LA, 3=LV, 4=RA, 5=RV, 6=Ao, 7=PA, 8=LAA, 9=Coronary, 10=PV
for class_id in {1..10}
do
    # 定义类名，用于寻找文件夹
    case $class_id in
        1) CLASS_NAME="Myocardium"; DIR_NAME="class1_Myocardium_results_256" ;;
        2) CLASS_NAME="LA";         DIR_NAME="class2_LA_results_256" ;;
        3) CLASS_NAME="LV";         DIR_NAME="class3_LV_results_256" ;;
        4) CLASS_NAME="RA";         DIR_NAME="class4_RA_results_256" ;;
        5) CLASS_NAME="RV";         DIR_NAME="class5_RV_results_256" ;;
        6) CLASS_NAME="Aorta";      DIR_NAME="class6_Aorta_results_256" ;;
        7) CLASS_NAME="PA";         DIR_NAME="class7_PA_results_256" ;;
        8) CLASS_NAME="LAA";        DIR_NAME="class8_LAA_results_256" ;;
        9) CLASS_NAME="Coronary";   DIR_NAME="class9_Coronary_results_256" ;;
        10) CLASS_NAME="PV";        DIR_NAME="class10_PV_results_256" ;;
    esac

    # 寻找输入目录：可能在 classX_NAME/DIR_NAME 或者直接在 DIR_NAME
    # 根据之前的结构，LV 是在 class3_LV/class3_LV_results_256
    # 我们先检查 classX_NAME/DIR_NAME
    
    FOLDER_PREFIX="class${class_id}_${CLASS_NAME}"
    INPUT_PATH="${BASE_DIR}/${FOLDER_PREFIX}/${DIR_NAME}"
    
    # 如果不存在，尝试直接找 DIR_NAME (兼容旧结构)
    if [ ! -d "$INPUT_PATH" ]; then
        INPUT_PATH="${BASE_DIR}/${DIR_NAME}"
    fi

    OUTPUT_PATH="${INPUT_PATH}_processed"
    
    echo "=================================================="
    echo "Starting Phase 2 for Class ${class_id}: ${CLASS_NAME}"
    echo "Input: ${INPUT_PATH}"
    
    # 检查输入目录是否存在
    if [ ! -d "$INPUT_PATH" ]; then
        echo "WARNING: Input directory ${INPUT_PATH} does not exist. Skipping."
        continue
    fi
    
    echo "Output: ${OUTPUT_PATH}"

    # 运行 Python 脚本
    python perform_morphology_v2.py \
        --input_dir "$INPUT_PATH" \
        --output_dir "$OUTPUT_PATH" \
        --target_class $class_id \
        --ref_dir "$REF_DIR"

    echo "Finished Class ${class_id}"
    echo "=================================================="
done

echo "Phase 2 All Done!"
