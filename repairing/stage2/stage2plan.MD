# 阶段二实施方案：形态学修复与清洗 (Stage 2 Plan)

本方案旨在对 Stage 1 (Single Class Training) 产生的原始预测 Mask 进行精细化的形态学修复和连通分量清洗，为 Stage 3 的多类融合提供高质量的 Per-class Ground Truth。

## 1. 核心策略：分层处理 (Tiered Strategy)

根据心脏解剖结构的物理特性，我们将 10 个类别分为三组，采用不同的修复策略。

### Group 1: 大器官组 (Big Structures)
*   **包含类别**: LV (3), RV (5), RA (4), LA (2), Aorta (6), PA (7), Myocardium (1)
*   **特征**: 体积大，通常为单连通或双连通（如 Aorta/PA 可能因 FOV 截断而断开）。
*   **策略**:
    *   **Closing**: 半径 2 (修复表面小裂缝)
    *   **Fill Holes**: 允许 (填补内部空洞) *注：Myocardium 除外*
    *   **Filtering**: 保留 Top-1 或 Top-2 大分量。

### Group 2: 细微结构组 (Thin/Small Structures)
*   **包含类别**: Coronary (9), LAA (8), PV (10)
*   **特征**: 细长管状或分散结构，极易断裂。
*   **策略**:
    *   **Closing**: 半径 1 (轻微连接)
    *   **Fill Holes**: **禁止** (防止填死管腔)
    *   **Filtering**: 宽松的 Top-K 或 Ratio 策略，防止误删断裂的血管段。

### Group 3: 特殊规则 (Special Rules)
*   **Myocardium (Class 1)**: 环状结构。
    *   **Trick**: `Dilate -> Fill Holes -> Erode`。既能补洞，又不会填满中间的心室。
*   **Coronary (Class 9)**: 极难修复，Dice 较低。
    *   **Trick**: 额外做一次 Closing 连接断点，并保留 Top-2 (左冠+右冠)。

## 2. 参数配置表

| Class ID | Name | Group | Radius | Fill Holes | Strategy | K | Min Vol | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Myocardium | Special | 2 | False | Top-K | 1 | 500 | 特殊闭合策略 |
| 2 | LA | Big | 2 | True | Top-K | 2 | 500 | 防 PV 入口截断 |
| 3 | LV | Big | 2 | True | Top-K | 1 | 500 | |
| 4 | RA | Big | 2 | True | Top-K | 1 | 500 | |
| 5 | RV | Big | 2 | True | Top-K | 1 | 500 | |
| 6 | Aorta | Big | 2 | True | Top-K | 2 | 500 | 升/降主动脉可能分开 |
| 7 | PA | Big | 2 | True | Top-K | 2 | 200 | 左右肺动脉可能分开 |
| 8 | LAA | Small | 1 | False | Top-K | 1 | 50 | |
| 9 | Coronary | Small | 1 | False | Top-K | 2 | 50 | 左右冠 |
| 10 | PV | Small | 1 | False | Top-K | 4 | 50 | 4根肺静脉 |

## 3. 实施步骤

### 步骤 1: 准备环境
确保安装了 `connected-components-3d` 和 `nibabel`。
*(已完成)*

### 步骤 2: 部署脚本
在 `/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/stage2` 下部署：
1.  `perform_morphology_v2.py`: 核心处理脚本。
2.  `run_phase2.sh`: 批量执行脚本。

### 步骤 3: 执行处理
运行 `run_phase2.sh`。脚本将遍历所有已生成的推理结果文件夹，生成对应的 `_processed` 文件夹。
*   输入: `.../stage2/classX_NAME_results_256/*.npy`
*   输出: `.../stage2/classX_NAME_results_256_processed/*.npy` (及 .nii.gz)

### 步骤 4: 验证
检查 `morphology_stats.csv` 和生成的 `.nii.gz` 文件，确认：
1.  大器官的 CC 数量是否稳定在 1 或 2。
2.  冠脉是否保留了主要分支。
3.  心肌是否没有内部空洞。

## 4. 文件路径说明
*   **代码目录**: `/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/stage2`
*   **原始数据 (Ref)**: `/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset_backup/dataset/original`
*   **推理结果 (Input)**: 同代码目录下的 `classX_...` 文件夹。
