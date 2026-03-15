# Phase2 当前代码说明

这份文档按当前仓库中的 Phase2 主线代码整理，描述的是实际实现，而不是早期计划稿。

当前主线对应的核心文件是：

- `repairing/phase2/perform_morphology_v2.py`
- `scripts/hpc/phase2/run_phase2_sbatch.sh`

辅助文件包括：

- `repairing/phase2/calculate_phase2_cc.py`
- `repairing/phase2/calculate_phase2_ratios.py`

其中真正执行形态学清理的是 `perform_morphology_v2.py`。

## 1. 整体定位

Phase2 的作用是把 Phase1 生成的单类全局 mask 做进一步清洗，输出更适合后续 Phase3 融合的每类二值结果。

当前实现主要包含三步：

1. 一般形态学处理
2. 连通域筛选
3. 少数类别的额外 special rule

## 2. 输入与输出

### 2.1 输入

`perform_morphology_v2.py` 当前按目录处理输入，默认寻找：

- `input_dir/*_mask.npy`

这些文件通常来自 Phase1 全局推理输出，例如：

- `dataset/{class}_global/{case_id}_mask.npy`

### 2.2 输出

输出目录中会保存：

- 清理后的 `*_mask.npy`
- 可视化用的 `{case_id}_clean.nii.gz`
- 统计文件 `morphology_stats.csv`

当前 NIfTI 导出主要是为了检查结果，代码里保存时实际使用的是单位阵 affine，不是精确回写原始影像坐标。

## 3. 处理流程

`perform_morphology_v2.py` 的处理逻辑可以拆成 3 个函数：

- `step1_general_morphology`
- `step2_cc_filtering`
- `step3_per_class_special`

### 3.1 Step 1: General Morphology

当前实现先做：

- `binary_closing`

如果当前类别配置允许，再做：

- `binary_fill_holes`

closing 的 `iterations` 由每个类别自己的 `radius` 决定。

### 3.2 Step 2: Connected Components Filtering

当前实现使用：

- `cc3d.connected_components`
- 26 邻域连通

筛选规则目前统一走 `top_k` 策略，但会叠加两个限制：

- 绝对体素阈值 `min_vol`
- 相对最大连通域的体积比例检查

如果候选都被过滤掉，但最大连通域本身不至于太小，则会保底保留最大连通域。

### 3.3 Step 3: Per-Class Special Rules

当前代码只对两个类别做额外处理：

- `Myocardium`
- `Coronary`

#### Myocardium

当前实现已经不是早期的：

- `Dilate -> Fill Holes -> Erode`

而是额外再做一次：

- `binary_closing(mask, iterations=2)`

代码注释里也明确写了，这是为了避免把心室腔体错误填满。

#### Coronary

当前会额外再做一次：

- `binary_closing(mask, iterations=1)`

目的是尽量连接容易断裂的冠脉分支。

## 4. 当前类别参数表

下面这张表对应 `perform_morphology_v2.py` 中 `CONFIG` 的实际内容。

| Class ID | Name | radius | fill_holes | strategy | k | min_vol |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Myocardium | 2 | False | top_k | 1 | 500 |
| 2 | LA | 2 | True | top_k | 2 | 500 |
| 3 | LV | 2 | True | top_k | 1 | 500 |
| 4 | RA | 2 | True | top_k | 1 | 500 |
| 5 | RV | 2 | True | top_k | 1 | 500 |
| 6 | Aorta | 2 | True | top_k | 2 | 500 |
| 7 | PA | 2 | True | top_k | 2 | 200 |
| 8 | LAA | 1 | False | top_k | 1 | 50 |
| 9 | Coronary | 1 | False | top_k | 2 | 50 |
| 10 | PV | 1 | False | top_k | 4 | 50 |

从当前实现看，类别大致分成两组：

- 大器官：更倾向于 closing + fill holes + Top-1/Top-2
- 小结构：closing 半径更小，不做 fill holes，Top-K 更宽松

## 5. 当前集群入口

当前保留的主集群脚本是：

- `scripts/hpc/phase2/run_phase2_sbatch.sh`

它的行为是：

1. 通过 SLURM array 处理 10 个类别
2. 从 `dataset/{class}_global/` 读取输入
3. 输出到 `dataset/{class}_morph/`
4. 调用 `repairing/phase2/perform_morphology_v2.py`

也就是说，当前主线已经不是旧文档里的：

- `run_phase2.sh`
- `*_processed/`
- `/repairing/stage2/`

而是：

- `scripts/hpc/phase2/run_phase2_sbatch.sh`
- `{class}_morph`

## 6. 当前代码的几个实际注意点

### 6.1 当前实现只做单类目录处理

`perform_morphology_v2.py` 是“一个类别一次处理一个目录”，不是一次性完成 10 类全流程。

### 6.2 NIfTI 主要用于检查

当前 `.nii.gz` 输出并不是严格用于正式回写原始空间，而更像可视化检查产物。

### 6.3 策略实现比早期计划更简单

当前代码没有实现更复杂的按距离或方向的形态学修复，也没有单独为所有类别写特殊几何规则。

真正实现的是：

- closing
- fill holes
- connected component filtering
- 少量 per-class trick

### 6.4 当前文档只描述代码行为

这份说明不再记录：

- 某次清洗前后的数值结果
- 某个类别当时的 Dice 改善
- 某次实验里使用的固定路径和目录历史

如果后续需要公开更系统的实验分析，建议把统计结果单独写进新的结果文档。
