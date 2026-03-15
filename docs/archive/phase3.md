# Phase3 当前代码说明

这份文档按当前仓库中的 Phase3 主线代码整理，描述的是实际实现，而不是早期设计目标。

当前主线对应的核心文件是：

- `repairing/phase3/phase3.py`
- `repairing/phase3/evaluate_repair_quality.py`
- `scripts/hpc/phase3/run_phase3_sbatch.sh`
- `scripts/hpc/phase3/run_evaluation_sbatch.sh`

辅助调试脚本已经迁到：

- `tools/debug/phase3_run_case.py`

## 1. 整体定位

Phase3 的作用是把 10 个单类结果融合成一个最终多类分割，并在融合后做少量解剖规则修正。

当前代码实际分成两部分：

1. `phase3.py`
   负责融合与规则修正
2. `evaluate_repair_quality.py`
   负责评估 P1 / P2 / P3 相对 GT 的变化

## 2. 输入与输出

### 2.1 Phase3 输入

`phase3.py` 会对每个 case 调用 `load_masks(case_id, data_root)`，按下面顺序寻找每个类别的 mask：

1. `data_root/{class}_morph/{case_id}_mask.npy`
2. `data_root/{class}_global/{case_id}_mask.npy`
3. `data_root/{class}_global/{case_id}.npy`（legacy fallback）

因此当前优先使用的是：

- Phase2 输出 `{class}_morph`

如果某一类没有 morphology 结果，则退回到：

- Phase1 输出 `{class}_global`

### 2.2 Phase3 输出

当前 `phase3.py` 对每个 case 保存：

- `{case_id}_phase3.npy`
- `{case_id}_phase3.nii.gz`

输出目录通常是：

- `data_root/repaired_phase3/`

这与早期“保存为 `final_segmentation.npy/.nii.gz`”的设计不同。

## 3. 当前融合优先级

当前优先级定义在 `phase3.py` 的 `PRIORITY_ORDER`：

1. Coronary
2. PV
3. LAA
4. LV
5. RV
6. LA
7. RA
8. Myocardium
9. Aorta
10. PA

实现方式是：

- 先把优先级列表反转
- 从低优先级往高优先级写入 `final_mask`
- 后写入的类别覆盖先前类别

因此最终效果仍然是：

- 高优先级结构覆盖低优先级结构

## 4. 当前实际执行的解剖规则

当前真正执行的规则集中在 `enforce_anatomical_constraints()` 中。

### 4.1 PV -> LA

函数：

- `connect_structure_to_target(final_mask, source_cls=10, target_cls=2, max_dist=5)`

逻辑：

- 如果 LA 不存在，直接删除 PV
- 如果 PV 某个连通域离 LA 太远，则删除
- 如果距离在阈值内但尚未接触，则通过膨胀把它接到 LA 附近

### 4.2 LAA -> LA

函数：

- `connect_structure_to_target(final_mask, source_cls=8, target_cls=2, max_dist=3)`

逻辑与 PV 类似，只是距离阈值更小。

### 4.3 Coronary -> Myocardium / Aorta

函数：

- `filter_floating_structures(final_mask, source_cls=9, target_mask=myo_or_ao, max_dist=6)`

逻辑：

- 计算冠脉各连通域到 `Myo ∪ Aorta` 的距离
- 超出阈值的分量会被删除
- 距离足够近的分量保留

### 4.4 Chamber fragment cleanup

函数：

- `clean_chamber_fragments(final_mask, cls)`

当前对：

- LA
- LV
- RA
- RV

逐个处理，规则是：

- 保留最大连通域
- 其余碎片如果离最大连通域足够近（距离 < 3）则保留
- 否则删除

## 5. 当前未启用或未实现的规则

这部分非常重要，因为当前代码和早期设计稿差异主要在这里。

### 5.1 Chamber Enclosure 已禁用

`phase3.py` 中保留了相关注释代码，但已经明确禁用。

当前原因写在代码注释里：

- 该规则可能会误删本来正确的腔室预测
- 特别是当 myocardium 预测不完整时，会破坏 LV 等结果

所以，当前 Phase3 不会执行 chamber enclosure。

### 5.2 当前没有实现单独的 Vessel Exclusion 规则

早期设计里提到：

- Aorta / PA 与 Myocardium 的互斥修正

但当前 `phase3.py` 没有这段实际逻辑。

### 5.3 当前没有统一的 Global Sanity 小噪点清理步骤

Phase3 当前的“清理”主要依靠：

- priority fusion
- PV / LAA 连接修正
- coronary attachment 过滤
- chamber fragments 清理

并没有再做一轮对所有类别统一的全局小连通域删除。

## 6. 当前 case 处理流程

`phase3.py` 的单 case 处理流程是：

1. 加载 10 类 mask
2. 缺失类别补零
3. 做 priority fusion
4. 执行 `enforce_anatomical_constraints`
5. 保存 `.npy`
6. 保存 `.nii.gz`

当前 case 列表的发现方式是：

- 优先从 `lv_morph/` 枚举
- 如果没有，则回退到 `lv_global/`

## 7. 当前评估脚本行为

评估入口是：

- `repairing/phase3/evaluate_repair_quality.py`

### 7.1 评估对象

当前脚本会逐 case、逐类别比较：

- Phase1 vs GT
- Phase2 vs GT
- Phase3 vs GT

### 7.2 数据来源

脚本会读取：

- GT：`gt_root`
- P1：`data_root/{class}_global/{case_id}_mask.npy`
- P2：`data_root/{class}_morph/{case_id}_mask.npy`
- P3：`data_root/repaired_phase3/{case_id}_phase3.nii.gz`

如果某个 Phase2 文件缺失，当前实现会把：

- `mask_p2 = mask_p1.copy()`

### 7.3 指标

当前实现会计算：

- Dice
- HD95
- ASD
- 预测体积
- GT 体积
- 预测连通域数
- GT 连通域数
- P1->P2 / P2->P3 的体积变化
- Dice 改变量

### 7.4 GT 尺寸处理

如果 GT 不是 `256 x 256 x 256`，当前脚本会：

- 用最近邻 resize 到 `256^3`

这与当前 Phase1 / Phase2 / Phase3 输出都在统一全局尺寸上的假设一致。

### 7.5 输出结果

评估脚本会输出：

- 原始逐 case CSV
- 按类别聚合后的 summary CSV
- 控制台 worst cases 信息

此外，脚本还支持：

- `--skip_hd95`
- `--use_gt_coronary`

## 8. 当前集群入口

当前保留的主集群脚本是：

- `scripts/hpc/phase3/run_phase3_sbatch.sh`
- `scripts/hpc/phase3/run_evaluation_sbatch.sh`

它们分别负责：

- 运行 `phase3.py`
- 运行 `evaluate_repair_quality.py`

因此当前主线已经不是旧文档里的：

- `/repairing/stage3/`
- `stage3.py`
- 单一 `final_segmentation.*`

而是：

- `repairing/phase3/phase3.py`
- `repairing/phase3/evaluate_repair_quality.py`
- `repaired_phase3/{case_id}_phase3.*`

## 9. 当前代码中的几个实际注意点

### 9.1 当前 Phase3 是“少量规则修正”，不是完整规则引擎

当前真正执行的规则数量不多，重点集中在：

- PV / LAA 的连接性
- Coronary 的依附性
- 腔室碎片清理

### 9.2 当前评估脚本是按类别比较 P1 / P2 / P3

它不是只看最终多类图，而是会分别把各阶段结果和 GT 对比，再汇总。

### 9.3 当前文档只描述代码行为

这份说明不再记录：

- 某次融合实验的结果截图
- 某次规则调参前后的数值改善
- 某个数据集版本下的固定路径和统计

如果后续要公开实验结果，建议另写新的结果说明文档，而不是继续把结果混在实现说明里。
