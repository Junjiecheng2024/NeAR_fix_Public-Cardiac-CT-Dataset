# NeAR v2.0 Tier2 - Coronary Artery Refinement

## 项目概述

本项目专注于使用 **NeAR (Neural Annotation Refinement)** 隐式函数模型来修复和细化冠状动脉的分割标注。相比原始 HINTLab 的 NeAR，我们进行了多项改进以适应心脏 CT 数据中细小血管结构的特殊挑战。

---

## 问题与挑战

### 核心问题
冠状动脉在心脏 CT 中是**极度稀疏**的结构：
- **原始体素占比**: 仅 ~0.12%（全扫描中）
- **结构特点**: 树状、细长、边界模糊
- **传统方法痛点**: 标签不平衡严重，边界学习困难

### 解决思路

```
┌─────────────────────────────────────────────────────────────┐
│                    Tier2 策略                                │
├─────────────────────────────────────────────────────────────┤
│  1. 数据层面: 基于冠状动脉 bbox 裁剪 → 提高体素密度           │
│  2. 特征层面: 融合多种信息源 → 增强边界感知                   │
│  3. 采样层面: 边界过采样 50% → 关注细节区域                   │
│  4. 损失层面: Dice + Boundary Dice + Focal → 平衡学习        │
└─────────────────────────────────────────────────────────────┘
```

---

## 模型架构 (Fusion v2.1)

### 特征融合策略

```
输入特征 (328 channels total):
├── Grid position:      3 ch   (x, y, z 坐标)
├── Shape features:   160 ch   (4层 decoder: 64+48+32+16)
├── Multi-scale App:  160 ch   (CNN 编码器: 32+64+64)
├── Raw CT values:      1 ch   ★ 来自 HINTLab 原版
└── 3D Sobel Gradient:  4 ch   ★ 新增: gx, gy, gz, magnitude

MLP 结构:
Input (328) → FC1 (256) → FC2 (256) → Skip → FC3 (128) → FC4 (64) → Output (1)
```

### 与原版 NeAR 对比

| 特性 | HINTLab 原版 | 我们的 v2.0 | **Fusion v2.1** |
|------|-------------|-------------|-----------------|
| 多尺度外观特征 | ❌ | ✅ | ✅ |
| 原始 CT 值 | ✅ | ❌ | ✅ |
| 3D边缘梯度 | ❌ | ❌ | ✅ |
| Context Mask | ❌ | ✅ | ✅ |
| Skip Connection MLP | ❌ | ✅ | ✅ |

---

## 文件结构与说明

### 训练相关

| 文件 | 作用 |
|------|------|
| `train_tier2.py` | **主训练脚本** - PyTorch Lightning Trainer，支持 SLURM 多 GPU |
| `lightning_module_tier2.py` | Lightning Module 定义，包含 forward、training_step、validation_step |
| `run_phase1_sbatch.sh` | SLURM 提交脚本，配置 4×A100 GPU 训练 |

### 配置相关

| 文件 | 作用 |
|------|------|
| `configs/coronary_tier2_fusion.py` | **Fusion 模型配置** - 分辨率 128³，启用梯度+原始CT |

### 模型相关 (位于 `near/models/nn3d/`)

| 文件 | 作用 |
|------|------|
| `model_shape_appearance.py` | 模型定义，包含 `FusionDecoderShapeAppearance`, `SobelGradient3D` |
| `grid.py` | 网格采样工具 `GatherGridsFromVolumes` |
| `blocks.py` | 基础模块 `ConvNormAct`, `LatentCodeUpsample` |

### 数据相关 (位于 `near/datasets/` 和 `data_prepare/`)

| 文件 | 作用 |
|------|------|
| `coronary_tier2_dataset.py` | Dataset 类，加载预处理后的 Tier2 数据 |
| `prepare_coronary_tier2.py` | 数据预处理脚本，裁剪、归一化、保存 |
| `visualize_tier2_sample.py` | 可视化工具，查看预处理结果 |
| `generate_tier2_stats.py` | 统计工具，计算体素占比等指标 |

---

## 数据准备

### 预处理步骤

```bash
# 在 Mahti 集群上运行
cd /projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset

python data_prepare/prepare_coronary_tier2.py \
    --images_dir /path/to/images \
    --labels_dir /path/to/labels \
    --output_dir /scratch/project_2016517/junjie/dataset/coronary_tier2 \
    --margin 5 \
    --target_resolution 256 \
    --n_workers 20
```

### 输出结构

```
coronary_tier2/
├── 1/
│   ├── ct.npy              # 归一化 CT [0,1]
│   ├── mask_coronary.npy   # 冠状动脉二值掩码
│   ├── mask_context.npy    # 上下文掩码 (Myo+Aorta)
│   ├── seg_full.npy        # 完整分割标签
│   └── crop_params.json    # 裁剪参数
├── 2/
│   └── ...
└── 998/
```

### 当前数据统计 (margin=5)

- **样本数**: 998
- **原始体素比例**: 0.12%
- **处理后体素比例**: ~0.51%
- **改善倍数**: ~4.35x
- **上下文覆盖**: 13.39%

---

## 训练

### SLURM 提交

```bash
cd /projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset
sbatch repairing/phase1/run_phase1_sbatch.sh
```

### 脚本配置

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --time=36:00:00

srun --gpu-bind=closest python train_tier2.py \
    --config configs/coronary_tier2_fusion.py \
    --devices auto \
    --strategy auto
```

### 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 分辨率 | 128³ | 避免 OOM (256³ 太大) |
| 学习率 | 5e-4 | 配合 cosine schedule |
| Epochs | 600 | 较长训练周期 |
| Batch Size | 1 | 单样本 + 梯度累积 4 |
| 边界采样比例 | 50% | 增强边界学习 |

### 损失函数权重

- Dice Loss: 0.55
- Boundary Dice: 0.25
- Focal Loss: 0.15
- L2 Regularization: 1e-4

---

## 监控与可视化

### WandB

训练日志自动同步到 WandB：
- Project: `NeAR_v2_Tier2_Coronary`
- Metrics: dice_score, boundary_dice, focal_loss

### 本地可视化

```bash
python data_prepare/visualize_tier2_sample.py \
    --sample_dir /path/to/coronary_tier2/1 \
    --save output.png
```

---

## 开发历程

1. **问题识别**: 原始体素占比仅 0.12%，传统训练效果差
2. **Tier2 策略**: 裁剪到冠状动脉 bbox，提高体素密度
3. **特征融合**: 结合 HINTLab 的原始 CT + 我们的多尺度编码器
4. **3D 梯度**: 添加 Sobel 边缘检测，增强边界感知
5. **多 GPU 调试**: 解决 SLURM + Lightning + DDP 的配置问题

---

## TODO

- [ ] 验证多 GPU 训练正常工作
- [ ] 完成 600 epochs 训练
- [ ] 评估修复效果
- [ ] 实现 inference 脚本
- [ ] 将结果映射回全局坐标系
