# NeAR Phase1 Coronary Training Pipeline

## 技术报告 - 冠状动脉分割模型训练

**项目**: NeAR v2.0 Phase1 Coronary Tier2  
**日期**: 2025-12-20  
**验证结果**: Dice Score ≈ 0.89

---

## 1. 项目概述

### 1.1 目标
训练一个神经隐式表示（Neural Implicit Representation）模型，学习冠状动脉的形态先验，用于**修复/补全**现有的粗糙分割标注。

### 1.2 核心方法: NeAR (Neural Annotation Refinement)
NeAR 使用**形状编码 + 外观特征**的隐式表示方法，通过学习数据集中所有样本的形状分布，自动纠正标注中的断裂、缺失和噪声。

### 1.3 关键创新点
| 创新 | 描述 |
|------|------|
| Shape + Appearance 融合 | 结合 CT 外观特征与形状先验 |
| Context Mask 引导 | 使用心肌+主动脉作为解剖上下文 |
| 多损失函数组合 | 5种损失函数专为小结构优化 |
| 边界偏置采样 | 提高边界区域学习效率 |

---

## 2. 数据准备

### 2.1 数据集结构
```
coronary_tier2/
├── case_001/
│   ├── ct.npy              # CT 图像 (D, H, W), 归一化到 [0,1]
│   ├── mask_coronary.npy   # 冠状动脉二值掩码
│   ├── mask_context.npy    # 上下文掩码 (心肌 + 主动脉)
│   └── crop_params.json    # 裁剪参数 (用于坐标映射)
├── case_002/
│   ...
└── (共 998 个样本)
```

### 2.2 数据预处理
- **输入分辨率**: 128 × 128 × 128 (各向同性)
- **CT 归一化**: Min-Max 到 [0, 1]
- **坐标系**: 标准化到 [-1, 1]

### 2.3 数据增强
```python
# 在线增强 (augment=True)
- 随机 3D 翻转 (沿各轴 50% 概率)
- 随机 90° 旋转 (xyz 平面)
- 随机转置 (轴交换)
- 亮度/对比度扰动 (±10%)
- 高斯噪声 (σ=0.02)
- Gamma 校正 (γ ∈ [0.8, 1.2])
```

---

## 3. 模型架构

### 3.1 整体架构: EmbeddingDecoderShapeAppearanceWithContext

```
┌─────────────────────────────────────────────────────────────────┐
│                         输入                                     │
├─────────────────────────────────────────────────────────────────┤
│  • CT 图像: (B, 1, 128, 128, 128)                                │
│  • Context Mask: (B, 1, 128, 128, 128) - 心肌+主动脉             │
│  • Sample Index: (B,) - 用于查询形状嵌入                         │
│  • Query Grid: (B, N, 3) - 采样点坐标 [-1, 1]                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              AppearanceEncoder (CT 特征提取)                     │
├─────────────────────────────────────────────────────────────────┤
│  Level 1: Conv3D(1→32) → (B, 32, 128, 128, 128)                 │
│  Level 2: Downsample + Conv3D(32→64) → (B, 64, 64, 64, 64)      │
│  Level 3: Downsample + Conv3D(64→64) → (B, 64, 32, 32, 32)      │
│                                                                  │
│  输出: 多尺度特征金字塔 {f1, f2, f3}                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Shape Embedding (形状先验)                          │
├─────────────────────────────────────────────────────────────────┤
│  Embedding Table: (998 samples × 256 dim)                        │
│  每个样本有独立的可学习形状嵌入向量                               │
│                                                                  │
│  latent = embedding[sample_index]  # (B, 256)                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         LatentCodeUpsample (形状嵌入上采样)                      │
├─────────────────────────────────────────────────────────────────┤
│  (B, 256) → (B, 64, 8, 8, 8) → (B, 48, 16, 16, 16)              │
│          → (B, 32, 32, 32, 32) → (B, 16, 64, 64, 64)            │
│                                                                  │
│  通过转置卷积逐步上采样形状嵌入                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           ImplicitDecoder (隐式解码器)                           │
├─────────────────────────────────────────────────────────────────┤
│  对于每个查询点 (x, y, z):                                       │
│  1. 从上采样的形状特征采样 → shape_feat                          │
│  2. 从 CT 外观特征采样 → appearance_feat                         │
│  3. 从 Context 掩码采样 → context_feat                           │
│  4. 特征拼接 → MLP → 占用概率 σ(logit)                          │
│                                                                  │
│  输出: (B, N, 1) 每个查询点的占用概率                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         输出                                     │
├─────────────────────────────────────────────────────────────────┤
│  • pred_logit: (B, N, 1) - 预测 logit                           │
│  • pred_prob: sigmoid(pred_logit) - 预测概率                     │
│  • encoded: (B, 256) - 形状嵌入 (用于 L2 正则化)                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 模型参数量
- **总参数**: ~2.5M
- **AppearanceEncoder**: ~0.8M
- **LatentCodeUpsample**: ~0.5M
- **ImplicitDecoder**: ~0.7M
- **Embedding Table**: 998 × 256 = 0.25M

---

## 4. 损失函数设计

### 4.1 多损失组合策略

针对冠状动脉的**极端类别不平衡**（前景 < 1%）和**细长拓扑结构**，采用 5 种损失函数组合：

```python
Total Loss = 0.30 × Dice Loss
           + 0.35 × Tversky Loss    # 强调 Recall
           + 0.20 × Boundary Dice   # 关注边界
           + 0.10 × Focal Loss      # 处理类别不平衡
           + 0.05 × TopK Loss       # 困难样本挖掘
           + 1e-4 × L2 Regularization
```

### 4.2 各损失函数详解

#### Dice Loss (权重: 0.30)
```python
Dice = 2 × |P ∩ T| / (|P| + |T|)
Loss = 1 - Dice
```
- **作用**: 衡量整体区域重叠度
- **优点**: 对类别不平衡相对鲁棒

#### Tversky Loss (权重: 0.35) ⭐ 核心
```python
Tversky = TP / (TP + α×FP + β×FN)
# 设置 α=0.2, β=0.8 → 强烈惩罚漏检 (FN)
```
- **作用**: 可调节 Precision/Recall 权衡
- **关键**: α < β 强调 Recall，减少冠状动脉漏检

#### Boundary Dice Loss (权重: 0.20)
```python
# 仅在边界区域计算 Dice
boundary = dilate(mask) - erode(mask)  # 宽度=2 voxels
loss = 1 - Dice(pred × boundary, target × boundary)
```
- **作用**: 强化边界精度
- **实现**: 通过 MaxPool3D 实现可微分的膨胀/腐蚀

#### Focal Loss (权重: 0.10)
```python
FL = -α × (1-p)^γ × log(p)
# 设置 α=0.25, γ=4.0
```
- **作用**: 下调简单样本权重，聚焦困难样本
- **特点**: γ=4.0 比常规更激进

#### TopK Loss (权重: 0.05)
```python
# 只对 loss 最高的 10% 体素计算
```
- **作用**: 困难样本挖掘
- **场景**: 处理边界模糊区域

---

## 5. 训练策略

### 5.1 采样策略: 边界偏置采样

```python
# 训练时: 50% 采样点来自边界附近
boundary_bias_ratio = 0.5
boundary_dilation_radius = 3  # voxels

# 验证时: 100% 均匀采样 (公平评估)
boundary_bias_ratio = 0.0
```

**原理**: 冠状动脉体积极小（< 1%），均匀采样会导致大量无效计算。边界偏置采样提高训练效率。

### 5.2 训练配置

| 参数 | 值 |
|------|-----|
| Epochs | 600 |
| Batch Size | 1 |
| Gradient Accumulation | 4 步 |
| 有效 Batch Size | 4 |
| Learning Rate | 5e-4 |
| Optimizer | AdamW (weight_decay=1e-5) |
| Scheduler | Cosine Annealing + 2% Warmup |
| Precision | FP16 混合精度 |
| 分布式训练 | DDP (4× A100 GPU) |

### 5.3 训练时间
- **硬件**: 4× NVIDIA A100 40GB
- **每 Epoch 时间**: ~7 分钟
- **总训练时间**: ~70 小时

---

## 6. 推理流程

### 6.1 单样本推理

```python
# 1. 加载模型权重
model.load_state_dict(checkpoint['state_dict'])

# 2. 准备输入
appearance = ct_volume.unsqueeze(0)      # (1, 1, 128, 128, 128)
context = context_mask.unsqueeze(0)      # (1, 1, 128, 128, 128)
index = torch.tensor([sample_idx])       # 样本索引

# 3. 创建全分辨率网格
grid = create_full_grid((128, 128, 128)) # (1, 128³, 3)

# 4. 推理
pred_logit, _ = model(index, grid, appearance, context)
pred_prob = torch.sigmoid(pred_logit)
pred_mask = (pred_prob > 0.5).float()
```

### 6.2 滑动窗口推理（大体积）
对于超过 128³ 的体积，采用重叠滑动窗口策略，chunk_size=64, overlap=16。

---

## 7. 结果验证

### 7.1 定量指标

| 指标 | 训练集 | 验证集 |
|------|--------|--------|
| Dice Score | ~0.89 | ~0.89 |
| 正样本体素 | ~10,000 | ~10,000 |
| 预测体素 | ~12,000 | ~11,000 |

### 7.2 定性分析

通过 3D Slicer 验证：
- ✅ 预测 mask 贴合 CT 血管解剖结构
- ✅ 血管连续性良好
- ✅ 分支结构保留完整
- ⚠️ 部分末梢分支略有过分割

---

## 8. 文件结构

```
repairing/phase1/
├── config.py              # 配置类定义
├── train.py               # 训练主脚本
├── lightning_module.py    # PyTorch Lightning 模块
├── inference.py           # 批量推理
├── visualize_sample.py    # 2D 切片可视化
├── visualize_3d.py        # 3D 表面可视化
├── export_nifti.py        # NIfTI 导出 (3D Slicer)
└── scripts/
    └── run_sbatch.sh      # SLURM 提交脚本
```

---

## 9. 后续工作

1. **全量推理**: 对 998 样本全部推理，生成修复后的 mask
2. **质量筛选**: 设置 Dice 阈值，筛选低质量样本人工检查
3. **Phase2 训练**: 使用修复后的数据训练最终分割模型
4. **泛化测试**: 在无标注的新数据上测试泛化能力

---

## 10. 关键代码路径

| 模块 | 路径 |
|------|------|
| 模型定义 | `near/models/nn3d/model_shape_appearance.py` |
| 损失函数 | `near/models/losses.py` |
| 数据集 | `near/datasets/coronary_tier2_dataset.py` |
| 训练模块 | `repairing/phase1/lightning_module.py` |
| 配置 | `repairing/phase1/config.py` |

---

*生成时间: 2025-12-20*
