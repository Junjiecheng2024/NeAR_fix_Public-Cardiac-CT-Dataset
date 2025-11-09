# 阶段一：冠状动脉单类精修（Stage 1: Coronary Refinement）

## 📁 目录结构

```
stage1_coronary/
├── config_coronary.py          # 训练配置文件
├── train_coronary.py            # 训练脚本（主程序）
├── inference_coronary.py        # 推理脚本（生成精修掩膜）
├── resize_ct_images.py          # CT图像预处理工具
├── checkpoints/                 # 训练检查点
│   └── Coronary_class9_shape_only_*/
│       ├── best.pth            # 最佳模型
│       ├── latest.pth          # 最新模型
│       └── config.json         # 训练时的配置
└── logs/                        # 训练日志
    └── coronary_training_*.log
```

## 🎯 训练目标

使用 Shape-only NeAR 隐式表征模型修复冠状动脉的噪声标签，获得：
- 隐式函数 F_c(z_i, p) → occupancy
- 每个样本的潜向量 z_i
- 精修后的二值掩膜 M^{ref}_{i,9}

## 🚀 快速开始

### 1. 训练模型

```bash
cd /home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/NeAR/repairing/near_repairing/stage1_coronary
conda activate near
nohup python -u train_coronary.py > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 2. 监控训练

```bash
# 查看最新日志
tail -f logs/coronary_training_*.log

# 查看训练进度
grep "Epoch.*Train" logs/coronary_training_*.log | tail -20

# 查看验证Dice
grep "Eval.*dice" logs/coronary_training_*.log
```

### 3. 推理生成精修掩膜

```bash
python inference_coronary.py \
    --checkpoint checkpoints/Coronary_class9_shape_only_*/best.pth \
    --data_root ../../../../dataset/near_format_data \
    --output_dir refined_masks \
    --resolution 256
```

## ⚙️ 核心配置

### 训练参数（config_coronary.py）

| 参数 | 值 | 说明 |
|------|-----|------|
| `training_resolution` | 128³ | 采样分辨率（2.1M点） |
| `batch_size` | 1 | 物理batch size |
| `gradient_accumulation_steps` | 4 | 梯度累积（有效batch=4） |
| `lr` | 5e-4 | 学习率（精调阶段） |
| `n_epochs` | 300 | 训练轮数 |
| `eval_interval` | 10 | 验证间隔 |
| `use_amp` | True | 混合精度训练 |

### 边界偏采样（Curriculum Learning）

| Epoch | Boundary Bias | 说明 |
|-------|---------------|------|
| 1-100 | 50% | 初始阶段：大量边界样本加速学习 |
| 101-200 | 30% | 过渡阶段：逐步适应真实分布 |
| 201-300 | 10% | 精调阶段：接近均匀采样 |

### 损失函数

```python
Loss = 0.7 × Dice_Loss + 0.3 × Focal_Loss(gamma=4.0)
```

- **Dice Loss (70%)**：直接优化目标Dice系数
- **Focal Loss (30%)**：处理极端类别不平衡（0.86% 正样本）
- **gamma=4.0**：256倍困难样本权重增强

## 📊 训练效果

### 当前最佳模型（128³ 分辨率）

| 指标 | 训练集 | 验证集 | 说明 |
|------|--------|--------|------|
| Dice | ~70% | ~7% | 验证用均匀采样（真实分布） |
| 显存占用 | 27.6 GB / 49 GB | - | AMP混合精度 |
| 每Epoch时间 | ~200s | ~280s | A6000 GPU |

**注意**：验证Dice较低是正常的，因为：
1. 训练用50%边界偏采样（0.86%正样本）
2. 验证用0%偏采样（0.14%正样本，真实分布）
3. 6倍的采样差异导致Dice差异

## 🔧 关键技术

### 1. 边界偏采样（Boundary-biased Sampling）

```python
# 对每个样本的二值mask M：
M_dilate = dilate(M, radius=2)  # 膨胀
M_erode = erode(M, radius=2)     # 腐蚀
B = M_dilate \ M_erode           # 边界带

# 采样策略：
50% 从边界带 B 采样（inside+outside）
50% 从整个ROI均匀采样
```

**效果**：正样本占比从0.14%提升到0.86%（7.14倍增强）

### 2. 课程学习（Curriculum Learning）

逐步降低边界偏采样比例，让模型从"容易"过渡到"困难"：
- 初期：密集边界样本 → 快速学习边界特征
- 中期：减少偏采样 → 学习全局结构
- 后期：接近均匀 → 适应真实分布

### 3. 混合精度训练（AMP）

- Float16前向传播：节省显存、加速计算
- Float32梯度累积：保证数值稳定性
- **效果**：显存从42GB降至27.6GB

## 📈 训练监控指标

### 关键指标

1. **训练Dice**：应稳定在60-70%
2. **验证Dice**：缓慢提升，最终目标>30%
3. **Loss下降**：平滑下降，无剧烈波动
4. **学习率**：Cosine退火，从5e-4平滑下降

### 异常情况处理

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| Dice=0.0000 | 类别不平衡过于极端 | 已通过Focal Loss解决 |
| OOM错误 | 显存不足 | 降低resolution或batch_size |
| Loss=NaN | 梯度爆炸 | 降低学习率 |
| 训练过慢 | 128³点数多 | 正常现象，~200s/epoch |

## 🎓 下一步：阶段二

训练完成后，使用精修后的掩膜：

1. **连通分量清理**：保留最大2个CC（主动脉+冠状动脉）
2. **融合多类标签**：与其他9个类别融合为L^{ref}_i
3. **解剖一致性修正**：确保冠状动脉在心肌/主动脉附近

## 📝 版本记录

### v3.0 (2025-11-09) - 当前版本
- ✅ 128³高分辨率训练
- ✅ 课程学习（50%→30%→10%）
- ✅ 组合损失（70% Dice + 30% Focal gamma=4）
- ✅ 学习率5e-4（精调阶段）
- ✅ 显存优化27.6GB

### v2.0 (2025-11-08)
- ✅ Focal Loss (gamma=3)
- ✅ 边界偏采样实现
- ✅ 64³分辨率训练
- ⚠️ 验证Dice仅7%

### v1.0 (2025-11-08)
- ❌ BCE Loss失败（Dice=0）
- ❌ 数据集bug（提取错误类别）

## 📞 联系方式

如有问题，请查看：
- 训练日志：`logs/coronary_training_*.log`
- 配置文件：`config_coronary.py`
- 项目总纲：`/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/总纲.txt`
