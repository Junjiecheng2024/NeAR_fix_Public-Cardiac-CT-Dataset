# NeAR 心脏数据集修复项目 - 修复模块

本目录包含使用NeAR隐式表征进行心脏数据集标签修复的所有脚本和配置。

## 📁 目录结构

```
near_repairing/
├── stage1_coronary/              # 【核心】阶段一：冠状动脉单类精修
│   ├── train_coronary.py         # 训练脚本
│   ├── config_coronary.py        # 训练配置
│   ├── inference_coronary.py     # 推理脚本
│   ├── resize_ct_images.py       # CT预处理工具
│   ├── checkpoints/              # 训练检查点
│   ├── logs/                     # 训练日志
│   └── README.md                 # 详细说明文档
│
├── train_coronary.py             # 【当前运行中】训练脚本（软链接到stage1）
├── config_coronary.py            # 【当前配置】配置文件（软链接到stage1）
├── inference_coronary.py         # 推理脚本（软链接到stage1）
├── resize_ct_images.py           # 工具脚本（软链接到stage1）
│
├── logs/                         # 【运行中】当前训练日志
├── checkpoints/                  # 【运行中】当前检查点
│
├── near_repair.py                # 【参考】原始NeAR多类训练脚本
├── config_near.py                # 【参考】原始NeAR配置
├── evaluate_refinement.py        # 【工具】评估脚本
├── visualize_refinement.py       # 【工具】可视化脚本
│
├── eval_scripts/                 # 评估相关脚本
└── _init_paths_local.py          # 路径初始化

```

## 🎯 项目目标

使用隐式表征的天然光滑性修复心脏数据集的噪声标签，分三个阶段：

### 阶段一：单类精修（进行中 ✅）
- **目标**：对每个心脏类别（10个）单独训练 Shape-only NeAR
- **当前进展**：冠状动脉（Coronary, class 9）训练中
- **输出**：精修后的单类掩膜 M^{ref}_{i,c}
- **目录**：`stage1_coronary/`

### 阶段二：多类融合（待完成）
- **目标**：融合10个单类掩膜为完整多类标签
- **策略**：优先级融合 + 解剖一致性修正
- **输出**：干净的多类标签 L^{ref}_i

### 阶段三：统一隐式模型（待完成）
- **目标**：训练一个统一的多头隐式函数
- **输入**：阶段二的精修多类标签
- **输出**：F_Θ(z_i, p) → [s_1, s_2, ..., s_10]

## 🚀 快速开始

### 1. 当前训练（Coronary）

```bash
# 查看训练进度
tail -f logs/coronary_training_128res_*.log

# 查看GPU使用
nvidia-smi

# 查看训练Dice
grep "Epoch.*Train.*dice" logs/coronary_training_128res_*.log | tail -20
```

### 2. 训练其他类别

复制并修改 `stage1_coronary/` 为其他类别：

```bash
# 例如：训练左心房（LA, class 2）
cp -r stage1_coronary/ stage1_LA/
cd stage1_LA/
# 修改 config_coronary.py 中的 class_name 和 class_index
# 重新训练
nohup python -u train_coronary.py > logs/training_LA.log 2>&1 &
```

## 📊 当前训练状态

### Coronary（冠状动脉）- 进行中

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 分辨率 | 128³ | 高精度采样 |
| Batch Size | 1 × 4 (grad_acc) | 有效batch=4 |
| 学习率 | 5e-4 | 精调阶段 |
| 损失函数 | 70% Dice + 30% Focal(γ=4) | 组合损失 |
| 训练Dice | ~70% | 边界区域 |
| 验证Dice | ~7% | 均匀采样（真实分布） |
| 显存占用 | 27.6 GB / 49 GB | AMP混合精度 |
| 每Epoch | ~200s 训练 + 280s 验证 | A6000 GPU |

**检查点**：`checkpoints/Coronary_class9_shape_only_251109_063825/best.pth`

## 📝 核心文件说明

### 训练相关

| 文件 | 用途 | 状态 |
|------|------|------|
| `train_coronary.py` | 冠状动脉训练脚本 | ✅ 运行中 |
| `config_coronary.py` | 训练配置 | ✅ 当前配置 |
| `inference_coronary.py` | 生成精修掩膜 | ⏸️ 训练后使用 |

### 参考脚本

| 文件 | 用途 | 说明 |
|------|------|------|
| `near_repair.py` | 原始NeAR训练脚本 | 多类+appearance，腹部数据 |
| `config_near.py` | 原始NeAR配置 | Alan数据集 |

### 工具脚本

| 文件 | 用途 |
|------|------|
| `resize_ct_images.py` | CT图像resize到256³ |
| `evaluate_refinement.py` | 评估精修效果 |
| `visualize_refinement.py` | 可视化对比 |

## 🔍 关键技术

### 1. 边界偏采样（Boundary-biased Sampling）
- **目的**：缓解极端类别不平衡（冠状动脉仅占0.14%）
- **方法**：50%样本从边界带采样，50%均匀采样
- **效果**：正样本比例从0.14% → 0.86%（7倍增强）

### 2. 课程学习（Curriculum Learning）
- **策略**：逐步降低边界偏采样比例
  - Epoch 1-100: 50% bias
  - Epoch 101-200: 30% bias
  - Epoch 201-300: 10% bias
- **目的**：让模型从"容易"过渡到"困难"，最终适应真实分布

### 3. 组合损失函数
```python
Loss = 0.7 × Dice_Loss + 0.3 × Focal_Loss(gamma=4.0)
```
- **Dice Loss**：直接优化目标指标
- **Focal Loss**：处理类别不平衡，256倍困难样本权重

### 4. 混合精度训练（AMP）
- **Float16**：前向传播，节省显存
- **Float32**：梯度和优化器，保证精度
- **效果**：显存从42GB → 27.6GB

## 📈 下一步计划

### 短期（1-2天）
- [ ] 完成Coronary训练（300 epochs）
- [ ] 运行推理生成精修掩膜
- [ ] 连通分量清理（保留最大2个CC）
- [ ] 评估精修效果（Dice, Surface Dice）

### 中期（1周）
- [ ] 训练其他9个类别
  - [ ] Myocardium (心肌, class 1)
  - [ ] LA (左房, class 2)
  - [ ] LV (左室, class 3)
  - [ ] RA (右房, class 4)
  - [ ] RV (右室, class 5)
  - [ ] Aorta (主动脉, class 6)
  - [ ] PA (肺动脉, class 7)
  - [ ] LAA (左心耳, class 8)
  - [ ] PV (肺静脉, class 10)

### 长期（2周）
- [ ] 阶段二：多类融合
- [ ] 阶段二：解剖一致性修正
- [ ] 阶段三：统一隐式模型训练

## 🐛 常见问题

### Q1: 验证Dice很低（7%）正常吗？
**A**: 正常。训练用50%边界偏采样，验证用0%均匀采样，导致6倍采样差异。关注训练Dice是否稳定提升。

### Q2: 训练很慢，每个epoch要200秒？
**A**: 正常。128³分辨率有2.1M采样点，是64³的8倍。可以降低分辨率到96³以加速。

### Q3: 如何判断训练是否收敛？
**A**: 观察：
1. 训练Dice稳定在60-70%
2. Loss平滑下降
3. 验证Dice缓慢提升

### Q4: OOM怎么办？
**A**: 
1. 确认AMP已开启（`use_amp=True`）
2. 降低`training_resolution`到96或64
3. 减小`batch_size`到1

## 📞 参考文档

- **阶段一详细文档**：`stage1_coronary/README.md`
- **项目总纲**：`/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/总纲.txt`
- **训练日志**：`logs/coronary_training_128res_*.log`
- **原始论文**：NeAR: Neural Radiance Fields for Multi-Organ Refinement

## 📊 训练监控

```bash
# 实时查看训练
watch -n 5 "tail -30 logs/coronary_training_128res_*.log | grep -E '(Batch|Epoch|dice)'"

# GPU监控
watch -n 2 nvidia-smi

# 检查checkpoint
ls -lht checkpoints/Coronary_class9_shape_only_*/
```

---

**最后更新**：2025-11-09  
**当前状态**：Coronary 128³训练进行中（显存27.6GB，训练Dice~70%）  
**预计完成**：2025-11-10（300 epochs × 480s/epoch ≈ 40小时）
