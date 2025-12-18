# NeAR v2.0 Phase1 - Shape + Appearance Training

本目录包含 NeAR v2.0 Phase1 训练代码，使用 **Shape + Appearance** 模型对 Coronary 等小结构进行分割。

## 目录结构

```
phase1/
├── config.py              # 统一配置（使用 dataclass）
├── train.py               # 主训练脚本
├── inference.py           # 推理脚本
├── lightning_module.py    # PyTorch Lightning 模块
├── map_tier2_to_global.py # 坐标映射（Tier2 → 全局）
├── README.md
└── scripts/
    ├── run_sbatch.sh      # SLURM 提交脚本
    └── run_local.sh       # 本地运行脚本
```

## 集群环境路径

| 类型 | 路径 |
|------|------|
| 容器镜像 | `/scratch/project_2016517/JunjieCheng/pytorch.sif` |
| 数据集 | `/scratch/project_2016517/JunjieCheng/dataset/coronary_tier2` |
| 项目代码 | `/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset` |
| 输出目录 | `/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1` |

## 数据格式

每个样本文件夹包含：
```
sample_xxx/
├── crop_params.json    # 裁剪参数（用于映射回全局）
├── ct.npy              # CT 体积 (D, H, W)
├── mask_coronary.npy   # Coronary 分割掩码
├── mask_context.npy    # 上下文掩码（Myo + Aorta）
└── seg_full.npy        # 完整分割
```

## 快速开始

### 1. SLURM 提交

```bash
cd /projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset
sbatch repairing/phase1/scripts/run_sbatch.sh
```

### 2. 本地调试

```bash
cd repairing/phase1
python train.py --config config.py --devices 1
```

## 配置系统

使用 Python dataclass 继承：

```python
from config import CoronaryConfig, get_config

# 方式1：直接使用
cfg = CoronaryConfig()

# 方式2：通过名称获取
cfg = get_config("coronary")
```

支持的类别：`coronary`, `aorta`, `myocardium`, `la`, `lv`, `ra`, `rv`, `pa`, `pv`, `laa`

## 模型架构

```
Shape + Appearance Model
├── Latent Embedding (per-sample)
├── Appearance Encoder (CT 特征提取)
├── Context Encoder (可选，Myo+Aorta 掩码)
└── Implicit Decoder (融合解码)
```

## 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_epochs` | 600 | 训练轮数 |
| `batch_size` | 1 | 每 GPU 批量 |
| `gradient_accumulation_steps` | 4 | 梯度累积 |
| `lr` | 5e-4 | 学习率 |
| `dice_weight` | 0.55 | Dice Loss 权重 |
| `boundary_dice_weight` | 0.25 | 边界 Dice 权重 |
| `focal_weight` | 0.15 | Focal Loss 权重 |

## 训练监控

使用 WandB 监控，项目名：`NeAR_v2_Tier2_{class_name}`

查看日志：
```bash
tail -f /scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/logs/{job_id}.out
```

## 推理

```bash
python inference.py \
    --config config.py \
    --checkpoint /path/to/best.ckpt \
    --output_dir /path/to/output
```
