# Phase1 当前代码说明

这份文档按当前仓库中的 Phase1 主线代码整理，描述的是实际实现，而不是早期设计方案或实验总结。

当前主线对应的核心文件是：

- `repairing/phase1/config.py`
- `repairing/phase1/train.py`
- `repairing/phase1/lightning_module.py`
- `repairing/phase1/inference.py`
- `near/datasets/coronary_tier2_dataset.py`
- `near/models/nn3d/model_shape_appearance.py`
- `near/models/losses.py`

## 1. 整体定位

Phase1 的目标是对单个类别做隐式重建与修复。当前实现使用的是：

- `Shape + Appearance + Context`
- 每个样本一个独立的 latent embedding
- 在查询网格上预测 occupancy / mask

这套实现不是早期的 `shape_only` 流程。当前训练和主推理都依赖：

- `EmbeddingDecoderShapeAppearanceWithContext`

## 2. 输入数据格式

当前数据集类 `near/datasets/coronary_tier2_dataset.py` 实际读取的是每个 case 目录下的以下文件：

- `ct.npy`
- `mask_target.npy`
- `mask_context.npy`
- `crop_params.json`

其中：

- `ct.npy` 是裁剪后的 CT 体积
- `mask_target.npy` 是当前类别的二值目标 mask
- `mask_context.npy` 是上下文 mask
- `crop_params.json` 用于把 crop 空间预测映射回全局坐标

注意：

- 当前数据集类不要求类别文件名叫 `mask_coronary.npy`
- 当前主线代码是按通用 `mask_target.npy` 读取的
- 数据准备脚本可能还会保存 `seg_full.npy`，但 Phase1 核心训练/推理并不直接读取它

## 3. 数据集类的实际行为

`near/datasets/coronary_tier2_dataset.py` 的主要逻辑如下：

### 3.1 case 发现方式

- 遍历 `root/`
- 只把包含 `mask_target.npy` 的子目录视为有效样本

### 3.2 返回字段

`__getitem__` 返回：

- `index`
- `shape`
- `context`
- `case_id`
- `appearance`（当 `use_appearance=True` 时）

这里的 `index` 很重要，因为模型会直接用它去查 embedding table。

### 3.3 分辨率处理

- 如果配置里提供 `resolution`，则会把 `mask`、`context`、`ct` resize 到立方体分辨率
- `mask/context` 用最近邻
- `ct` 用三次插值

### 3.4 数据增强

当 `augment=True` 时，当前实现会做：

- 按轴随机翻转
- `xy` 平面随机 90 度旋转
- 随机轴转置
- CT 亮度扰动
- CT 对比度扰动
- CT 高斯噪声
- CT gamma 校正

### 3.5 训练/验证数据的关系

当前 `train.py` 里的 `Tier2DataModule` 会创建两份数据集：

- 训练集：开启 augmentation，开启 boundary-biased sampling
- 验证集：关闭 augmentation，关闭 boundary-biased sampling

但两者来自同一个数据根目录，不是独立验证集。

## 4. 当前模型结构

当前主模型定义在 `near/models/nn3d/model_shape_appearance.py` 中，对应类：

- `EmbeddingDecoderShapeAppearanceWithContext`

### 4.1 输入

模型前向实际接收：

- `indices`
- `grid`
- `appearance`
- `context`

其中：

- `indices` 用于查每个样本独立的 embedding
- `grid` 是归一化到 `[-1, 1]` 的 3D 查询坐标
- `appearance` 是 CT
- `context` 是上下文 mask

### 4.2 shape 分支

shape 分支的核心是：

- `nn.Embedding(n_samples, latent_dimension)`
- 多层上采样解码器 `decoder_1 ~ decoder_4`

流程是：

1. 用 `indices` 查出样本 embedding
2. 把 embedding reshape 成低分辨率张量
3. 逐层上采样得到 4 级 shape feature map
4. 在 `grid` 上用 `grid_sample` 取样

### 4.3 appearance + context 分支

当前实现不是“直接在 query point 上采样 context mask”，而是：

1. 先用一个小的 `context_encoder` 对 context mask 编码
2. 把 `context_feat` 与原始 CT 在通道维拼接
3. 再送入 `AppearanceEncoder`
4. 从多尺度 appearance feature 上按 `grid` 取样

因此，当前的 context 是通过 appearance 分支融合进来的。

### 4.4 MLP 解码

shape feature 和 appearance feature 会与坐标一起拼接，然后经过：

- `fc1`
- `fc2`
- skip connection
- `fc3`
- `fc4`
- `output`

最终输出 1 通道 logit。

## 5. 当前训练逻辑

训练入口是 `repairing/phase1/train.py`。

### 5.1 配置加载

`train.py` 支持：

- `--config`
- `--class_name`

如果传了 `--class_name`，会调用 `config.py` 中的 `get_config(class_name)` 取对应类别配置。

如果不传，则回退到 `config.py` 里的默认 `cfg`。

### 5.2 当前配置文件

`repairing/phase1/config.py` 中定义了 10 个类别的 dataclass 配置。

公共配置包括：

- `model_type = "shape_appearance"`
- `use_appearance = True`
- `use_context = True`
- `latent_dimension = 256`
- `target_resolution = 128`
- `batch_size = 1`
- `gradient_accumulation_steps = 4`
- `lr = 5e-4`

不同类别主要在这些方面不同：

- `data_path`
- `run_flag`
- `class_name`
- `class_index`
- `n_epochs`
- 少量 loss 权重

### 5.3 LightningModule 中的损失函数

当前 `repairing/phase1/lightning_module.py` 实际使用的损失项是：

- Dice loss
- Tversky loss
- Boundary Dice loss
- Focal loss
- TopK loss
- latent L2 penalty

默认权重来自配置：

- `dice_weight = 0.3`
- `tversky_weight = 0.35`
- `boundary_dice_weight = 0.2`
- `focal_weight = 0.1`
- `topk_weight = 0.05`
- `l2_penalty_weight = 1e-4`

损失函数的具体实现位于 `near/models/losses.py`。

### 5.4 采样策略

训练时，采样策略依赖 `GatherGridsFromVolumes`，并使用：

- `sampling_bias_ratio = 0.5`
- `sampling_dilation_radius = 3`

验证时会把 `boundary_bias_ratio` 设为 0。

### 5.5 优化器与调度器

当前实现使用：

- `AdamW`
- 默认 `weight_decay=1e-5`

如果 `use_cosine_schedule=True`，则使用：

- warmup + cosine schedule

否则回退到：

- `MultiStepLR`

### 5.6 训练基础设施

当前 `train.py` 使用 PyTorch Lightning 的 `Trainer`，支持：

- 单卡
- 多卡 DDP
- mixed precision

当前代码里还保留了 WandB 登录和记录逻辑。

## 6. 当前推理逻辑

推理入口是 `repairing/phase1/inference.py`。

### 6.1 配置加载方式

和 `train.py` 不同，当前 `inference.py` 只会读取配置文件里的默认 `cfg`：

- `load_config(path)` 返回 `cfg_module.cfg`

它不会像 `train.py` 一样通过 `--class_name` 调 `get_config()`。

这意味着：

- 推理类别通常由传入的配置文件默认值决定
- 当前外层 HPC 脚本需要确保 `config.py` 默认配置和实际要推理的类别一致，或者脚本本身进一步改造

### 6.2 推理输入

推理时会：

1. 构造 `CoronaryTier2Dataset`
2. 关闭 augmentation
3. 关闭 boundary bias
4. 构造 `EmbeddingDecoderShapeAppearanceWithContext`
5. 从 checkpoint 恢复权重

### 6.3 网格推理

`inference.py` 支持两种方式：

- 全体积推理
- sliding window 推理

但当前保留的 HPC 主脚本 `scripts/hpc/phase1/run_inference_sbatch.sh` 明确传了：

- `--no_sliding_window`

因此当前主线上更接近“全体积推理 + 映射回全局”的用法。

### 6.4 全局坐标映射

如果样本有 `crop_params.json`，`inference.py` 会：

1. 先把 crop 空间预测还原到原始体积尺寸
2. 再根据 `--global_shape` resize 到统一全局尺寸

映射逻辑在 `map_to_global()` 中实现。

### 6.5 输出格式

如果能拿到 `crop_params`，输出是：

- `output_dir/{case_id}_mask.npy`

如果没有 `crop_params`，则退化成 crop 空间输出：

- `output_dir/{case_id}/pred_mask.npy`

当前主线 Phase2/Phase3 依赖的是前一种全局输出格式。

## 7. 当前脚本关系

当前 Phase1 相关脚本路径已经整理过：

- 训练/推理集群脚本：`scripts/hpc/phase1/`
- 调试脚本：`tools/debug/`
- 可视化脚本：`tools/vis/`
- 历史入口和 workaround：`legacy/`

因此，`repairing/phase1/` 目录本身现在主要保留核心实现文件，而不是所有辅助脚本。

## 8. 当前代码中的几个实际注意点

### 8.1 train 和 inference 的配置加载方式不完全一致

- `train.py` 支持 `--class_name`
- `inference.py` 当前只读取默认 `cfg`

这是当前代码实现中的真实状态。

### 8.2 验证集不是独立数据划分

当前 validation 更接近“同一批样本上关闭增强和采样偏置后的重建评估”，不是泛化验证。

### 8.3 主线依赖样本索引 embedding

模型通过 `index -> embedding` 的方式工作，因此当前 Phase1 更像对现有样本做重建/修复，而不是标准的“对未知样本直接泛化”的纯前馈分割网络。

### 8.4 当前文档只描述代码行为

这份说明不再记录：

- 具体数据规模
- 某次训练的日期
- 某个历史 checkpoint 的结果
- 某次实验中的 Dice 数值

如果后续需要公开结果，建议另写实验报告或 `docs/results/` 文档。
