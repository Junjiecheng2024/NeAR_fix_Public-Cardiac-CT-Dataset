"""
阶段一：冠状动脉（Coronary）训练配置文件

配置说明：
- 训练分辨率：128³（2.1M采样点，提供高精度细节）
- 批处理：batch_size=1, gradient_accumulation=4（有效batch=4）
- 学习率：5e-4（精调阶段，配合Cosine退火）
- 边界偏采样：50%→30%→10%（课程学习策略）
- 损失函数：70% Dice + 30% Focal (gamma=4.0)
- 验证间隔：每10轮（节省时间）
- 混合精度：AMP开启（节省27.6GB显存）

Configuration for Coronary (class 9) single-class refinement.
Phase 1: Shape-only NeAR training to refine noisy labels.
"""

cfg = dict()

# Paths
cfg["base_path"] = "./checkpoints"
cfg["run_flag"] = "Coronary_class9_"
cfg['data_path'] = '/scratch/project_2016517/junjie/dataset/near_format_data'

# Class information
cfg['class_name'] = 'Coronary'
cfg['class_index'] = 9

# Training parameters
cfg["n_epochs"] = 400  # Can increase to 1500 if needed


# Model parameters
cfg['appearance'] = False  # Shape-only mode
cfg['decoder_channels'] = [64, 48, 32, 16]  # Decoder feature channels
cfg['latent_dimension'] = 256

# Data parameters
cfg["training_resolution"] = 128  
cfg["target_resolution"] = 128   # Dataset加载和验证时的分辨率
cfg["n_training_samples"] = None  # Use all samples (指的是所有病例，不是采样点数)

# Optimization
cfg["lr"] = 2e-3  # 恢复初始学习率，快速学习
cfg["batch_size"] = 1  
cfg["gradient_accumulation_steps"] = 6  
cfg["eval_batch_size"] = 1  
cfg["n_workers"] = 8

# Learning rate schedule 
cfg["use_cosine_schedule"] = True  
# cfg["milestones"] = [100, 200]  # Epoch 100和200降低学习率
cfg["gamma"] = 0.5  # 每次降低50%
cfg["warmup_ratio"] = 0.01  # 1% of total steps for warmup

# Mixed precision training (AMP)
cfg["use_amp"] = True  # 使用自动混合精度训练，降低显存使用

# Validation interval
cfg["eval_interval"] = 5  # 每10轮验证一次（128³训练更慢）

# Sampling strategy
cfg["grid_noise"] = 0  # Grid noise for data augmentation (阶段1暂不使用)
cfg["uniform_grid_noise"] = True
cfg["sampling_bias_ratio"] = 0.2  # 初始50%边界采样，训练中动态调整
cfg["sampling_dilation_radius"] = 2  # Boundary region dilation

# Loss weights
cfg['l2_penalty_weight'] = 3e-4  

# Resume training from checkpoint
cfg["resume_checkpoint"] = "/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/near_repairing/stage1_coronary/checkpoints/Coronary_class9_shape_only_251121_105840/best.ckpt"  # 继承enhanced训练的best model (Epoch 1, Dice 7.31%)

# Note: We want overfitting to get the best refined labels
# So we use all samples as both train and eval, no validation split
