"""
Configuration for Coronary (class 9) single-class refinement.
Phase 1: Shape-only NeAR training to refine noisy labels.
"""

cfg = dict()

# Paths
cfg["base_path"] = "./checkpoints"
cfg["run_flag"] = "Coronary_class9_shape_only_"
cfg['data_path'] = '../../../dataset/near_format_data'

# Class information
cfg['class_name'] = 'Coronary'
cfg['class_index'] = 9

# Training parameters
cfg["n_epochs"] = 300  # Can increase to 1500 if needed
# Note: milestones已废弃，现在使用Cosine Annealing scheduler

# Model parameters
cfg['appearance'] = False  # Shape-only mode
cfg['decoder_channels'] = [64, 48, 32, 16]  # Decoder feature channels
cfg['latent_dimension'] = 256

# Data parameters
cfg["training_resolution"] = 128   # 训练时的采样分辨率（128³ = 2.1M点，提升细节）
cfg["target_resolution"] = 128   # Dataset加载和验证时的分辨率
cfg["n_training_samples"] = None  # Use all samples (指的是所有病例，不是采样点数)

# Optimization
cfg["lr"] = 5e-4  # 降低学习率以适应128³高分辨率和精调阶段
cfg["batch_size"] = 1  # Physical batch size per GPU (降低以适应128³)
cfg["gradient_accumulation_steps"] = 4  # Simulate batch_size = 4
cfg["eval_batch_size"] = 1  # 验证时用 128³ 采样，降低 batch_size 避免 OOM
cfg["n_workers"] = 8

# Learning rate schedule (warmup + cosine annealing)
cfg["use_cosine_schedule"] = True
cfg["warmup_ratio"] = 0.01  # 1% of total steps for warmup

# Mixed precision training (AMP)
cfg["use_amp"] = True  # 使用自动混合精度训练，降低显存使用

# Validation interval
cfg["eval_interval"] = 10  # 每10轮验证一次（128³训练更慢）

# Sampling strategy
cfg["grid_noise"] = 0  # Grid noise for data augmentation (过拟合目标下不使用)
cfg["uniform_grid_noise"] = True
cfg["sampling_bias_ratio"] = 0.5  # 50% samples near boundaries
cfg["sampling_dilation_radius"] = 2  # Boundary region dilation

# Loss weights
cfg['l2_penalty_weight'] = 1e-3  # L2 regularization on latent codes

# Resume training from checkpoint
cfg["resume_checkpoint"] = "./checkpoints/Coronary_class9_shape_only_251109_061814/best.pth"  # 继承enhanced训练的best model (Epoch 1, Dice 7.31%)

# Note: We want overfitting to get the best refined labels
# So we use all samples as both train and eval, no validation split
