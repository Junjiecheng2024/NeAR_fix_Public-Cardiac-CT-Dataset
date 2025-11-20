"""
阶段一：Myocardium 训练配置文件

配置说明：
- 训练分辨率：128³
- 批处理：batch_size=1, gradient_accumulation=4
- 学习率：5e-4
- 边界偏采样：50%→30%→10%
- 损失函数：70% Dice + 30% Focal (gamma=4.0)
- 验证间隔：每10轮
- 混合精度：AMP开启

Configuration for Myocardium (class 1) single-class refinement.
Phase 1: Shape-only NeAR training to refine noisy labels.
"""

cfg = dict()

# Paths
cfg["base_path"] = "./checkpoints"
cfg["run_flag"] = "Myocardium_class1_shape_only_"
cfg['data_path'] = '/scratch/project_2016517/junjie/dataset/near_format_data'

# Class information
cfg['class_name'] = 'Myocardium'
cfg['class_index'] = 1

# Training parameters
cfg["n_epochs"] = 400
# Note: milestones已废弃，现在使用Cosine Annealing scheduler

# Model parameters
cfg['appearance'] = False  # Shape-only mode
cfg['decoder_channels'] = [64, 48, 32, 16]
cfg['latent_dimension'] = 256

# Data parameters
cfg["training_resolution"] = 128  
cfg["target_resolution"] = 128   
cfg["n_training_samples"] = None 

# Optimization
cfg["lr"] = 1e-3
cfg["batch_size"] = 1  
cfg["gradient_accumulation_steps"] = 6
cfg["eval_batch_size"] = 1  
cfg["n_workers"] = 8

# Learning rate schedule 
cfg["use_cosine_schedule"] = True  
cfg["gamma"] = 0.5
cfg["warmup_ratio"] = 0.01

# Mixed precision training (AMP)
cfg["use_amp"] = True

# Validation interval
cfg["eval_interval"] = 5

# Sampling strategy
cfg["grid_noise"] = 0
cfg["uniform_grid_noise"] = True
cfg["sampling_bias_ratio"] = 0.05
cfg["sampling_dilation_radius"] = 2

# Loss weights
cfg['l2_penalty_weight'] = 1e-4

# Resume training from checkpoint
cfg["resume_checkpoint"] = None

# Note: We want overfitting to get the best refined labels
