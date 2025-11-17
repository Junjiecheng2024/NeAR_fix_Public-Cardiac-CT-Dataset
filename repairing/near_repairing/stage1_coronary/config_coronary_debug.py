"""
调试配置：快速测试 Lightning 训练流程
"""

cfg = dict()

# Paths
cfg["base_path"] = "./checkpoints"
cfg["run_flag"] = "Coronary_class9_DEBUG_"
cfg['data_path'] = '/scratch/project_2016517/junjie/dataset/near_format_data'

# Class information
cfg['class_name'] = 'Coronary'
cfg['class_index'] = 9

# Training parameters - 调试模式：只跑2个epoch
cfg["n_epochs"] = 2

# Model parameters
cfg['appearance'] = False
cfg['decoder_channels'] = [64, 48, 32, 16]
cfg['latent_dimension'] = 256

# Data parameters
cfg["training_resolution"] = 64
cfg["target_resolution"] = 128
cfg["n_training_samples"] = None

# Optimization
cfg["lr"] = 1e-3
cfg["batch_size"] = 2
cfg["gradient_accumulation_steps"] = 2
cfg["eval_batch_size"] = 2
cfg["n_workers"] = 4  # 减少worker避免调试时资源占用

# Learning rate schedule
cfg["use_cosine_schedule"] = False
cfg["milestones"] = [100, 200]
cfg["gamma"] = 0.5
cfg["warmup_ratio"] = 0.01

# Mixed precision training
cfg["use_amp"] = True

# Validation interval
cfg["eval_interval"] = 1  # 每轮都验证

# Sampling strategy
cfg["grid_noise"] = 0
cfg["uniform_grid_noise"] = True
cfg["sampling_bias_ratio"] = 0.5
cfg["sampling_dilation_radius"] = 2

# Loss weights
cfg['l2_penalty_weight'] = 1e-4

# Resume training from checkpoint
cfg["resume_checkpoint"] = "./checkpoints/Coronary_class9_shape_only_251109_061814/best.pth"
