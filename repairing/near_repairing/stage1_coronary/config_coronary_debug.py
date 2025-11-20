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

# Data parameters - 降低分辨率以节省内存
cfg["training_resolution"] = 64  # 从128降到64 (内存降低8倍)
cfg["target_resolution"] = 64    # 从128降到64
cfg["n_training_samples"] = None

# Optimization
cfg["lr"] = 1e-3
cfg["batch_size"] = 1  # 从2降到1 (内存降低一半)
cfg["gradient_accumulation_steps"] = 4  # 从2增加到4，保持有效batch size=4

# Scheduler
cfg["use_cosine_schedule"] = False
cfg["milestones"] = [100, 200]
cfg["gamma"] = 0.5

# Regularization
cfg["l2_penalty_weight"] = 1e-4

# Grid sampling
cfg["grid_noise"] = 0
cfg["uniform_grid_noise"] = True
cfg["sampling_bias_ratio"] = 0.0  # 调试时关闭边界采样以节省内存
cfg["sampling_dilation_radius"] = 2

# Evaluation
cfg["eval_batch_size"] = 1
cfg["eval_interval"] = 1

# System
cfg["n_workers"] = 0  # 单GPU调试时设为0
cfg["use_amp"] = False  # 可以设为True使用混合精度进一步节省内存