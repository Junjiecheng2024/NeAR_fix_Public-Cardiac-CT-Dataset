"""
Configuration for coronary (Phase 1).
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
cfg["n_epochs"] = 1000  # Can increase to 1500 if needed

# Model parameters
cfg['appearance'] = False  # Shape-only mode
cfg['decoder_channels'] = [64, 48, 32, 16]  # Decoder feature channels
cfg['latent_dimension'] = 256

# Data parameters
cfg["training_resolution"] = 128  
cfg["target_resolution"] = 128
cfg["n_training_samples"] = None

# Optimization
cfg["lr"] = 5e-4
cfg["batch_size"] = 1  
cfg["gradient_accumulation_steps"] = 6  
cfg["eval_batch_size"] = 1  
cfg["n_workers"] = 16

# Learning rate schedule 
cfg["use_cosine_schedule"] = True  
cfg["gamma"] = 0.5
cfg["warmup_ratio"] = 0  # 1% of total steps for warmup

# Mixed precision training (AMP)
cfg["use_amp"] = True

# Validation interval
cfg["eval_interval"] = 5

# Sampling strategy
cfg["grid_noise"] = 0
cfg["uniform_grid_noise"] = True
cfg["sampling_bias_ratio"] = 0.2
cfg["sampling_dilation_radius"] = 3  # Boundary region dilation

# Loss weights
cfg['l2_penalty_weight'] = 1e-4  

# Resume training from checkpoint
cfg["resume_checkpoint"] = "/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/near_repairing/phase1_coronary/checkpoints/Coronary_class9_shape_only_251121_105840/best.ckpt"

# Note: We want overfitting to get the best refined labels
# So we use all samples as both train and eval, no validation split
