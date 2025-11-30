"""
Configuration for PV (Phase 1).
"""


cfg = dict()

# Paths
cfg["base_path"] = "./checkpoints"
cfg["run_flag"] = "PV_class10_"
cfg['data_path'] = '/scratch/project_2016517/junjie/dataset/near_format_data'

# Class information
cfg['class_name'] = 'PV'
cfg['class_index'] = 10

# Training parameters
cfg["n_epochs"] = 1500

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
cfg["sampling_bias_ratio"] = 0.0
cfg["sampling_dilation_radius"] = 2

# Loss weights
cfg['l2_penalty_weight'] = 3e-4

# Resume training from checkpoint
cfg["resume_checkpoint"] = None

# Note: We want overfitting to get the best refined labels
