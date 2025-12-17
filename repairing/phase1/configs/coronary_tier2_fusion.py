"""
Configuration for Coronary Tier2 Fusion Model (NeAR v2.1)
Uses FusionDecoderShapeAppearance with raw CT + gradient features.
"""

cfg = dict()

# ==============================================================================
# Paths
# ==============================================================================
cfg["base_path"] = "./checkpoints"
cfg["run_flag"] = "Coronary_Tier2_Fusion_"

# Data path: output of prepare_coronary_tier2.py
cfg['data_path'] = '/scratch/project_2016517/junjie/dataset/coronary_tier2'

# ==============================================================================
# Class information
# ==============================================================================
cfg['class_name'] = 'Coronary'
cfg['class_index'] = 9

# ==============================================================================
# Model architecture - Fusion Model
# ==============================================================================
cfg['model_type'] = 'fusion'             # NEW: Use FusionDecoderShapeAppearance
cfg['use_appearance'] = True             # Enable multi-scale appearance features
cfg['use_context'] = True                # Enable context mask (Myo + Aorta)
cfg['use_gradient'] = True               # NEW: Enable 3D Sobel gradient features
cfg['use_raw_ct'] = True                 # NEW: Include raw CT values (like HINTLab)
cfg['appearance_channels'] = 64          # Appearance encoder output channels
cfg['decoder_channels'] = [64, 48, 32, 16]  # Shape decoder channels
cfg['latent_dimension'] = 256

# ==============================================================================
# Data parameters
# ==============================================================================
cfg["target_resolution"] = 128            # CHANGED: 256 causes OOM, use 128
cfg["n_training_samples"] = None          # None = use all samples

# ==============================================================================
# Training parameters
# ==============================================================================
cfg["n_epochs"] = 600                    # Slightly longer for fusion model
cfg["batch_size"] = 1                    # Batch size per GPU
cfg["gradient_accumulation_steps"] = 4   # Effective batch = 4
cfg["eval_batch_size"] = 1

# ==============================================================================
# Optimization
# ==============================================================================
cfg["lr"] = 5e-4                         # Learning rate
cfg["use_cosine_schedule"] = True        # Cosine annealing
cfg["warmup_ratio"] = 0.02               # 2% warmup

# ==============================================================================
# Loss weights (total = 1.0)
# ==============================================================================
cfg['dice_weight'] = 0.55                # Standard dice loss
cfg['boundary_dice_weight'] = 0.25       # Boundary-focused dice (slightly higher)
cfg['focal_weight'] = 0.15               # Focal loss for class imbalance
cfg['l2_penalty_weight'] = 1e-4          # Latent regularization

# ==============================================================================
# Sampling strategy
# ==============================================================================
cfg["sampling_bias_ratio"] = 0.5         # 50% boundary sampling
cfg["sampling_dilation_radius"] = 3      # Boundary region dilation

# ==============================================================================
# Data augmentation
# ==============================================================================
cfg["augment"] = True                    # Enable random flips and rotations

# ==============================================================================
# Training infrastructure
# ==============================================================================
cfg["n_workers"] = 8                     # Data loader workers
cfg["use_amp"] = True                    # Mixed precision (FP16)
cfg["eval_interval"] = 5                 # Validate every N epochs

# ==============================================================================
# Resume training
# ==============================================================================
cfg["resume_checkpoint"] = None          # Set to checkpoint path to resume

# ==============================================================================
# Notes - NeAR v2.1 Fusion Model
# ==============================================================================
# Key improvements from v2.0:
# 1. FusionDecoderShapeAppearance combines all approaches
# 2. Raw CT values preserved (like original HINTLab NeAR)
# 3. 3D Sobel gradients for edge detection
# 4. Multi-scale appearance features (from v2.0)
# 5. Context mask for spatial guidance (from v2.0)
# 6. Skip connection MLP with larger capacity
# 
# Total feature channels: 163 (shape) + 160 (appearance) + 1 (raw CT) + 4 (gradient) = 328
