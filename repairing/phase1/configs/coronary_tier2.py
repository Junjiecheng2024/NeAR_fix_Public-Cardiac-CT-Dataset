"""
Configuration for Coronary Tier2 (NeAR v2.0 - Shape + Appearance)
"""

cfg = dict()

# ==============================================================================
# Paths
# ==============================================================================
cfg["base_path"] = "./checkpoints"
cfg["run_flag"] = "Coronary_Tier2_v2_"

# Data path: output of prepare_coronary_tier2.py
cfg['data_path'] = '/scratch/project_2016517/junjie/dataset/coronary_tier2'

# ==============================================================================
# Class information
# ==============================================================================
cfg['class_name'] = 'Coronary'
cfg['class_index'] = 9

# ==============================================================================
# Model architecture
# ==============================================================================
cfg['use_appearance'] = True           # NEW: Enable CT appearance branch
cfg['use_context'] = True              # NEW: Enable context mask (Myo + Aorta)
cfg['appearance_channels'] = 64        # NEW: Appearance encoder output channels
cfg['decoder_channels'] = [64, 48, 32, 16]  # Shape decoder channels
cfg['latent_dimension'] = 256

# ==============================================================================
# Data parameters
# ==============================================================================
cfg["target_resolution"] = None         # None = keep original crop size (recommended)
cfg["n_training_samples"] = None        # None = use all samples

# ==============================================================================
# Training parameters
# ==============================================================================
cfg["n_epochs"] = 500                   # Epochs (can increase to 800-1000)
cfg["batch_size"] = 1                   # Batch size per GPU
cfg["gradient_accumulation_steps"] = 4  # Effective batch = 4
cfg["eval_batch_size"] = 1

# ==============================================================================
# Optimization
# ==============================================================================
cfg["lr"] = 5e-4                        # Learning rate
cfg["use_cosine_schedule"] = True       # Cosine annealing
cfg["warmup_ratio"] = 0.02              # 2% warmup

# ==============================================================================
# Loss weights (total = 1.0)
# ==============================================================================
cfg['dice_weight'] = 0.6                # Standard dice loss
cfg['boundary_dice_weight'] = 0.2       # NEW: Boundary-focused dice
cfg['focal_weight'] = 0.15              # Focal loss for class imbalance
cfg['l2_penalty_weight'] = 1e-4         # Latent regularization

# ==============================================================================
# Sampling strategy
# ==============================================================================
cfg["sampling_bias_ratio"] = 0.5        # 50% boundary sampling (increased from 20%)
cfg["sampling_dilation_radius"] = 3     # Boundary region dilation

# ==============================================================================
# Data augmentation
# ==============================================================================
cfg["augment"] = True                   # Enable random flips and rotations

# ==============================================================================
# Training infrastructure
# ==============================================================================
cfg["n_workers"] = 8                    # Data loader workers
cfg["use_amp"] = True                   # Mixed precision (FP16)
cfg["eval_interval"] = 5                # Validate every N epochs

# ==============================================================================
# Resume training
# ==============================================================================
cfg["resume_checkpoint"] = None         # Set to checkpoint path to resume

# ==============================================================================
# Notes
# ==============================================================================
# Key differences from v1.0 (shape-only):
# 1. Uses CT appearance branch for boundary-aware learning
# 2. Uses context mask (Myocardium + Aorta) for spatial guidance
# 3. Includes Boundary Dice loss for fine structure preservation
# 4. Higher sampling_bias_ratio (50% vs 20%) for better boundary coverage
# 5. Data is class-specific cropped (higher voxel ratio ~5-8% vs 0.12%)
