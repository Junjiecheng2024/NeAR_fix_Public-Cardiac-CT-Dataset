"""
NeAR v2.0 Phase1 Configuration - Project 2016526
=================================================
Configuration for classes with data stored in project_2016526:
LA, LV, RA, RV, PA

Use this config file instead of config.py for these 5 classes.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BaseConfig:
    """Base configuration for Phase1 training."""
    
    # Paths
    base_path: str = "/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/checkpoints"
    run_flag: str = "Phase1_"
    data_path: str = ""
    
    # Class information
    class_name: str = ""
    class_index: int = 0
    
    # Model architecture
    model_type: str = "shape_appearance"
    use_appearance: bool = True
    use_context: bool = True
    use_gradient: bool = True
    use_raw_ct: bool = True
    appearance_channels: int = 64
    decoder_channels: List[int] = field(default_factory=lambda: [64, 48, 32, 16])
    latent_dimension: int = 256
    
    # Data parameters
    target_resolution: int = 128
    n_training_samples: Optional[int] = None
    
    # Training parameters
    n_epochs: int = 400
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    eval_batch_size: int = 1
    
    # Optimization
    lr: float = 5e-4
    use_cosine_schedule: bool = True
    warmup_ratio: float = 0.02
    
    # Loss weights
    dice_weight: float = 0.3
    tversky_weight: float = 0.35
    boundary_dice_weight: float = 0.2
    focal_weight: float = 0.1
    topk_weight: float = 0.05
    l2_penalty_weight: float = 1e-4
    
    # Sampling strategy
    sampling_bias_ratio: float = 0.5
    sampling_dilation_radius: int = 3
    
    # Data augmentation
    augment: bool = True
    
    # Training infrastructure
    n_workers: int = 8
    use_amp: bool = True
    eval_interval: int = 5
    
    # Resume training
    resume_checkpoint: Optional[str] = None
    
    def to_dict(self):
        """Convert to dict for backward compatibility."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ==============================================================================
# Classes in project_2016526
# ==============================================================================

@dataclass 
class LAConfig(BaseConfig):
    """Left Atrium configuration."""
    class_name: str = "LA"
    class_index: int = 2
    run_flag: str = "LA_Tier2_"
    data_path: str = "/scratch/project_2016517/JunjieCheng/dataset/la_tier2"
    n_epochs: int = 400


@dataclass
class LVConfig(BaseConfig):
    """Left Ventricle configuration."""
    class_name: str = "LV"
    class_index: int = 3
    run_flag: str = "LV_Tier2_"
    data_path: str = "/scratch/project_2016517/JunjieCheng/dataset/lv_tier2"
    n_epochs: int = 400


@dataclass
class RAConfig(BaseConfig):
    """Right Atrium configuration."""
    class_name: str = "RA"
    class_index: int = 4
    run_flag: str = "RA_Tier2_"
    data_path: str = "/scratch/project_2016517/JunjieCheng/dataset/ra_tier2"
    n_epochs: int = 400


@dataclass
class RVConfig(BaseConfig):
    """Right Ventricle configuration."""
    class_name: str = "RV"
    class_index: int = 5
    run_flag: str = "RV_Tier2_"
    data_path: str = "/scratch/project_2016517/JunjieCheng/dataset/rv_tier2"
    n_epochs: int = 400


@dataclass
class PAConfig(BaseConfig):
    """Pulmonary Artery configuration."""
    class_name: str = "PA"
    class_index: int = 7
    run_flag: str = "PA_Tier2_"
    data_path: str = "/scratch/project_2016517/JunjieCheng/dataset/pa_tier2"
    n_epochs: int = 400


# Config registry
CONFIGS = {
    "la": LAConfig,
    "lv": LVConfig,
    "ra": RAConfig,
    "rv": RVConfig,
    "pa": PAConfig,
}


def get_config(class_name: str) -> BaseConfig:
    """Get configuration by class name."""
    key = class_name.lower()
    if key not in CONFIGS:
        raise ValueError(f"Unknown class: {class_name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[key]()


# Default config (for backward compatibility)
cfg = LAConfig().to_dict()
