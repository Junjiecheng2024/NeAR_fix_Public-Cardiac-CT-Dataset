"""
NeAR v2.0 Phase1 Configuration System
Uses class inheritance for different cardiac classes.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BaseConfig:
    """Base configuration for Phase1 training."""
    
    # Paths (to be overridden per class)
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
    n_epochs: int = 600
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    eval_batch_size: int = 1
    
    # Optimization
    lr: float = 5e-4
    use_cosine_schedule: bool = True
    warmup_ratio: float = 0.02
    
    # Loss weights (optimized for small structures like coronary)
    dice_weight: float = 0.3
    tversky_weight: float = 0.35        # Emphasize recall
    boundary_dice_weight: float = 0.2   # Focus on boundaries
    focal_weight: float = 0.1           # Handle class imbalance
    topk_weight: float = 0.05           # Hard example mining
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
# Class-specific configurations
# ==============================================================================

@dataclass
class CoronaryConfig(BaseConfig):
    """Coronary artery configuration (Tier2 - small structure)."""
    class_name: str = "Coronary"
    class_index: int = 9
    run_flag: str = "Coronary_Tier2_"
    data_path: str = "/scratch/project_2016517/JunjieCheng/dataset/coronary_tier2"
    
    # Coronary-specific: longer training, higher boundary focus
    n_epochs: int = 600
    boundary_dice_weight: float = 0.25


@dataclass
class AortaConfig(BaseConfig):
    """Aorta configuration (Tier1 - medium structure)."""
    class_name: str = "Aorta"
    class_index: int = 6
    run_flag: str = "Aorta_"
    data_path: str = "/scratch/project_2016517/junjie/dataset/aorta_tier2"
    
    # Aorta: simpler structure, less boundary focus needed
    n_epochs: int = 400
    boundary_dice_weight: float = 0.2


@dataclass
class MyocardiumConfig(BaseConfig):
    """Myocardium configuration (Tier1 - large structure)."""
    class_name: str = "Myocardium"
    class_index: int = 5
    run_flag: str = "Myocardium_"
    data_path: str = "/scratch/project_2016517/junjie/dataset/myocardium_tier2"
    n_epochs: int = 400


@dataclass 
class LAConfig(BaseConfig):
    """Left Atrium configuration."""
    class_name: str = "LA"
    class_index: int = 1
    run_flag: str = "LA_"
    data_path: str = "/scratch/project_2016517/junjie/dataset/la_tier2"


@dataclass
class LVConfig(BaseConfig):
    """Left Ventricle configuration."""
    class_name: str = "LV"
    class_index: int = 2
    run_flag: str = "LV_"
    data_path: str = "/scratch/project_2016517/junjie/dataset/lv_tier2"


@dataclass
class RAConfig(BaseConfig):
    """Right Atrium configuration."""
    class_name: str = "RA"
    class_index: int = 3
    run_flag: str = "RA_"
    data_path: str = "/scratch/project_2016517/junjie/dataset/ra_tier2"


@dataclass
class RVConfig(BaseConfig):
    """Right Ventricle configuration."""
    class_name: str = "RV"
    class_index: int = 4
    run_flag: str = "RV_"
    data_path: str = "/scratch/project_2016517/junjie/dataset/rv_tier2"


@dataclass
class PAConfig(BaseConfig):
    """Pulmonary Artery configuration."""
    class_name: str = "PA"
    class_index: int = 7
    run_flag: str = "PA_"
    data_path: str = "/scratch/project_2016517/junjie/dataset/pa_tier2"


@dataclass
class PVConfig(BaseConfig):
    """Pulmonary Vein configuration."""
    class_name: str = "PV"
    class_index: int = 8
    run_flag: str = "PV_"
    data_path: str = "/scratch/project_2016517/junjie/dataset/pv_tier2"


@dataclass
class LAAConfig(BaseConfig):
    """Left Atrial Appendage configuration."""
    class_name: str = "LAA"
    class_index: int = 10
    run_flag: str = "LAA_"
    data_path: str = "/scratch/project_2016517/junjie/dataset/laa_tier2"


# Config registry for easy access
CONFIGS = {
    "coronary": CoronaryConfig,
    "aorta": AortaConfig,
    "myocardium": MyocardiumConfig,
    "la": LAConfig,
    "lv": LVConfig,
    "ra": RAConfig,
    "rv": RVConfig,
    "pa": PAConfig,
    "pv": PVConfig,
    "laa": LAAConfig,
}


def get_config(class_name: str) -> BaseConfig:
    """Get configuration by class name."""
    key = class_name.lower()
    if key not in CONFIGS:
        raise ValueError(f"Unknown class: {class_name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[key]()


# For backward compatibility: export current default as cfg dict
cfg = CoronaryConfig().to_dict()
