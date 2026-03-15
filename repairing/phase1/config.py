"""
NeAR v2.0 Phase1 Configuration System
Uses class inheritance for different cardiac classes.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(os.environ.get("NEAR_DATA_ROOT", REPO_ROOT / "dataset"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("NEAR_OUTPUT_ROOT", REPO_ROOT / "outputs"))
DEFAULT_PHASE1_CHECKPOINT_ROOT = Path(
    os.environ.get("NEAR_PHASE1_CHECKPOINT_ROOT", DEFAULT_OUTPUT_ROOT / "phase1" / "checkpoints")
)


def default_class_data_path(class_dir: str) -> str:
    return str(DEFAULT_DATA_ROOT / class_dir)


@dataclass
class BaseConfig:
    """Base configuration for Phase1 training."""
    
    # Paths (to be overridden per class)
    base_path: str = str(DEFAULT_PHASE1_CHECKPOINT_ROOT)
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
    data_path: str = default_class_data_path("coronary_tier2")
    
    # Coronary-specific: longer training
    n_epochs: int = 600


@dataclass
class AortaConfig(BaseConfig):
    """Aorta configuration (Tier1 - medium structure)."""
    class_name: str = "Aorta"
    class_index: int = 6
    run_flag: str = "Aorta_Tier2_"
    data_path: str = default_class_data_path("aorta_tier2")
    
    # Aorta: simpler structure
    n_epochs: int = 400
    boundary_dice_weight: float = 0.15


@dataclass
class MyocardiumConfig(BaseConfig):
    """Myocardium configuration (Tier1 - large structure)."""
    class_name: str = "Myocardium"
    class_index: int = 1
    run_flag: str = "Myocardium_Tier2_"
    data_path: str = default_class_data_path("myocardium_tier2")
    n_epochs: int = 400
    boundary_dice_weight: float = 0.15


@dataclass 
class LAConfig(BaseConfig):
    """Left Atrium configuration."""
    class_name: str = "LA"
    class_index: int = 2
    run_flag: str = "LA_Tier2_"
    data_path: str = default_class_data_path("la_tier2")
    n_epochs: int = 400


@dataclass
class LVConfig(BaseConfig):
    """Left Ventricle configuration."""
    class_name: str = "LV"
    class_index: int = 3
    run_flag: str = "LV_Tier2_"
    data_path: str = default_class_data_path("lv_tier2")
    n_epochs: int = 400


@dataclass
class RAConfig(BaseConfig):
    """Right Atrium configuration."""
    class_name: str = "RA"
    class_index: int = 4
    run_flag: str = "RA_Tier2_"
    data_path: str = default_class_data_path("ra_tier2")
    n_epochs: int = 400


@dataclass
class RVConfig(BaseConfig):
    """Right Ventricle configuration."""
    class_name: str = "RV"
    class_index: int = 5
    run_flag: str = "RV_Tier2_"
    data_path: str = default_class_data_path("rv_tier2")
    n_epochs: int = 400


@dataclass
class PAConfig(BaseConfig):
    """Pulmonary Artery configuration."""
    class_name: str = "PA"
    class_index: int = 7
    run_flag: str = "PA_Tier2_"
    data_path: str = default_class_data_path("pa_tier2")
    n_epochs: int = 400


@dataclass
class PVConfig(BaseConfig):
    """Pulmonary Vein configuration (small structure)."""
    class_name: str = "PV"
    class_index: int = 10
    run_flag: str = "PV_Tier2_"
    data_path: str = default_class_data_path("pv_tier2")
    n_epochs: int = 500  # More epochs for small structure


@dataclass
class LAAConfig(BaseConfig):
    """Left Atrial Appendage configuration (small structure)."""
    class_name: str = "LAA"
    class_index: int = 8
    run_flag: str = "LAA_Tier2_"
    data_path: str = default_class_data_path("laa_tier2")
    n_epochs: int = 500  # More epochs for small structure


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
