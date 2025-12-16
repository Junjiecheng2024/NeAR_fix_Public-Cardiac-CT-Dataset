"""
Dataset for Tier2 cardiac structure refinement using NeAR v2.0.
Supports Shape + Appearance (CT) training with class-specific crops.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.ndimage import binary_dilation, binary_erosion


class CoronaryTier2Dataset(Dataset):
    """
    Dataset for Coronary refinement with Shape + Appearance.
    
    Data structure expected (per case):
    root/
        case_001/
            ct.npy              # Normalized CT (D, H, W), range [0, 1]
            mask_coronary.npy   # Binary mask for coronary (D, H, W)
            mask_context.npy    # Binary mask for context (Myo + Aorta)
            crop_params.json    # Crop parameters for coordinate mapping
        case_002/
            ...
    """
    
    def __init__(
        self,
        root: str,
        resolution: int = None,  # None = keep original size
        n_samples: int = None,
        use_appearance: bool = True,
        boundary_bias_ratio: float = 0.5,
        boundary_dilation_radius: int = 3,
        augment: bool = False
    ):
        """
        Args:
            root: path to tier2 dataset root
            resolution: target resolution for resizing (None = keep original)
            n_samples: limit number of samples (None = use all)
            use_appearance: whether to load and return CT data
            boundary_bias_ratio: ratio of samples near boundaries
            boundary_dilation_radius: dilation radius for boundary region
            augment: whether to apply data augmentation
        """
        self.root = Path(root)
        self.resolution = resolution
        self.use_appearance = use_appearance
        self.boundary_bias_ratio = boundary_bias_ratio
        self.boundary_dilation_radius = boundary_dilation_radius
        self.augment = augment
        
        # Find all case directories
        self.case_dirs = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and (d / "mask_coronary.npy").exists():
                self.case_dirs.append(d)
        
        # Limit samples if specified
        if n_samples is not None and len(self.case_dirs) > n_samples:
            self.case_dirs = self.case_dirs[:n_samples]
        
        # Load crop params for all cases (needed for coordinate mapping later)
        self.crop_params = {}
        for case_dir in self.case_dirs:
            params_path = case_dir / "crop_params.json"
            if params_path.exists():
                with open(params_path) as f:
                    self.crop_params[case_dir.name] = json.load(f)
        
        print(f"[CoronaryTier2Dataset] Loaded {len(self.case_dirs)} cases from {root}")
        print(f"  - Resolution: {resolution or 'Original'}")
        print(f"  - Use appearance: {use_appearance}")
        print(f"  - Boundary bias: {boundary_bias_ratio*100:.0f}%")
    
    def __len__(self):
        return len(self.case_dirs)
    
    def __getitem__(self, index):
        """
        Returns:
            Dict containing:
                - index: sample index (for embedding lookup)
                - shape: binary mask tensor (1, D, H, W)
                - appearance: CT tensor (1, D, H, W) if use_appearance=True
                - context: context mask tensor (1, D, H, W) for Myo + Aorta
                - case_id: case identifier
        """
        case_dir = self.case_dirs[index]
        case_id = case_dir.name
        
        # Load coronary mask (shape)
        mask = np.load(case_dir / "mask_coronary.npy").astype(np.float32)
        
        # Load CT if using appearance
        if self.use_appearance:
            ct = np.load(case_dir / "ct.npy").astype(np.float32)
        else:
            ct = None
        
        # Load context mask
        context = np.load(case_dir / "mask_context.npy").astype(np.float32)
        
        # Resize if needed
        if self.resolution is not None:
            target_shape = (self.resolution,) * 3
            if mask.shape != target_shape:
                from skimage.transform import resize as sk_resize
                mask = sk_resize(mask, target_shape, order=0, 
                                preserve_range=True, anti_aliasing=False)
                mask = (mask > 0.5).astype(np.float32)
                
                context = sk_resize(context, target_shape, order=0,
                                   preserve_range=True, anti_aliasing=False)
                context = (context > 0.5).astype(np.float32)
                
                if ct is not None:
                    ct = sk_resize(ct, target_shape, order=3,
                                  preserve_range=True).astype(np.float32)
        
        # Apply augmentation if enabled
        if self.augment:
            mask, ct, context = self._augment(mask, ct, context)
        
        # Convert to tensors
        shape_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # (1, D, H, W)
        context_tensor = torch.from_numpy(context).float().unsqueeze(0)
        
        result = {
            "index": index,
            "shape": shape_tensor,
            "context": context_tensor,
            "case_id": case_id
        }
        
        if ct is not None:
            result["appearance"] = torch.from_numpy(ct).float().unsqueeze(0)
        
        return result
    
    def _augment(self, mask, ct, context):
        """Apply random augmentations."""
        # Random flip along each axis
        for axis in range(3):
            if np.random.random() > 0.5:
                mask = np.flip(mask, axis=axis).copy()
                context = np.flip(context, axis=axis).copy()
                if ct is not None:
                    ct = np.flip(ct, axis=axis).copy()
        
        # Random 90-degree rotation in xy plane
        k = np.random.randint(4)
        if k > 0:
            mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
            context = np.rot90(context, k=k, axes=(1, 2)).copy()
            if ct is not None:
                ct = np.rot90(ct, k=k, axes=(1, 2)).copy()
        
        return mask, ct, context
    
    def get_boundary_mask(self, shape: np.ndarray) -> np.ndarray:
        """
        Compute boundary region mask for biased sampling.
        
        Args:
            shape: binary mask (D, H, W)
        Returns:
            boundary_mask: binary mask of boundary region
        """
        # Dilate the shape
        dilated = binary_dilation(shape > 0.5, iterations=self.boundary_dilation_radius)
        # Erode the shape
        eroded = binary_erosion(shape > 0.5, iterations=1)
        # Boundary = dilated minus eroded (captures both sides of boundary)
        boundary = np.logical_and(dilated, ~eroded)
        
        return boundary.astype(np.float32)
    
    def get_crop_params(self, case_id: str) -> dict:
        """Get crop parameters for a specific case."""
        return self.crop_params.get(case_id, None)


class CoronaryTier2DataModule:
    """
    Data module for PyTorch Lightning compatibility.
    """
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage=None):
        """Setup datasets."""
        self.train_dataset = CoronaryTier2Dataset(
            root=self.cfg["data_path"],
            resolution=self.cfg.get("target_resolution", None),
            n_samples=self.cfg.get("n_training_samples", None),
            use_appearance=self.cfg.get("use_appearance", True),
            boundary_bias_ratio=self.cfg.get("sampling_bias_ratio", 0.5),
            boundary_dilation_radius=self.cfg.get("sampling_dilation_radius", 3),
            augment=self.cfg.get("augment", True)
        )
        
        # For NeAR, we typically use same data for train/val (overfit mode)
        self.val_dataset = CoronaryTier2Dataset(
            root=self.cfg["data_path"],
            resolution=self.cfg.get("target_resolution", None),
            n_samples=self.cfg.get("n_training_samples", None),
            use_appearance=self.cfg.get("use_appearance", True),
            boundary_bias_ratio=0.0,  # No bias for validation
            boundary_dilation_radius=self.cfg.get("sampling_dilation_radius", 3),
            augment=False
        )
    
    def train_dataloader(self):
        import torch
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.get("batch_size", 1),
            shuffle=True,
            num_workers=self.cfg.get("n_workers", 4),
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        import torch
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.get("eval_batch_size", 1),
            shuffle=False,
            num_workers=self.cfg.get("n_workers", 4),
            pin_memory=True
        )
