"""
Dataset for single-class cardiac structure refinement using NeAR.
Loads bbox-cropped shape data for each class.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class CardiacClassDataset(Dataset):
    """
    Dataset for single cardiac structure class (e.g., Coronary, LAA, etc.)
    
    Data structure expected:
    root/
        shape/
            1.npy
            2.npy
            ...
        info.csv  (contains sample IDs)
    
    Each .npy file contains a binary mask (0 or 1) for the specific class,
    already cropped to bbox+margin.
    """
    
    def __init__(self, root, class_name='Coronary', resolution=128, n_samples=None, class_index=9):
        """
        Args:
            root: path to dataset root (e.g., '/path/to/near_format_data')
            class_name: name of the class (for logging/debugging)
            resolution: target resolution for the mask
            n_samples: limit number of samples (None = use all)
            class_index: class label index to extract from multi-class segmentation (e.g., 9 for Coronary)
        """
        self.root = root
        self.class_name = class_name
        self.resolution = resolution
        self.class_index = class_index  # NEW: store class index
        self.shape_dir = os.path.join(root, 'shape')
        
        # Load sample IDs from info.csv
        info_path = os.path.join(root, 'info.csv')
        if os.path.exists(info_path):
            df = pd.read_csv(info_path)
            self.sample_ids = df['id'].tolist()
        else:
            # If no info.csv, scan shape directory
            shape_files = [f for f in os.listdir(self.shape_dir) if f.endswith('.npy')]
            self.sample_ids = [f.replace('.npy', '') for f in sorted(shape_files, key=lambda x: int(x.replace('.npy', '')))]
        
        # Limit samples if specified
        if n_samples is not None and len(self.sample_ids) > n_samples:
            self.sample_ids = self.sample_ids[:n_samples]
        
        print(f"Loaded {len(self.sample_ids)} samples for class '{class_name}'")
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, index):
        """
        Returns:
            index: sample index (for embedding lookup)
            shape: binary mask tensor (1, D, H, W) for the specific class
        """
        sample_id = self.sample_ids[index]
        shape_path = os.path.join(self.shape_dir, f'{sample_id}.npy')
        
        # Load multi-class mask
        shape_multi = np.load(shape_path)
        
        # Extract specific class (e.g., class_index=9 for Coronary)
        shape = (shape_multi == self.class_index).astype(np.float32)
        
        # Resize if needed
        if shape.shape != (self.resolution, self.resolution, self.resolution):
            from skimage.transform import resize
            shape = resize(shape, (self.resolution, self.resolution, self.resolution), 
                          order=0, preserve_range=True, anti_aliasing=False)  # order=0 for binary
            shape = (shape > 0.5).astype(np.float32)
        
        # Convert to tensor with channel dimension
        shape_tensor = torch.from_numpy(shape).float().unsqueeze(0)  # (1, D, H, W)
        
        return index, shape_tensor


class CardiacClassDatasetWithBiasedSampling(CardiacClassDataset):
    """
    Extended dataset that supports biased sampling near object boundaries.
    This is used during training to focus on boundary refinement.
    """
    
    def __init__(self, root, class_name='Coronary', resolution=128, n_samples=None,
                 sampling_bias_ratio=0.5, sampling_dilation_radius=2, class_index=9):
        """
        Args:
            sampling_bias_ratio: ratio of samples near boundaries (0.5 = 50% near boundary)
            sampling_dilation_radius: dilation radius for boundary region (in voxels)
            class_index: class label index to extract from multi-class segmentation
        """
        super().__init__(root, class_name, resolution, n_samples, class_index)
        self.sampling_bias_ratio = sampling_bias_ratio
        self.sampling_dilation_radius = sampling_dilation_radius
        
        print(f"Using biased sampling: {sampling_bias_ratio*100:.0f}% near boundaries "
              f"(dilation radius={sampling_dilation_radius})")
    
    def get_boundary_mask(self, shape):
        """
        Compute boundary region by dilating the object and XORing with original.
        
        Args:
            shape: binary mask (D, H, W)
        Returns:
            boundary_mask: binary mask of boundary region
        """
        from scipy.ndimage import binary_dilation
        
        # Dilate the shape
        dilated = binary_dilation(shape, iterations=self.sampling_dilation_radius)
        
        # Boundary = dilated XOR original
        boundary = np.logical_xor(dilated, shape)
        
        return boundary.astype(np.float32)
