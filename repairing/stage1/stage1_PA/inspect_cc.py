
import numpy as np
import os
from scipy.ndimage import label
import argparse

def inspect_cc(sample_id, data_path, target_class=7):
    print(f"\n--- Sample {sample_id} ---")
    
    # Try loading mask
    orig_mask_path = os.path.join(data_path, 'masks', f'{sample_id}.npy')
    if not os.path.exists(orig_mask_path):
        orig_mask_path = os.path.join(data_path, 'shape', f'{sample_id}.npy')
    if not os.path.exists(orig_mask_path):
        orig_mask_path = os.path.join(data_path, f'{sample_id}.npy')
        
    if not os.path.exists(orig_mask_path):
        print("Mask file not found.")
        return

    mask = np.load(orig_mask_path)
    
    # Handle multi-class
    if mask.max() > 1:
        mask = (mask == target_class).astype(np.uint8)
        
    print(f"Mask Shape: {mask.shape}")
    print(f"Total Voxels: {mask.sum()}")
    
    # Compute CC (Default connectivity)
    labeled, n_components = label(mask)
    print(f"Number of Components (Default): {n_components}")
    
    # Compute CC (26-connectivity)
    s = np.ones((3,3,3), dtype=int)
    labeled_26, n_components_26 = label(mask, structure=s)
    print(f"Number of Components (26-conn): {n_components_26}")
    
    if n_components > 0:
        # Calculate size of each component
        sizes = []
        for i in range(1, n_components + 1):
            size = (labeled == i).sum()
            sizes.append(size)
        
        sizes.sort(reverse=True)
        print(f"Component Sizes (sorted): {sizes}")
        
        if len(sizes) > 1:
            print(f"Ratio of 2nd largest to largest: {sizes[1]/sizes[0]:.6f}")
            
        # Simulate filtering
        threshold_ratio = 0.05
        total_voxels = mask.sum()
        valid_count = 0
        for sz in sizes:
            if sz / total_voxels >= threshold_ratio:
                valid_count += 1
        print(f"Components after 0.05 filter: {valid_count}")

if __name__ == "__main__":
    # Hardcoded for quick check based on user's context
    data_path = '/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset_backup/dataset/near_format_data'
    # The 10 samples from the inference run
    samples = ['1', '10', '100', '1000', '101', '102', '103', '104', '105', '106']
    
    print(f"Checking {len(samples)} samples with threshold 0.05...")
    
    for s in samples:
        inspect_cc(s, data_path, target_class=7)
