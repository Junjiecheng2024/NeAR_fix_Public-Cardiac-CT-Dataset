
import numpy as np
import os

def inspect_mask(sample_id):
    path = f'/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/near_repairing/stage1_PA/class7_PA_results_256/{sample_id}_refined.npy'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    mask = np.load(path)
    print(f"\n--- Sample {sample_id} ---")
    print(f"Shape: {mask.shape}")
    print(f"Sum: {mask.sum()}")
    
    if mask.sum() > 0:
        # Find bounding box
        coords = np.argwhere(mask > 0)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        print(f"BBox Z: {z_min}-{z_max} (Center: {(z_min+z_max)//2})")
        print(f"BBox Y: {y_min}-{y_max}")
        print(f"BBox X: {x_min}-{x_max}")
        
        # Check middle slice
        mid_z = mask.shape[0] // 2
        mid_slice_sum = mask[mid_z, :, :].sum()
        print(f"Sum at middle slice ({mid_z}): {mid_slice_sum}")
    else:
        print("Mask is empty!")

inspect_mask('1')
inspect_mask('10')
inspect_mask('100')
