import os
import argparse
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, label, generate_binary_structure
from tqdm import tqdm
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    nibabel = None
    NIBABEL_AVAILABLE = False

def get_structuring_element(radius):
    """Generate a spherical structuring element."""
    # A simple way to get a sphere is to use distance from center
    L = 2 * radius + 1
    z, y, x = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    element = (z**2 + y**2 + x**2) <= radius**2
    return element

def keep_largest_k_components(mask, k=1):
    """Keep the k largest connected components."""
    if np.sum(mask) == 0:
        return mask, 0
    
    labeled_mask, num_features = label(mask)
    if num_features <= k:
        return mask, num_features
    
    # Calculate sizes of all components
    component_sizes = np.bincount(labeled_mask.ravel())
    # Ignore background (0)
    component_sizes[0] = 0
    
    # Get indices of k largest components
    largest_indices = np.argsort(component_sizes)[::-1][:k]
    
    # Create new mask
    new_mask = np.isin(labeled_mask, largest_indices).astype(np.uint8)
    return new_mask, num_features

def process_single_mask(mask_path, output_path, radius, k_components, save_nii=False):
    """Process a single mask: Closing -> Keep Largest K CCs."""
    mask = np.load(mask_path)
    
    # 1. Morphological Closing
    # Use a spherical structuring element
    structure = get_structuring_element(radius)
    closed_mask = binary_closing(mask, structure=structure).astype(np.uint8)
    
    # 2. Keep Largest K Components
    final_mask, original_cc_count = keep_largest_k_components(closed_mask, k=k_components)
    
    # Save .npy
    np.save(output_path, final_mask)
    
    # Optional: Save .nii.gz
    if save_nii:
        if not NIBABEL_AVAILABLE:
            print(f"Warning: nibabel not installed, skipping .nii.gz save for {output_path}")
        else:
            nii_path = output_path.replace('.npy', '.nii.gz')
            # Create a simple identity affine if we don't have the original reference
            affine = np.eye(4)
            nii_img = nib.Nifti1Image(final_mask, affine)
            nib.save(nii_img, nii_path)
        
    return original_cc_count

def main():
    parser = argparse.ArgumentParser(description='Perform Morphological Processing on Masks')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing refined .npy masks')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed masks')
    parser.add_argument('--class_name', type=str, required=True, help='Target class name (e.g., LV, PA, Coronary)')
    parser.add_argument('--radius', type=int, default=2, help='Radius for morphological closing')
    parser.add_argument('--save_nii', action='store_true', help='Save as .nii.gz as well')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine K based on class name
    # Rules from 总纲.txt
    # Coronary (9) -> 2
    # PV (10) -> 4
    # Others -> 1
    
    class_name_lower = args.class_name.lower()
    if 'coronary' in class_name_lower:
        k = 2
    elif 'pv' in class_name_lower:
        k = 4
    else:
        k = 1
        
    print(f"Processing class '{args.class_name}' with K={k} components, Closing Radius={args.radius}")
    
    files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.npy') and 'refined' in f])
    
    results = []
    
    for f in tqdm(files):
        input_path = os.path.join(args.input_dir, f)
        output_path = os.path.join(args.output_dir, f.replace('_refined.npy', '_processed.npy'))
        
        # Process
        original_cc_count = process_single_mask(input_path, output_path, args.radius, k, args.save_nii)
        
        # Verify result CC
        final_mask = np.load(output_path)
        _, final_cc_count = label(final_mask)
        
        results.append({
            'filename': f,
            'original_cc': original_cc_count,
            'final_cc': final_cc_count
        })
        
    # Save stats
    df = pd.DataFrame(results)
    stats_path = os.path.join(args.output_dir, 'morphology_stats.csv')
    df.to_csv(stats_path, index=False)
    
    print("\nProcessing Complete.")
    print(f"Mean Final CC: {df['final_cc'].mean():.4f}")
    print(f"Max Final CC: {df['final_cc'].max()}")
    print(f"Stats saved to {stats_path}")

if __name__ == "__main__":
    main()
