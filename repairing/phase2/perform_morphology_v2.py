"""
Script to perform morphological processing and cleaning on NeAR outputs (Phase 2).
This script applies binary closing and connected component filtering to refine the masks.
"""
import os
import argparse
import numpy as np
import nibabel as nib
import cc3d
from scipy.ndimage import binary_closing, binary_fill_holes, binary_dilation, binary_erosion
from tqdm import tqdm
import glob
import pandas as pd

# ================= Configuration Area =================
CONFIG = {
    # === Group 1: Large Organs ===
    # Myocardium: Special handling (False fill_holes here, handle in step3)
    1: {'name': 'Myocardium', 'radius': 2, 'fill_holes': False, 'strategy': 'top_k', 'k': 1, 'min_vol': 500},
    # LA: Allow 2, prevent truncation at PV entrance
    2: {'name': 'LA',         'radius': 2, 'fill_holes': True,  'strategy': 'top_k', 'k': 2, 'min_vol': 500},
    3: {'name': 'LV',         'radius': 2, 'fill_holes': True,  'strategy': 'top_k', 'k': 1, 'min_vol': 500},
    4: {'name': 'RA',         'radius': 2, 'fill_holes': True,  'strategy': 'top_k', 'k': 1, 'min_vol': 500},
    5: {'name': 'RV',         'radius': 2, 'fill_holes': True,  'strategy': 'top_k', 'k': 1, 'min_vol': 500},
    # Aorta: Allow 2 (Ascending/Descending aorta might be disconnected)
    6: {'name': 'Aorta',      'radius': 2, 'fill_holes': True,  'strategy': 'top_k', 'k': 2, 'min_vol': 500},
    # PA: Allow 2 (Left/Right PA bifurcation might be disconnected)
    7: {'name': 'PA',         'radius': 2, 'fill_holes': True,  'strategy': 'top_k', 'k': 2, 'min_vol': 200},
    
    # === Group 2: Fine Structures (No Fill Holes) ===
    # LAA: Left Atrial Appendage, single connected
    8: {'name': 'LAA',        'radius': 1, 'fill_holes': False, 'strategy': 'top_k', 'k': 1, 'min_vol': 50},
    # Coronary: Left+Right=2. Due to low Dice, use Top-2 to avoid accidental deletion
    9: {'name': 'Coronary',   'radius': 1, 'fill_holes': False, 'strategy': 'top_k', 'k': 2, 'min_vol': 50},
    # PV: 4 Pulmonary Veins
    10:{'name': 'PV',         'radius': 1, 'fill_holes': False, 'strategy': 'top_k', 'k': 4, 'min_vol': 50},
}

def step1_general_morphology(mask, cfg):
    """Step 1: General Closing + Large Organ Hole Filling"""
    # 1. Binary Closing (Stitch small cracks)
    if cfg['radius'] > 0:
        mask = binary_closing(mask, iterations=cfg['radius']).astype(np.uint8)
    
    # 2. Fill Holes (Only for configured large organs)
    if cfg['fill_holes']:
        mask = binary_fill_holes(mask).astype(np.uint8)
        
    return mask

def step2_cc_filtering(mask, cfg):
    """Step 2: Connected Components Analysis and Filtering"""
    # Use 26-connectivity (better for vessels)
    labels_out, N = cc3d.connected_components(mask, connectivity=26, return_N=True)
    
    if N == 0:
        return np.zeros_like(mask), 0, 0
        
    stats = cc3d.statistics(labels_out)
    voxel_counts = stats['voxel_counts'][1:] # Exclude Background 0
    
    # Sort: Large to Small
    sorted_indices = np.argsort(voxel_counts)[::-1]
    max_vol = voxel_counts[sorted_indices[0]]
    
    keep_labels = []
    
    # --- Execute Strategy ---
    if cfg['strategy'] == 'top_k':
        # Try to keep top K
        potential_k = min(N, cfg['k'])
        candidates = sorted_indices[:potential_k]
        
        # Double check: Prevent keeping tiny noise
        valid_candidates = []
        for idx in candidates:
            vol = voxel_counts[idx]
            # Must be larger than absolute volume threshold
            if vol > cfg['min_vol']:
                if idx == sorted_indices[0]:
                    valid_candidates.append(idx)
                else:
                    # Relative ratio check, prevent treating noise as 2nd component
                    if (vol / max_vol > 0.01):
                        valid_candidates.append(idx)
        
        # Fallback: Keep at least the largest 1 (Unless largest is smaller than min_vol, then all noise)
        if not valid_candidates and max_vol > 10: 
             valid_candidates.append(sorted_indices[0])
             
        keep_labels = [i + 1 for i in valid_candidates]

    # Reconstruct mask
    mask_filtered = np.isin(labels_out, keep_labels).astype(np.uint8)
    
    return mask_filtered, N, len(keep_labels)

def step3_per_class_special(mask, class_id):
    """Step 3: Class-specific Tricks"""
    
    # === Myocardium Trick: Dilate -> Fill -> Erode ===
    if class_id == 1: 
        # Prevent holes in Myocardium wall, but don't fill ventricle
        dilated = binary_dilation(mask, iterations=2).astype(np.uint8)
        filled = binary_fill_holes(dilated).astype(np.uint8)
        eroded = binary_erosion(filled, iterations=2).astype(np.uint8)
        return eroded
    
    # === Coronary Trick: Extra connection for breakpoints ===
    if class_id == 9:
        # Coronary is prone to breaking, extra closing (r=1) to connect branches
        mask = binary_closing(mask, iterations=1).astype(np.uint8)
        
    return mask

def process_directory(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    ref_dir = args.ref_dir
    target_class = args.target_class
    
    os.makedirs(output_dir, exist_ok=True)
    
    if target_class not in CONFIG:
        print(f"Error: Class {target_class} not found in CONFIG.")
        return

    cfg = CONFIG[target_class]
    
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    # Filter out non-refined files if mixed
    npy_files = [f for f in npy_files if 'refined' in f]
    
    print(f"--- Processing Class {target_class}: {cfg['name']} ---")
    print(f"Strategy: {cfg}")
    print(f"Found {len(npy_files)} files.")
    
    stats_list = []
    
    for npy_path in tqdm(npy_files):
        filename = os.path.basename(npy_path)
        # filename format: {id}_refined.npy
        file_id = filename.split('_refined')[0]
        
        # Load
        try:
            data = np.load(npy_path)
            mask = (data > 0.5).astype(np.uint8)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
            
        # Step 1: General
        mask_s1 = step1_general_morphology(mask, cfg)
        
        # Step 2: CC Filter
        mask_s2, original_cc, final_cc = step2_cc_filtering(mask_s1, cfg)
        
        # Step 3: Special
        mask_final = step3_per_class_special(mask_s2, target_class)
        
        # Save .npy (for Phase 3)
        np.save(os.path.join(output_dir, filename), mask_final)
        
        # Save .nii.gz (for Visualization)
        if ref_dir:
            # Find corresponding original nii file for affine
            candidates = [
                os.path.join(ref_dir, f"{file_id}.nii.gz"),
                os.path.join(ref_dir, "segmentations", f"{file_id}.nii.gz")
            ]
            ref_path = next((p for p in candidates if os.path.exists(p)), None)
            
            if ref_path:
                try:
                    ref_img = nib.load(ref_path)
                    # Ensure mask is same shape? 
                    # Refined mask is 256^3. Original might be different.
                    # We should NOT resize back here for Phase 3, but for visualization it's nice.
                    # BUT, if we save with ref affine, it will be misaligned if we don't resize.
                    # For now, let's just save with identity affine if shapes don't match, 
                    # OR just save the 256 version with a scaled affine?
                    # The user plan implies saving for check. 
                    # Let's save the 256 version with a simple affine to avoid complexity of resizing back.
                    
                    # Actually, better to just save with identity affine for quick check in Paraview/ITK
                    # unless we really want to overlay on original.
                    # Given the complexity, let's use identity affine for the 256 output.
                    
                    affine = np.eye(4)
                    new_img = nib.Nifti1Image(mask_final, affine)
                    nib.save(new_img, os.path.join(output_dir, f"{file_id}_clean.nii.gz"))
                except Exception as e:
                    print(f"Error saving NIfTI for {file_id}: {e}")
            else:
                # Save with identity
                affine = np.eye(4)
                new_img = nib.Nifti1Image(mask_final, affine)
                nib.save(new_img, os.path.join(output_dir, f"{file_id}_clean.nii.gz"))
        
        stats_list.append({
            'filename': filename,
            'original_cc': original_cc,
            'final_cc': final_cc
        })
        
    # Save Statistics CSV
    df = pd.DataFrame(stats_list)
    df.to_csv(os.path.join(output_dir, 'morphology_stats.csv'), index=False)
    print(f"Done. Stats saved to {os.path.join(output_dir, 'morphology_stats.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to folder with .npy files")
    parser.add_argument("--output_dir", required=True, help="Where to save result .npy and .nii.gz")
    parser.add_argument("--target_class", type=int, required=True)
    parser.add_argument("--ref_dir", default=None, help="Path to original dataset for NIfTI headers")
    
    args = parser.parse_args()
    process_directory(args)
