"""
Unified Verification Script for Phase 3.
Calculates all metrics:
1. Fidelity: Dice, HD95, ASD
2. Topology: Connected Components (CC), Anatomical Connectivity (PV-LA, LAA-LA, Cor-Myo)
3. Geometry: Isoperimetric Ratio (Surface Smoothness)
"""
import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import cc3d
from scipy.ndimage import distance_transform_edt
import multiprocessing
from tqdm import tqdm
import argparse
import warnings

# Add project root to path for surface_distance
sys.path.append("/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset")
try:
    from surface_distance import metrics
except ImportError:
    print("Warning: surface_distance module not found. HD95/ASD will be skipped.")
    metrics = None

# Suppress warnings
warnings.filterwarnings("ignore")

# Class mapping
CLASS_NAMES = {
    1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
    6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
}

def calculate_dice(mask1, mask2):
    """Calculate Dice coefficient."""
    intersection = np.logical_and(mask1, mask2).sum()
    sum_masks = mask1.sum() + mask2.sum()
    if sum_masks == 0:
        return 1.0 # Both empty
    return 2.0 * intersection / sum_masks

def check_connectivity(source_mask, target_mask, max_dist=5):
    """Check if source connects to target."""
    if source_mask.sum() == 0:
        return 1.0 # Empty is valid (e.g. no LAA) - Treat as 100% connected (0/0)
    if target_mask.sum() == 0:
        return 0.0 # Source exists but target missing
        
    labels_out, N = cc3d.connected_components(source_mask, connectivity=26, return_N=True)
    if N == 0:
        return 1.0
        
    dist_map = distance_transform_edt(1 - target_mask)
    
    connected_count = 0
    for i in range(1, N + 1):
        component = (labels_out == i)
        min_d = dist_map[component].min()
        if min_d <= max_dist: # Allow small gap
            connected_count += 1
            
    return connected_count / N

def compute_voxel_surface_area(mask):
    """Compute surface area by counting exposed voxel faces."""
    area = 0
    # x, y, z axis differences
    area += np.sum(np.abs(np.diff(mask, axis=0)))
    area += np.sum(np.abs(np.diff(mask, axis=1)))
    area += np.sum(np.abs(np.diff(mask, axis=2)))
    # Boundary faces
    area += np.sum(mask[0,:,:]) + np.sum(mask[-1,:,:])
    area += np.sum(mask[:,0,:]) + np.sum(mask[:,-1,:])
    area += np.sum(mask[:,:,0]) + np.sum(mask[:,:,-1])
    return area

def process_case(args):
    case_id, s3_path, gt_path = args
    results = {'case_id': case_id}
    
    try:
        # Load S3
        if s3_path.endswith('.npy'):
            s3_data = np.load(s3_path).astype(np.uint8)
        else:
            s3_img = nib.load(s3_path)
            s3_data = np.asanyarray(s3_img.dataobj).astype(np.uint8)
            
        # Load GT
        if gt_path.endswith('.npy'):
            gt_data = np.load(gt_path).astype(np.uint8)
        else:
            gt_img = nib.load(gt_path)
            gt_data = np.asanyarray(gt_img.dataobj).astype(np.uint8)
            
        if s3_data.shape != gt_data.shape:
            results['error'] = 'Shape mismatch'
            return results
            
        spacing_mm = (1.0, 1.0, 1.0) # Assume isotropic 1mm for metrics
        
        # --- Per Class Metrics ---
        for cid, cname in CLASS_NAMES.items():
            s3_bin = (s3_data == cid)
            gt_bin = (gt_data == cid)
            
            # 1. Dice
            results[f'Dice_{cname}'] = calculate_dice(s3_bin, gt_bin)
            
            # 2. CC Count (S3)
            _, N = cc3d.connected_components(s3_bin, connectivity=26, return_N=True)
            results[f'CC_{cname}'] = N
            
            # 3. Smoothness (IsoRatio)
            # S3
            if s3_bin.sum() > 0:
                s3_area = compute_voxel_surface_area(s3_bin.astype(np.int8))
                s3_vol = s3_bin.sum()
                results[f'IsoRatio_S3_{cname}'] = s3_area / (s3_vol ** (2/3))
            else:
                results[f'IsoRatio_S3_{cname}'] = np.nan
                
            # GT
            if gt_bin.sum() > 0:
                gt_area = compute_voxel_surface_area(gt_bin.astype(np.int8))
                gt_vol = gt_bin.sum()
                results[f'IsoRatio_GT_{cname}'] = gt_area / (gt_vol ** (2/3))
            else:
                results[f'IsoRatio_GT_{cname}'] = np.nan
                
            # 4. Surface Distance (HD95, ASD)
            if metrics is not None:
                if not s3_bin.any() and not gt_bin.any():
                    results[f'HD95_{cname}'] = 0.0
                    results[f'ASD_{cname}'] = 0.0
                elif not s3_bin.any() or not gt_bin.any():
                    results[f'HD95_{cname}'] = np.nan
                    results[f'ASD_{cname}'] = np.nan
                else:
                    surface_distances = metrics.compute_surface_distances(gt_bin, s3_bin, spacing_mm)
                    results[f'HD95_{cname}'] = metrics.compute_robust_hausdorff(surface_distances, 95)
                    asd_gt_to_pred, asd_pred_to_gt = metrics.compute_average_surface_distance(surface_distances)
                    results[f'ASD_{cname}'] = (asd_gt_to_pred + asd_pred_to_gt) / 2.0

        # --- Connectivity Metrics ---
        # PV(10) -> LA(2)
        results['Conn_PV_LA'] = check_connectivity(s3_data == 10, s3_data == 2)
        
        # LAA(8) -> LA(2)
        results['Conn_LAA_LA'] = check_connectivity(s3_data == 8, s3_data == 2)
        
        # Cor(9) -> Myo(1)/Ao(6)
        results['Conn_Cor_MyoAo'] = check_connectivity(s3_data == 9, np.logical_or(s3_data == 1, s3_data == 6))
        
        return results
        
    except Exception as e:
        results['error'] = str(e)
        return results

def main():
    parser = argparse.ArgumentParser(description="Unified Verification for Phase 3")
    parser.add_argument("--phase3_dir", required=True, help="Directory containing Phase 3 outputs (.nii.gz or .npy)")
    parser.add_argument("--gt_dir", required=True, help="Directory containing Ground Truth (.npy)")
    parser.add_argument("--output", default="verification_results_unified.csv", help="Output CSV file")
    args = parser.parse_args()
    
    # Find cases
    # Prefer .npy for speed if available in phase3_dir, else .nii.gz
    files = [f for f in os.listdir(args.phase3_dir) if f.endswith('.npy') and '_phase3' in f]
    if not files:
        files = [f for f in os.listdir(args.phase3_dir) if f.endswith('.nii.gz') and '_phase3' in f]
        
    case_ids = sorted(list(set([f.split('_phase3')[0] for f in files])))
    
    print(f"Verifying {len(case_ids)} cases...")
    print(f"Phase 3 Dir: {args.phase3_dir}")
    print(f"GT Dir: {args.gt_dir}")
    
    tasks = []
    for cid in case_ids:
        # Check for npy first
        s3_p = os.path.join(args.phase3_dir, f"{cid}_phase3.npy")
        if not os.path.exists(s3_p):
            s3_p = os.path.join(args.phase3_dir, f"{cid}_phase3.nii.gz")
            
        gt_p = os.path.join(args.gt_dir, f"{cid}.npy")
        tasks.append((cid, s3_p, gt_p))
        
    num_workers = min(16, multiprocessing.cpu_count())
    
    # Run processing
    all_results = []
    with multiprocessing.Pool(num_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_case, tasks), total=len(tasks)):
            all_results.append(res)
            
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\nDetailed results saved to {args.output}")
    
    # --- Summary Report ---
    print("\n" + "="*100)
    print("PHASE 3 VERIFICATION SUMMARY")
    print("="*100)
    
    # 1. Fidelity (Dice, HD95, ASD)
    print(f"\n{'Class':<12} | {'Dice':<8} | {'HD95':<8} | {'ASD':<8}")
    print("-" * 45)
    for cid, cname in CLASS_NAMES.items():
        dice = df[f'Dice_{cname}'].mean()
        hd95 = df[f'HD95_{cname}'].mean() if f'HD95_{cname}' in df.columns else np.nan
        asd = df[f'ASD_{cname}'].mean() if f'ASD_{cname}' in df.columns else np.nan
        print(f"{cname:<12} | {dice:<8.4f} | {hd95:<8.2f} | {asd:<8.2f}")
        
    # 2. Topology (CC, Connectivity)
    print("\n" + "-"*45)
    print("TOPOLOGY & CONNECTIVITY")
    print("-" * 45)
    print(f"PV -> LA Connectivity      : {df['Conn_PV_LA'].mean():.2%}")
    print(f"LAA -> LA Connectivity     : {df['Conn_LAA_LA'].mean():.2%}")
    print(f"Coronary -> Myo/Ao Attach  : {df['Conn_Cor_MyoAo'].mean():.2%}")
    print("-" * 45)
    print("Mean Connected Components (Phase 3):")
    for cid, cname in CLASS_NAMES.items():
        cc = df[f'CC_{cname}'].mean()
        print(f"  {cname:<12}: {cc:.2f}")
        
    # 3. Smoothness (IsoRatio)
    print("\n" + "-"*45)
    print("GEOMETRIC SMOOTHNESS (Isoperimetric Ratio)")
    print("-" * 45)
    print(f"{'Class':<12} | {'GT':<8} | {'S3':<8} | {'Change':<8}")
    for cid, cname in CLASS_NAMES.items():
        gt_iso = df[f'IsoRatio_GT_{cname}'].mean()
        s3_iso = df[f'IsoRatio_S3_{cname}'].mean()
        change = ((s3_iso - gt_iso) / gt_iso) * 100 if gt_iso != 0 else 0
        print(f"{cname:<12} | {gt_iso:<8.2f} | {s3_iso:<8.2f} | {change:+.1f}%")
        
    print("="*100)

if __name__ == "__main__":
    main()
