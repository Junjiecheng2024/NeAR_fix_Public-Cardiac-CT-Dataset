"""
Script to evaluate NeAR Phase 3 Repair Quality.
Calculates metrics: Dice, CC Count, Volume Ratio, HD95, ASD, Presence Rate.
Comparisons:
1. Phase 1 (Raw) vs Ground Truth (Accuracy Before)
2. Phase 3 (Repaired) vs Ground Truth (Accuracy After)
3. Phase 1 vs Phase 3 (Change Magnitude)
"""

import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
import cc3d
from scipy.ndimage import distance_transform_edt, binary_erosion
from tqdm import tqdm
import glob
import multiprocessing
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

CLASS_NAMES = {
    1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
    6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
}

def get_surface_distance(mask1, mask2, spacing=(1.0, 1.0, 1.0)):
    """
    Compute Surface Distances using EDT.
    """
    # Extract surfaces
    border1 = mask1 ^ binary_erosion(mask1)
    border2 = mask2 ^ binary_erosion(mask2)
    
    if border1.sum() == 0 or border2.sum() == 0:
        return np.nan, np.nan
        
    # EDT
    dt1 = distance_transform_edt(~border1, sampling=spacing)
    dt2 = distance_transform_edt(~border2, sampling=spacing)
    
    # Dists from 1 to 2
    d1_to_2 = dt2[border1]
    # Dists from 2 to 1
    d2_to_1 = dt1[border2]
    
    all_dists = np.concatenate([d1_to_2, d2_to_1])
    
    hd95 = np.percentile(all_dists, 95)
    asd = all_dists.mean()
    
    return hd95, asd

def compute_metrics(pred, gt, spacing=(1,1,1)):
    """
    Compute Dice, Vol, CC. 
    Metrics requiring surface (HD95, ASD) are computed if possible.
    """
    metrics = {}
    
    # 0. Presence
    pred_exists = pred.sum() > 0
    gt_exists = gt.sum() > 0
    metrics['gt_exists'] = gt_exists
    metrics['pred_exists'] = pred_exists
    
    if not gt_exists and not pred_exists:
        return {
            'dice': 1.0, 'hd95': 0.0, 'asd': 0.0, 
            'vol_pred': 0, 'vol_gt': 0, 'cc_pred': 0, 'cc_gt': 0
        }
    
    if not gt_exists and pred_exists:
         # False Positive
         return {
            'dice': 0.0, 'hd95': np.nan, 'asd': np.nan, 
            'vol_pred': pred.sum(), 'vol_gt': 0, 'cc_pred': 0, 'cc_gt': 0
        }
        
    if gt_exists and not pred_exists:
        # False Negative
        return {
            'dice': 0.0, 'hd95': np.nan, 'asd': np.nan, 
            'vol_pred': 0, 'vol_gt': gt.sum(), 'cc_pred': 0, 'cc_gt': 0
        }

    # 1. Dice
    intersection = np.logical_and(pred, gt).sum()
    dice = 2.0 * intersection / (pred.sum() + gt.sum())
    metrics['dice'] = dice
    
    # 2. Volume
    metrics['vol_pred'] = int(pred.sum())
    metrics['vol_gt'] = int(gt.sum())
    
    # 3. Connected Components
    _, n_cc_pred = cc3d.connected_components(pred, return_N=True)
    _, n_cc_gt = cc3d.connected_components(gt, return_N=True)
    metrics['cc_pred'] = n_cc_pred
    metrics['cc_gt'] = n_cc_gt
    
    # 4. Surface Metrics (HD95, ASD)
    # Only compute if volumes are not too tiny to avoid errors
    if pred.sum() > 0 and gt.sum() > 0:
        try:
            hd95, asd = get_surface_distance(pred, gt, spacing)
            metrics['hd95'] = hd95
            metrics['asd'] = asd
        except:
            metrics['hd95'] = np.nan
            metrics['asd'] = np.nan
    else:
        metrics['hd95'] = np.nan
        metrics['asd'] = np.nan
        
    return metrics

def process_case(args_tuple):
    case_id, data_root, ref_path, spacing = args_tuple
    
    results = []
    
    try:
        # Load GT
        if ref_path and os.path.exists(ref_path):
            gt_nii = nib.load(ref_path)
            gt_data = gt_nii.get_fdata().astype(np.uint8)
            # spacing = gt_nii.header.get_zooms()[:3] # Use spacing from file
        else:
            return [] # Skip if no GT
            
        # Load Phase 3
        phase3_path = os.path.join(data_root, "repaired_phase3", f"{case_id}_phase3.nii.gz")
        if os.path.exists(phase3_path):
            p3_data = nib.load(phase3_path).get_fdata().astype(np.uint8)
        else:
            p3_data = np.zeros_like(gt_data)
            
        # Load Phase 1 (Collection of masks)
        # Assuming we need to construct it or load per class
        # Ideally we compare Per-Class
        
        for cls_id in range(1, 11):
            cls_name = CLASS_NAMES[cls_id]
            cls_lower = cls_name.lower()
            
            # --- Load Masks ---
            mask_gt = (gt_data == cls_id).astype(np.uint8)
            mask_p3 = (p3_data == cls_id).astype(np.uint8)
            
            # Phase 1: Global Inference Output
            p1_path = os.path.join(data_root, f"{cls_lower}_global", f"{case_id}_mask.npy")
            if os.path.exists(p1_path):
                mask_p1 = np.load(p1_path)
                mask_p1 = (mask_p1 > 0.5).astype(np.uint8)
            else:
                mask_p1 = np.zeros((256, 256, 256), dtype=np.uint8)

            # Phase 2: Morphology Output
            p2_path = os.path.join(data_root, f"{cls_lower}_morph", f"{case_id}_mask.npy")
            if os.path.exists(p2_path):
                mask_p2 = np.load(p2_path)
                mask_p2 = (mask_p2 > 0.5).astype(np.uint8)
            else:
                # If Phase 2 missing, assume same as Phase 1 (skipped)
                mask_p2 = mask_p1.copy()
            
            # RESIZE GT to 256 if needed
            if mask_gt.shape != (256, 256, 256):
                 # Use scipy zoom. Nearest neighbor.
                 zoom_fac = np.array([256, 256, 256]) / np.array(mask_gt.shape)
                 from scipy.ndimage import zoom
                 mask_gt = zoom(mask_gt, zoom_fac, order=0)
                 mask_gt = (mask_gt > 0.5).astype(np.uint8)
            
            # --- Metrics: P1 vs GT (Inference) ---
            m_p1 = compute_metrics(mask_p1, mask_gt, spacing)
            
            # --- Metrics: P2 vs GT (Morphology) ---
            m_p2 = compute_metrics(mask_p2, mask_gt, spacing)
            
            # --- Metrics: P3 vs GT (Fusion) ---
            m_p3 = compute_metrics(mask_p3, mask_gt, spacing)
            
            # --- Change Ratios ---
            # Vol Change P1 -> P2
            if m_p1['vol_pred'] > 0:
                p2_change_ratio = (m_p2['vol_pred'] - m_p1['vol_pred']) / m_p1['vol_pred']
            else:
                p2_change_ratio = 0.0
            
            # Vol Change P2 -> P3
            if m_p2['vol_pred'] > 0:
                p3_change_ratio = (m_p3['vol_pred'] - m_p2['vol_pred']) / m_p2['vol_pred']
            else:
                p3_change_ratio = 0.0
                
            entry = {
                'case_id': case_id,
                'class_id': cls_id,
                'class_name': cls_name,
                
                # Phase 1 Scores
                'p1_dice': m_p1['dice'],
                'p1_hd95': m_p1['hd95'],
                'p1_asd': m_p1['asd'],
                'p1_cc': m_p1['cc_pred'],
                'p1_vol': m_p1['vol_pred'],
                
                # Phase 2 Scores
                'p2_dice': m_p2['dice'],
                'p2_hd95': m_p2['hd95'],
                'p2_asd': m_p2['asd'],
                'p2_cc': m_p2['cc_pred'],
                'p2_vol': m_p2['vol_pred'],
                
                # Phase 3 Scores
                'p3_dice': m_p3['dice'],
                'p3_hd95': m_p3['hd95'],
                'p3_asd': m_p3['asd'],
                'p3_cc': m_p3['cc_pred'],
                'p3_vol': m_p3['vol_pred'],
                
                # Changes
                'p2_vol_change': p2_change_ratio,
                'p3_vol_change': p3_change_ratio,
                'p2_cc_change': m_p2['cc_pred'] - m_p1['cc_pred'],
                'p3_cc_change': m_p3['cc_pred'] - m_p2['cc_pred'],
                'delta_dice_p2': m_p2['dice'] - m_p1['dice'],
                'delta_dice_p3': m_p3['dice'] - m_p1['dice']
            }
            results.append(entry)
            
    except Exception as e:
        print(f"Error {case_id}: {e}")
        return []

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Root of dataset e.g. /scratch/.../dataset")
    parser.add_argument("--gt_root", required=True, help="Folder containing original GT .nii.gz")
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    
    # Discover cases
    # Look in repaired_phase3
    p3_dir = os.path.join(args.data_root, "repaired_phase3")
    files = glob.glob(os.path.join(p3_dir, "*_phase3.nii.gz"))
    case_ids = [os.path.basename(f).split('_phase3')[0] for f in files]
    case_ids = sorted(list(set(case_ids)))
    
    print(f"Found {len(case_ids)} cases in {p3_dir}")
    
    # Prepare jobs
    # Look for GT file
    tasks = []
    for cid in case_ids:
        # Check GT paths
        # Try various patterns found in user dataset
        candidates = [
            os.path.join(args.gt_root, f"{cid}.nii.gz"),
            os.path.join(args.gt_root, "segmentations", f"{cid}.nii.gz"),
            os.path.join(args.gt_root, f"{cid}.nii.img.nii.gz"),  # Found pattern
            os.path.join(args.gt_root, "segmentations", f"{cid}.nii.img.nii.gz") 
        ]
        
        gt_path = next((p for p in candidates if os.path.exists(p)), None)
        
        if gt_path:
             tasks.append((cid, args.data_root, gt_path, (1,1,1)))
        
    print(f"Found GT for {len(tasks)} cases. Starting evaluation...")
    
    cpu_count = min(32, multiprocessing.cpu_count())
    all_results = []
    
    with multiprocessing.Pool(cpu_count) as pool:
        for res in tqdm(pool.imap_unordered(process_case, tasks), total=len(tasks)):
            all_results.extend(res)
            
    # Save Raw
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_csv, index=False)
    
    # Summary
    if not df.empty:
        print("\n=== Summary by Class ===")
        summary = df.groupby(['class_id', 'class_name'])[[
            'p1_dice', 'p2_dice', 'p3_dice', 
            'p1_hd95', 'p2_hd95', 'p3_hd95',
            'p1_cc', 'p2_cc', 'p3_cc',
            'p2_vol_change', 'p3_vol_change'
        ]].mean().reset_index()
        
        print(summary.to_string())
        summary_file = args.output_csv.replace(".csv", "_summary.csv")
        summary.to_csv(summary_file, index=False)
        print(f"\nFull results saved to {args.output_csv}")
        print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main()
