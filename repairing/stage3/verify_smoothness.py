import os
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
import argparse

import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Class mapping
CLASS_NAMES = {
    1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
    6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
}

def compute_voxel_surface_area(mask):
    """
    Compute surface area by counting exposed voxel faces.
    Each voxel has 6 faces. If a neighbor is 0, that face is exposed.
    Area = Number of exposed faces.
    """
    # Shift in 6 directions
    # x+1, x-1, y+1, y-1, z+1, z-1
    # XOR with shift gives boundary.
    # Actually, simpler:
    # For each axis, diff. Non-zero diff means boundary.
    # Area = sum(abs(diff(x))) + sum(abs(diff(y))) + sum(abs(diff(z)))
    # This counts faces.
    
    area = 0
    # x-axis
    area += np.sum(np.abs(np.diff(mask, axis=0)))
    # y-axis
    area += np.sum(np.abs(np.diff(mask, axis=1)))
    # z-axis
    area += np.sum(np.abs(np.diff(mask, axis=2)))
    
    # Add boundary faces (first and last slice if non-zero)
    area += np.sum(mask[0,:,:]) + np.sum(mask[-1,:,:])
    area += np.sum(mask[:,0,:]) + np.sum(mask[:,-1,:])
    area += np.sum(mask[:,:,0]) + np.sum(mask[:,:,-1])
    
    return area

def process_case(args):
    case_id, s3_path, gt_path = args
    results = []
    
    try:
        # Load S3
        if s3_path.endswith('.npy'):
            s3_mask = np.load(s3_path)
        else:
            import nibabel as nib
            s3_mask = np.asanyarray(nib.load(s3_path).dataobj).astype(np.uint8)
            
        # Load GT
        gt_mask = np.load(gt_path).astype(np.uint8)
        
        if s3_mask.shape != gt_mask.shape:
            return []
            
        for cid, cname in CLASS_NAMES.items():
            # S3
            s3_bin = (s3_mask == cid).astype(np.int8) # int8 for diff
            if s3_bin.sum() > 0:
                s3_area = compute_voxel_surface_area(s3_bin)
                s3_vol = s3_bin.sum()
                s3_iso = s3_area / (s3_vol ** (2/3)) if s3_vol > 0 else np.nan
            else:
                s3_area = np.nan
                s3_iso = np.nan
                
            # GT
            gt_bin = (gt_mask == cid).astype(np.int8)
            if gt_bin.sum() > 0:
                gt_area = compute_voxel_surface_area(gt_bin)
                gt_vol = gt_bin.sum()
                gt_iso = gt_area / (gt_vol ** (2/3)) if gt_vol > 0 else np.nan
            else:
                gt_area = np.nan
                gt_iso = np.nan
                
            results.append({
                'case_id': case_id,
                'class_id': cid,
                'class_name': cname,
                'S3_IsoRatio': s3_iso,
                'GT_IsoRatio': gt_iso,
                'S3_Area': s3_area,
                'GT_Area': gt_area
            })
            
        return results
    except Exception as e:
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    args = parser.parse_args()
    
    files = [f for f in os.listdir(args.stage3_dir) if f.endswith('.npy')]
    case_ids = [f.split('_stage3')[0] for f in files]
    
    # Limit to 20 cases for speed -> REMOVED
    # case_ids = case_ids[:20] 
    print(f"Processing {len(case_ids)} cases for Smoothness (Isoperimetric Ratio)...")
    
    tasks = []
    for cid in case_ids:
        s3_p = os.path.join(args.stage3_dir, f"{cid}_stage3.npy")
        gt_p = os.path.join(args.gt_dir, f"{cid}.npy")
        tasks.append((cid, s3_p, gt_p))
        
    num_workers = min(16, multiprocessing.cpu_count())
    all_results = []
    
    with multiprocessing.Pool(num_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_case, tasks), total=len(tasks)):
            if res:
                all_results.extend(res)
                
    df = pd.DataFrame(all_results)
    
    # Summary
    print("\n" + "="*80)
    print(f"{'Class':<12} | {'GT IsoRatio':<12} | {'S3 IsoRatio':<12} | {'Change':<10} | {'Smoother?':<10}")
    print("-" * 80)
    
    summary = df.groupby('class_name').mean(numeric_only=True).reset_index()
    summary['class_id'] = summary['class_name'].map({v: k for k, v in CLASS_NAMES.items()})
    summary = summary.sort_values('class_id')
    
    for _, row in summary.iterrows():
        gt = row['GT_IsoRatio']
        s3 = row['S3_IsoRatio']
        change = s3 - gt
        # Lower IsoRatio = Smoother/More Compact
        better = "YES" if change < 0 else "NO"
        print(f"{row['class_name']:<12} | {gt:<12.4f} | {s3:<12.4f} | {change:+.4f}     | {better:<10}")
        
    print("="*80)
    print("Metric: Isoperimetric Ratio = Surface Area / Volume^(2/3).")
    print("Lower value indicates a smoother, more compact shape (Sphere ~ 4.8).")
    print("Higher value indicates roughness, fragmentation, or complex branching.")

if __name__ == "__main__":
    main()
