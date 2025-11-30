import os
import sys
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
import argparse

# Add project root to path to import surface_distance
sys.path.append("/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset")
from surface_distance import metrics

# Class mapping
CLASS_NAMES = {
    1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
    6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
}

def compute_metrics(args):
    case_id, s3_path, gt_path = args
    results = []
    
    try:
        if not os.path.exists(s3_path) or not os.path.exists(gt_path):
            return []
            
        # Load masks
        # S3 is .nii.gz or .npy? The output folder has both. .npy is faster.
        # But verify_stage3 used .nii.gz. Let's use .npy if available.
        s3_npy = s3_path.replace('.nii.gz', '.npy')
        if os.path.exists(s3_npy):
            s3_mask = np.load(s3_npy)
        else:
            import nibabel as nib
            s3_mask = np.asanyarray(nib.load(s3_path).dataobj).astype(np.uint8)
            
        gt_mask = np.load(gt_path).astype(np.uint8)
        
        if s3_mask.shape != gt_mask.shape:
            return []
            
        # Spacing: assume 1.0 (results in voxels)
        spacing_mm = (1.0, 1.0, 1.0)
        
        for cid, cname in CLASS_NAMES.items():
            # Create binary masks
            pred_bin = (s3_mask == cid)
            gt_bin = (gt_mask == cid)
            
            # Skip if both empty
            if not pred_bin.any() and not gt_bin.any():
                continue
                
            # Compute surface distances
            surface_distances = metrics.compute_surface_distances(
                gt_bin, pred_bin, spacing_mm
            )
            
            # HD95
            hd95 = metrics.compute_robust_hausdorff(surface_distances, 95)
            
            # ASD
            asd_gt_to_pred, asd_pred_to_gt = metrics.compute_average_surface_distance(surface_distances)
            asd = (asd_gt_to_pred + asd_pred_to_gt) / 2.0
            
            results.append({
                'case_id': case_id,
                'class_id': cid,
                'class_name': cname,
                'HD95': hd95,
                'ASD': asd
            })
            
        return results
        
    except Exception as e:
        # print(f"Error {case_id}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    args = parser.parse_args()
    
    # Find cases
    files = [f for f in os.listdir(args.stage3_dir) if f.endswith('.npy')]
    case_ids = [f.split('_stage3')[0] for f in files]
    
    print(f"Processing {len(case_ids)} cases for Surface Distance metrics...")
    
    tasks = []
    for cid in case_ids:
        s3_p = os.path.join(args.stage3_dir, f"{cid}_stage3.npy")
        gt_p = os.path.join(args.gt_dir, f"{cid}.npy")
        tasks.append((cid, s3_p, gt_p))
        
    num_workers = min(16, multiprocessing.cpu_count())
    all_metrics = []
    
    # Initialize CSV
    csv_file = "stage3_surface_metrics.csv"
    first_write = True
    
    with multiprocessing.Pool(num_workers) as pool:
        for res in tqdm(pool.imap_unordered(compute_metrics, tasks), total=len(tasks)):
            if res:
                all_metrics.extend(res)
                # Incremental save? Maybe just save at end for metrics, 
                # or chunked. Let's do chunked to be safe.
                df_chunk = pd.DataFrame(res)
                if not df_chunk.empty:
                    if first_write:
                        df_chunk.to_csv(csv_file, index=False, mode='w')
                        first_write = False
                    else:
                        df_chunk.to_csv(csv_file, index=False, mode='a', header=False)
    
    # Summary
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        # Handle Inf (empty prediction vs gt)
        # Replace inf with NaN for mean calculation
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        summary = df.groupby('class_name').agg({
            'HD95': 'mean',
            'ASD': 'mean'
        }).reset_index()
        
        # Sort by class ID order
        summary['class_id'] = summary['class_name'].map({v: k for k, v in CLASS_NAMES.items()})
        summary = summary.sort_values('class_id')
        
        print("\n" + "="*60)
        print(f"{'Class':<12} | {'HD95 (vox)':<12} | {'ASD (vox)':<12}")
        print("-" * 60)
        for _, row in summary.iterrows():
            print(f"{row['class_name']:<12} | {row['HD95']:<12.4f} | {row['ASD']:<12.4f}")
        print("="*60)
        print("(Note: Units are voxels in 256x256x256 space)")

if __name__ == "__main__":
    main()
