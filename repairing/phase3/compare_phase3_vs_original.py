"""
Script to compare Connected Components and Voxel Ratios between Phase 3 and Original data.
"""
import os
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
import argparse

# Class mapping
CLASS_NAMES = {
    1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
    6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
}

def get_phase3_stats(args):
    case_id, file_path = args
    try:
        # Load .npy (faster than nii.gz)
        mask = np.load(file_path)
        
        stats = []
        total_voxels = mask.size # 256^3
        
        for cid, cname in CLASS_NAMES.items():
            class_mask = (mask == cid)
            voxel_count = np.sum(class_mask)
            
            # We already have CC counts from the verification CSV, 
            # but let's recalculate or merge later. 
            # Calculating CC here might be slow. 
            # Let's just get Voxel Count here and use the existing CSV for CC.
            
            stats.append({
                'case_id': case_id,
                'class_id': cid,
                'class_name': cname,
                's3_voxel_count': voxel_count,
                's3_ratio': voxel_count / total_voxels
            })
            
        return stats
    except Exception as e:
        return []

def main():
    phase3_dir = "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/phase3/output"
    orig_csv_path = "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/original_cc_full_dataset.csv"
    s3_verif_csv = "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/phase3/phase3_verification_full.csv"
    
    # 1. Load Original Stats
    print("Loading Original Stats...")
    df_orig = pd.read_csv(orig_csv_path)
    orig_stats = df_orig.groupby('class_id').agg({
        'cc_count': 'mean',
        'voxel_count': 'mean' # This is raw voxel count, not ratio. Hard to compare without total volume.
    }).reset_index()
    
    # 2. Get Phase 3 Voxel Stats
    print("Calculating Phase 3 Voxel Stats...")
    files = [f for f in os.listdir(phase3_dir) if f.endswith('.npy')]
    tasks = [(f.split('_phase3')[0], os.path.join(phase3_dir, f)) for f in files]
    
    num_workers = min(16, multiprocessing.cpu_count())
    s3_voxel_data = []
    
    with multiprocessing.Pool(num_workers) as pool:
        for res in tqdm(pool.imap_unordered(get_phase3_stats, tasks), total=len(tasks)):
            s3_voxel_data.extend(res)
            
    df_s3_voxels = pd.DataFrame(s3_voxel_data)
    s3_voxel_stats = df_s3_voxels.groupby('class_id').agg({
        's3_ratio': 'mean'
    }).reset_index()
    
    # 3. Get Phase 3 CC Stats (from verification CSV)
    # We need to parse the repaired CSV or just re-read it if it's clean
    # The user said it was repaired.
    print("Loading Phase 3 CC Stats...")
    # We'll use the repaired logic just in case, or assume it's fixed.
    # Actually, let's just use the columns we know exist.
    try:
        df_s3_cc = pd.read_csv(s3_verif_csv)
        # Calculate mean CC per class
        s3_cc_means = {}
        for cid, cname in CLASS_NAMES.items():
            col = f'CC_{cname}'
            if col in df_s3_cc.columns:
                # Force numeric
                s3_cc_means[cid] = pd.to_numeric(df_s3_cc[col], errors='coerce').mean()
    except:
        print("Error reading verification CSV. Using placeholder.")
        s3_cc_means = {}

    # 4. Combine and Print
    print("\n" + "="*80)
    print(f"{'Class':<12} | {'Orig CC':<10} | {'S3 CC':<10} | {'Change':<10} || {'S3 Ratio (%)':<15}")
    print("-" * 80)
    
    # User provided Original Ratios for reference (hardcoded for comparison)
    # 1: 2.57%, 2: 1.65%, 3: 2.65%, 4: 1.90%, 5: 3.20%, 6: 2.20%, 7: 0.92%, 8: 0.21%, 9: 0.12%, 10: 0.50%
    orig_ratios = {
        1: 2.57, 2: 1.65, 3: 2.65, 4: 1.90, 5: 3.20,
        6: 2.20, 7: 0.92, 8: 0.21, 9: 0.12, 10: 0.50
    }
    
    for cid in sorted(CLASS_NAMES.keys()):
        cname = CLASS_NAMES[cid]
        
        # CC
        orig_cc = orig_stats.loc[orig_stats['class_id'] == cid, 'cc_count'].values[0]
        s3_cc = s3_cc_means.get(cid, 0.0)
        cc_change = s3_cc - orig_cc
        
        # Ratio
        s3_ratio = s3_voxel_stats.loc[s3_voxel_stats['class_id'] == cid, 's3_ratio'].values[0] * 100
        orig_ratio = orig_ratios.get(cid, 0.0)
        ratio_change = s3_ratio - orig_ratio
        
        print(f"{cname:<12} | {orig_cc:<10.4f} | {s3_cc:<10.4f} | {cc_change:+.4f}     || {s3_ratio:<10.4f}% (vs {orig_ratio}%)")
        
    print("="*80)

if __name__ == "__main__":
    main()
