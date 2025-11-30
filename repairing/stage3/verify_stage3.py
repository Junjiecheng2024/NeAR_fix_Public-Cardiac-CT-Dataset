import os
import numpy as np
import nibabel as nib
import cc3d
import pandas as pd
from scipy.ndimage import distance_transform_edt
import multiprocessing
from tqdm import tqdm
import argparse

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
        return True # Empty is valid (e.g. no LAA)
    if target_mask.sum() == 0:
        return False # Source exists but target missing
        
    labels_out, N = cc3d.connected_components(source_mask, connectivity=26, return_N=True)
    dist_map = distance_transform_edt(1 - target_mask)
    
    connected_count = 0
    for i in range(1, N + 1):
        component = (labels_out == i)
        min_d = dist_map[component].min()
        if min_d <= max_dist: # Allow small gap if any
            connected_count += 1
            
    # Return ratio of connected components
    return connected_count / N

def process_case(args):
    case_id, stage3_path, original_dir = args
    
    results = {'case_id': case_id}
    
    try:
        # Load Stage 3
        s3_img = nib.load(stage3_path)
        s3_data = np.asanyarray(s3_img.dataobj).astype(np.uint8)
        
        # Load Ground Truth (npy)
        gt_path = os.path.join(original_dir, f"{case_id}.npy")
        
        if os.path.exists(gt_path):
            gt_data = np.load(gt_path).astype(np.uint8)
            
            if s3_data.shape != gt_data.shape:
                results['shape_mismatch'] = True
                print(f"Shape mismatch: S3 {s3_data.shape} vs GT {gt_data.shape}")
            else:
                results['shape_mismatch'] = False
                # Calculate Dice per class
                for cid, cname in CLASS_NAMES.items():
                    d = calculate_dice(s3_data == cid, gt_data == cid)
                    results[f'Dice_{cname}'] = d
        else:
            results['missing_orig'] = True
            print(f"Missing GT: {gt_path}")

            
        # Check Anatomy (on S3 data)
        # 1. CC Counts
        for cid, cname in CLASS_NAMES.items():
            mask = (s3_data == cid).astype(np.uint8)
            _, N = cc3d.connected_components(mask, connectivity=26, return_N=True)
            results[f'CC_{cname}'] = N
            
        # 2. Connectivity
        # PV(10) -> LA(2)
        pv_conn = check_connectivity(s3_data == 10, s3_data == 2)
        results['Conn_PV_LA'] = pv_conn
        
        # LAA(8) -> LA(2)
        laa_conn = check_connectivity(s3_data == 8, s3_data == 2)
        results['Conn_LAA_LA'] = laa_conn
        
        # Coronary(9) -> Myo(1)/Ao(6)
        cor_conn = check_connectivity(s3_data == 9, np.logical_or(s3_data == 1, s3_data == 6))
        results['Conn_Cor_MyoAo'] = cor_conn
        
    except Exception as e:
        print(f"Error {case_id}: {e}")
        return None
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3_dir", required=True)
    parser.add_argument("--original_dir", required=True)
    args = parser.parse_args()
    
    files = [f for f in os.listdir(args.stage3_dir) if f.endswith('.nii.gz')]
    case_ids = [f.split('_stage3')[0] for f in files]
    
    print(f"Verifying {len(case_ids)} cases...")
    
    tasks = [(cid, os.path.join(args.stage3_dir, f"{cid}_stage3.nii.gz"), args.original_dir) for cid in case_ids]
    
    num_workers = min(16, multiprocessing.cpu_count())
    all_results = []
    
    # Initialize CSV with headers
    csv_file = "stage3_verification_full.csv"
    first_write = True
    
    with multiprocessing.Pool(num_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_case, tasks), total=len(tasks)):
            if res:
                df_chunk = pd.DataFrame([res])
                if first_write:
                    df_chunk.to_csv(csv_file, index=False, mode='w')
                    first_write = False
                else:
                    df_chunk.to_csv(csv_file, index=False, mode='a', header=False)
    
    # Read back for summary
    df = pd.read_csv(csv_file)
    
    # Summary
    print("\n--- Verification Summary ---")
    if 'shape_mismatch' in df.columns and df['shape_mismatch'].any():
        print(f"Warning: {df['shape_mismatch'].sum()} cases had shape mismatch (Dice skipped).")
        
    print("\nMean Dice (vs Original):")
    for cid, cname in CLASS_NAMES.items():
        col = f'Dice_{cname}'
        if col in df.columns:
            print(f"{cname:<15}: {df[col].mean():.4f}")
            
    print("\nMean CC Count (Stage 3):")
    for cid, cname in CLASS_NAMES.items():
        col = f'CC_{cname}'
        if col in df.columns:
            print(f"{cname:<15}: {df[col].mean():.4f}")
            
    print("\nConnectivity (Ratio of connected components):")
    print(f"PV -> LA       : {df['Conn_PV_LA'].mean():.4f}")
    print(f"LAA -> LA      : {df['Conn_LAA_LA'].mean():.4f}")
    print(f"Cor -> Myo/Ao  : {df['Conn_Cor_MyoAo'].mean():.4f}")

if __name__ == "__main__":
    main()
