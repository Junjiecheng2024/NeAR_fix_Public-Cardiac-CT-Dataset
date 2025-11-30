"""
Script to verify topological correctness (connectivity) of the original dataset.
Used for comparison with Phase 3 results.
"""
import os
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
import argparse
import cc3d
from scipy.ndimage import binary_dilation

# Class mapping
CLASS_NAMES = {
    1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
    6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
}

def check_connectivity(source_mask, target_mask, dilation=1):
    """
    Check what percentage of source components are connected to target mask.
    """
    if source_mask.sum() == 0:
        return 1.0 # Empty source is technically "not disconnected"
        
    labels, N = cc3d.connected_components(source_mask, connectivity=26, return_N=True)
    if N == 0:
        return 1.0
        
    # Dilate target to allow for slight gaps (adjacency)
    if dilation > 0:
        struct = np.ones((3,3,3), dtype=bool)
        target_dilated = binary_dilation(target_mask, structure=struct, iterations=dilation)
    else:
        target_dilated = target_mask
        
    connected_count = 0
    for i in range(1, N+1):
        component = (labels == i)
        # Check overlap with dilated target
        if np.any(component & target_dilated):
            connected_count += 1
            
    return connected_count / N

def process_case(args):
    case_id, gt_dir = args
    results = {}
    
    try:
        # Load necessary masks
        # GT dir contains {case_id}.npy which is multi-class? 
        # Wait, previous scripts said near_format_data/shape contains single class?
        # Let's check. The user said:
        # /home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset_backup/dataset/near_format_data/shape/
        # But wait, inference_and_evaluate.py loaded {sample_id}.npy from there.
        # If it's multi-class, we just load once.
        
        gt_path = os.path.join(gt_dir, f"{case_id}.npy")
        if not os.path.exists(gt_path):
            return None
            
        gt_mask = np.load(gt_path).astype(np.uint8)
        
        # Extract binary masks
        mask_la = (gt_mask == 2)
        mask_laa = (gt_mask == 8)
        mask_pv = (gt_mask == 10)
        mask_cor = (gt_mask == 9)
        mask_myo = (gt_mask == 1)
        mask_ao = (gt_mask == 6)
        
        # 1. PV -> LA Connectivity
        results['Conn_PV_LA'] = check_connectivity(mask_pv, mask_la, dilation=2)
        
        # 2. LAA -> LA Connectivity
        results['Conn_LAA_LA'] = check_connectivity(mask_laa, mask_la, dilation=2)
        
        # 3. Coronary -> Myo/Ao Connectivity
        # Coronary should attach to Myocardium OR Aorta
        target_cor = mask_myo | mask_ao
        results['Conn_Cor_MyoAo'] = check_connectivity(mask_cor, target_cor, dilation=2)
        
        results['case_id'] = case_id
        return results
        
    except Exception as e:
        print(f"Error processing {case_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True, help="Directory containing .npy GT files")
    args = parser.parse_args()
    
    files = [f for f in os.listdir(args.gt_dir) if f.endswith('.npy')]
    case_ids = [f.replace('.npy', '') for f in files]
    
    print(f"Verifying topology for {len(case_ids)} original cases...")
    
    tasks = [(cid, args.gt_dir) for cid in case_ids]
    
    num_workers = min(16, multiprocessing.cpu_count())
    all_results = []
    
    with multiprocessing.Pool(num_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_case, tasks), total=len(tasks)):
            if res:
                all_results.append(res)
                
    df = pd.DataFrame(all_results)
    
    print("\n--- Original Topology Summary ---")
    print(f"PV -> LA       : {df['Conn_PV_LA'].mean():.4f}")
    print(f"LAA -> LA      : {df['Conn_LAA_LA'].mean():.4f}")
    print(f"Cor -> Myo/Ao  : {df['Conn_Cor_MyoAo'].mean():.4f}")
    
    # Save detailed results
    df.to_csv("original_topology_verification.csv", index=False)
    print("Detailed results saved to original_topology_verification.csv")

if __name__ == "__main__":
    main()
