"""
Script to calculate connected component statistics for Phase 2 outputs.
Used to verify the effectiveness of morphological cleaning.
"""
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure
from tqdm import tqdm
import pandas as pd
import multiprocessing
import csv
import glob
import cc3d

# Class mapping
CLASS_NAMES = {
    1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
    6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
}

BASE_DIR = "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/phase2"

def process_single_file(args):
    """Process a single .npy file and return CC counts for the target class."""
    path, filename, class_id, class_name = args
    results = []
    
    # Define 26-connectivity structure (Face + Edge + Corner)
    structure = generate_binary_structure(3, 3)
    
    try:
        # Load .npy mask
        mask = np.load(path)
        # Ensure binary
        mask = (mask > 0.5).astype(np.uint8)
        
        voxel_total = mask.sum()
        
        if voxel_total > 0:
            # Use cc3d for faster processing
            # connectivity=26 matches scipy structure(3,3)
            labeled_array, n_components = cc3d.connected_components(mask, connectivity=26, return_N=True)
            
            if n_components > 0:
                stats = cc3d.statistics(labeled_array)
                # stats['voxel_counts'][0] is background
                component_sizes = stats['voxel_counts'][1:]
                
                max_cc_size = component_sizes.max()
                main_ratio = max_cc_size / voxel_total
                significant_cc_count = np.sum(component_sizes > (voxel_total * 0.05))
            else:
                n_components = 0
                significant_cc_count = 0
                main_ratio = 0

            results.append({
                'filename': filename,
                'class_id': class_id,
                'class_name': class_name,
                'cc_count': n_components,
                'significant_cc': significant_cc_count,
                'main_ratio': main_ratio,
                'voxel_count': voxel_total
            })
        else:
            results.append({
                'filename': filename,
                'class_id': class_id,
                'class_name': class_name,
                'cc_count': 0,
                'significant_cc': 0,
                'main_ratio': 0,
                'voxel_count': 0
            })
                
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return []
        
    return results

def calculate_cc_stats():
    output_csv = 'phase2_cc_full_dataset.csv'
    
    all_tasks = []
    
    print("Scanning files...")
    for class_id, class_name in CLASS_NAMES.items():
        # Find directory
        dir_name = f"class{class_id}_{class_name}_results_256_processed"
        # Try both locations (direct or nested)
        path1 = os.path.join(BASE_DIR, dir_name)
        path2 = os.path.join(BASE_DIR, f"class{class_id}_{class_name}", dir_name)
        
        target_dir = None
        if os.path.isdir(path1):
            target_dir = path1
        elif os.path.isdir(path2):
            target_dir = path2
            
        if target_dir:
            files = glob.glob(os.path.join(target_dir, "*.npy"))
            print(f"Class {class_id} ({class_name}): Found {len(files)} files in {target_dir}")
            for f in files:
                all_tasks.append((f, os.path.basename(f), class_id, class_name))
        else:
            print(f"Warning: Directory not found for Class {class_id} ({class_name})")

    print(f"Total files to process: {len(all_tasks)}")
    
    # Initialize CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'class_id', 'class_name', 'cc_count', 'significant_cc', 'main_ratio', 'voxel_count'])
        writer.writeheader()
    
    # Run multiprocessing
    num_workers = min(16, multiprocessing.cpu_count())
    
    all_results = []
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        for file_results in tqdm(pool.imap_unordered(process_single_file, all_tasks), total=len(all_tasks)):
            if file_results:
                all_results.extend(file_results)
                with open(output_csv, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['filename', 'class_id', 'class_name', 'cc_count', 'significant_cc', 'main_ratio', 'voxel_count'])
                    writer.writerows(file_results)

    print(f"\nProcessing complete. Raw results saved to {output_csv}")
    
    # Calculate summary statistics
    df = pd.DataFrame(all_results)
    if df.empty:
        print("No results found!")
        return

    print("\n--- Connected Components Statistics (Phase 2 Processed, 26-connectivity) ---")
    print(f"{'Class ID':<10} {'Name':<15} {'Mean CC':<10} {'Sig CC':<10} {'Main Ratio':<12} {'Max CC':<10} {'Samples':<10}")
    print("-" * 80)
    
    summary_stats = []
    for class_id in CLASS_NAMES.keys():
        class_df = df[df['class_id'] == class_id]
        present_df = class_df[class_df['voxel_count'] > 0]
        
        if not present_df.empty:
            mean_cc = present_df['cc_count'].mean()
            mean_sig_cc = present_df['significant_cc'].mean()
            mean_ratio = present_df['main_ratio'].mean()
            max_cc = present_df['cc_count'].max()
            n_samples = len(present_df)
            
            print(f"{class_id:<10} {CLASS_NAMES[class_id]:<15} {mean_cc:<10.4f} {mean_sig_cc:<10.4f} {mean_ratio:<12.4f} {max_cc:<10} {n_samples:<10}")
            summary_stats.append({
                'Class ID': class_id,
                'Name': CLASS_NAMES[class_id],
                'Mean CC': mean_cc,
                'Mean Significant CC': mean_sig_cc,
                'Mean Main Ratio': mean_ratio,
                'Max CC': max_cc,
                'Samples': n_samples
            })
        else:
            print(f"{class_id:<10} {CLASS_NAMES[class_id]:<15} {'N/A':<10} {'N/A':<10} {0:<10}")
            
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('phase2_cc_summary.csv', index=False)
    print("\nSummary saved to phase2_cc_summary.csv")

if __name__ == "__main__":
    calculate_cc_stats()
