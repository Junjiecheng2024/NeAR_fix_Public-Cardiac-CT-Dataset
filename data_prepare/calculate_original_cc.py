import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure
from tqdm import tqdm
import pandas as pd
import multiprocessing
import csv
import time

# Class mapping
CLASS_NAMES = {
    1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
    6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
}

def process_single_file(args):
    """Process a single NIfTI file and return CC counts for all classes."""
    path, filename = args
    results = []
    
    # Define 26-connectivity structure (Face + Edge + Corner)
    structure = generate_binary_structure(3, 3)
    
    try:
        img = nib.load(path)
        # Load data into memory
        data = np.asanyarray(img.dataobj)
        data = np.rint(data).astype(np.uint8)
        
        for class_id, class_name in CLASS_NAMES.items():
            mask = (data == class_id).astype(np.uint8)
            voxel_total = mask.sum()
            
            if voxel_total > 0:
                labeled_array, n_components = label(mask, structure=structure)
                
                # Calculate component sizes
                _, counts = np.unique(labeled_array, return_counts=True)
                # counts[0] is background, remove it
                component_sizes = counts[1:]
                
                max_cc_size = component_sizes.max()
                # Main component ratio
                main_ratio = max_cc_size / voxel_total
                
                # Significant fragments count (volume > 5% of total volume)
                significant_cc_count = np.sum(component_sizes > (voxel_total * 0.05))

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
    data_dir = '/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset_backup/dataset/original/segmentations'
    output_csv = 'original_cc_full_dataset.csv'
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
    files.sort()
    # files = files[:100] # Debug: test on 100 files first? No, user wants full.
    
    print(f"Found {len(files)} files. Starting processing on {multiprocessing.cpu_count()} cores...")
    
    # Prepare arguments
    tasks = [(os.path.join(data_dir, f), f) for f in files]
    
    # Initialize CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'class_id', 'class_name', 'cc_count', 'significant_cc', 'main_ratio', 'voxel_count'])
        writer.writeheader()
    
    # Run multiprocessing
    # Use a safe number of workers to avoid OOM. 8 is usually safe for 64GB RAM.
    num_workers = min(8, multiprocessing.cpu_count())
    
    all_results = []
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap_unordered for better responsiveness
        for file_results in tqdm(pool.imap_unordered(process_single_file, tasks), total=len(tasks)):
            if file_results:
                all_results.extend(file_results)
                # Append to CSV incrementally
                with open(output_csv, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['filename', 'class_id', 'class_name', 'cc_count', 'significant_cc', 'main_ratio', 'voxel_count'])
                    writer.writerows(file_results)

    print(f"\nProcessing complete. Raw results saved to {output_csv}")
    
    # Calculate summary statistics
    df = pd.DataFrame(all_results)
    if df.empty:
        print("No results found!")
        return

    print("\n--- Connected Components Statistics (Full Dataset, 26-connectivity) ---")
    print(f"{'Class ID':<10} {'Name':<15} {'Mean CC':<10} {'Sig CC':<10} {'Main Ratio':<12} {'Max CC':<10} {'Samples':<10}")
    print("-" * 80)
    
    summary_stats = []
    for class_id in CLASS_NAMES.keys():
        class_df = df[df['class_id'] == class_id]
        # Only consider samples where the class is present (voxel_count > 0)
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
            
    # Save summary
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('original_cc_summary.csv', index=False)
    print("\nSummary saved to original_cc_summary.csv")

if __name__ == "__main__":
    calculate_cc_stats()
