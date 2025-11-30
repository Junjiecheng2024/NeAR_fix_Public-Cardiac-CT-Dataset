import pandas as pd
import numpy as np

def calculate_ratios():
    csv_path = "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/stage2/stage2_cc_full_dataset.csv"
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    # Constants
    TOTAL_VOXELS = 256 ** 3  # 16,777,216
    
    print(f"{'Class ID':<10} {'Name':<15} {'Mean Voxel Count':<20} {'Ratio (%)':<10}")
    print("-" * 60)
    
    stats_list = []
    
    # Process classes 1-10
    total_foreground_ratio = 0.0
    
    # Get unique classes from CSV, sorted
    class_ids = sorted(df['class_id'].unique())
    
    for class_id in class_ids:
        class_df = df[df['class_id'] == class_id]
        class_name = class_df['class_name'].iloc[0]
        
        # Mean voxel count across all samples (including those with 0 if any, though stage 2 shouldn't have 0 usually)
        # The CSV contains one row per file per class.
        mean_voxels = class_df['voxel_count'].mean()
        ratio = (mean_voxels / TOTAL_VOXELS) * 100
        
        total_foreground_ratio += ratio
        
        print(f"{class_id:<10} {class_name:<15} {mean_voxels:<20.2f} {ratio:<10.2f}%")
        
        stats_list.append({
            'Class ID': class_id,
            'Name': class_name,
            'Ratio': ratio
        })
        
    # Estimate Background
    # Assuming no overlap (which is an approximation for Stage 2), Background = 100 - Sum(Foreground)
    bg_ratio = 100.0 - total_foreground_ratio
    print("-" * 60)
    print(f"{0:<10} {'Background':<15} {'-':<20} {bg_ratio:<10.2f}% (Estimated)")

if __name__ == "__main__":
    calculate_ratios()
