import os
import pandas as pd
import glob
import numpy as np

def verify_stage2():
    base_dir = "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/stage2"
    
    classes = {
        1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
        6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
    }
    
    summary_list = []
    
    print(f"{'Class':<15} | {'Samples':<8} | {'Mean Orig CC':<12} | {'Mean Final CC':<12} | {'Max Final CC':<12} | {'Zero Count':<10}")
    print("-" * 85)
    
    for class_id in range(1, 11):
        class_name = classes[class_id]
        
        # Find the processed directory
        # Try both naming conventions
        dir_name_1 = f"class{class_id}_{class_name}_results_256_processed"
        dir_name_2 = f"class{class_id}_{class_name}/class{class_id}_{class_name}_results_256_processed"
        
        path1 = os.path.join(base_dir, dir_name_1)
        path2 = os.path.join(base_dir, dir_name_2)
        
        stats_file = None
        if os.path.exists(os.path.join(path1, "morphology_stats.csv")):
            stats_file = os.path.join(path1, "morphology_stats.csv")
        elif os.path.exists(os.path.join(path2, "morphology_stats.csv")):
            stats_file = os.path.join(path2, "morphology_stats.csv")
            
        if stats_file:
            df = pd.read_csv(stats_file)
            n_samples = len(df)
            mean_orig = df['original_cc'].mean()
            mean_final = df['final_cc'].mean()
            max_final = df['final_cc'].max()
            zero_count = (df['final_cc'] == 0).sum()
            
            print(f"{class_name:<15} | {n_samples:<8} | {mean_orig:<12.2f} | {mean_final:<12.2f} | {max_final:<12} | {zero_count:<10}")
            
            summary_list.append({
                'Class': class_name,
                'Samples': n_samples,
                'Mean Orig CC': mean_orig,
                'Mean Final CC': mean_final,
                'Max Final CC': max_final,
                'Zero Count': zero_count
            })
        else:
            print(f"{class_name:<15} | {'MISSING':<8} | {'-':<12} | {'-':<12} | {'-':<12} | {'-':<10}")

    print("-" * 85)
    
    # Save summary
    if summary_list:
        pd.DataFrame(summary_list).to_csv(os.path.join(base_dir, "stage2_verification_summary.csv"), index=False)
        print(f"\nSummary saved to {os.path.join(base_dir, 'stage2_verification_summary.csv')}")

if __name__ == "__main__":
    verify_stage2()
