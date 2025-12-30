#!/usr/bin/env python3
"""
直接计算 Case 1 的各类 Dice，对比 evaluation 结果
"""
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

GT_ROOT = "/scratch/project_2016517/JunjieCheng/dataset/original/segmentations"
DATA_ROOT = "/scratch/project_2016517/JunjieCheng/dataset"

CLASS_NAMES = {
    1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
    6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
}

def compute_dice(pred, gt):
    intersection = np.logical_and(pred > 0, gt > 0).sum()
    return 2 * intersection / (pred.sum() + gt.sum() + 1e-8)

def main():
    case_id = "1"
    
    # Load GT
    gt_path = os.path.join(GT_ROOT, f"{case_id}.nii.img.nii.gz")
    gt_data = nib.load(gt_path).get_fdata()
    print(f"GT shape: {gt_data.shape}")
    
    print(f"\n=== Case {case_id} Dice 计算 ===")
    print(f"{'Class':<12} {'P1_Vol':>10} {'GT_Vol':>10} {'Dice':>8}")
    print("-" * 45)
    
    for cls_id, cls_name in CLASS_NAMES.items():
        cls_lower = cls_name.lower()
        
        # Load P1
        p1_path = os.path.join(DATA_ROOT, f"{cls_lower}_global", f"{case_id}_mask.npy")
        if not os.path.exists(p1_path):
            print(f"{cls_name:<12} NOT FOUND")
            continue
        
        p1 = np.load(p1_path)
        p1_bin = (p1 > 0.5).astype(np.uint8)
        
        # Extract GT class and zoom to 256
        gt_cls = (gt_data == cls_id).astype(np.uint8)
        zoom_fac = np.array([256, 256, 256]) / np.array(gt_cls.shape)
        gt_256 = zoom(gt_cls, zoom_fac, order=0)
        gt_256 = (gt_256 > 0.5).astype(np.uint8)
        
        # Dice
        dice = compute_dice(p1_bin, gt_256)
        
        print(f"{cls_name:<12} {p1_bin.sum():>10,} {gt_256.sum():>10,} {dice:>8.4f}")
    
    print("\n如果这些 Dice 值和 evaluation 结果不一致，说明 evaluate_repair_quality.py 有 bug")

if __name__ == "__main__":
    main()
