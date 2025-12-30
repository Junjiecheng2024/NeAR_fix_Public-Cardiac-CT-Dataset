#!/usr/bin/env python3
"""
诊断脚本：检查 GT 文件中的实际 class ID 分布
"""
import os
import numpy as np
import nibabel as nib

# 配置路径 (请根据实际情况修改)
GT_ROOT = "/scratch/project_2016517/JunjieCheng/dataset/original/segmentations"
DATA_ROOT = "/scratch/project_2016517/JunjieCheng/dataset"

# 我们定义的 CLASS_NAMES
CLASS_NAMES = {
    1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
    6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
}

def check_gt_labels(case_id="1"):
    """检查单个 GT 文件中的所有 label"""
    candidates = [
        os.path.join(GT_ROOT, f"{case_id}.nii.gz"),
        os.path.join(GT_ROOT, f"{case_id}.nii.img.nii.gz"),
    ]
    gt_path = next((p for p in candidates if os.path.exists(p)), None)
    
    if not gt_path:
        print(f"[ERROR] Cannot find GT for case {case_id}")
        return
    
    print(f"[OK] Found GT: {gt_path}")
    gt_data = nib.load(gt_path).get_fdata()
    print(f"  Shape: {gt_data.shape}")
    
    unique_labels = np.unique(gt_data).astype(int)
    print(f"\n=== GT 中的所有 Label ===")
    for label in unique_labels:
        count = (gt_data == label).sum()
        name = CLASS_NAMES.get(label, "UNKNOWN")
        print(f"  Label {label:2d}: {count:>10,} voxels  ({name})")
    
    print(f"\n=== 对比 CLASS_NAMES ===")
    for cls_id, cls_name in CLASS_NAMES.items():
        gt_vol = (gt_data == cls_id).sum()
        status = "OK" if gt_vol > 0 else "MISSING"
        print(f"  Class {cls_id:2d} ({cls_name:12s}): {gt_vol:>10,} voxels  {status}")

def check_p1_files():
    """检查 P1 输出文件是否存在"""
    print(f"\n=== 检查 P1 文件 ===")
    for cls_id, cls_name in CLASS_NAMES.items():
        cls_lower = cls_name.lower()
        p1_file = os.path.join(DATA_ROOT, f"{cls_lower}_global", "1_mask.npy")
        
        if os.path.exists(p1_file):
            data = np.load(p1_file)
            vol = (data > 0.5).sum()
            print(f"  {cls_name:12s}: Found, Volume = {vol:>10,}")
        else:
            print(f"  {cls_name:12s}: NOT FOUND")

if __name__ == "__main__":
    print("=" * 60)
    print("NeAR Evaluation 诊断脚本")
    print("=" * 60)
    check_gt_labels(case_id="1")
    check_p1_files()
    print("\n诊断完成")
