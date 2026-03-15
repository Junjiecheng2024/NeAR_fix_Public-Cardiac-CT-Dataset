#!/usr/bin/env python3
"""
Directly compute per-class Dice scores for one case and compare them with the evaluation output.
"""
import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(os.environ.get("NEAR_DATA_ROOT", REPO_ROOT / "dataset"))

CLASS_NAMES = {
    1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
    6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
}

def compute_dice(pred, gt):
    intersection = np.logical_and(pred > 0, gt > 0).sum()
    return 2 * intersection / (pred.sum() + gt.sum() + 1e-8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", default="214", help="Case ID to inspect")
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT), help="Dataset root")
    parser.add_argument("--gt_root", default=None, help="GT root (default: data_root/original/segmentations)")
    args = parser.parse_args()

    case_id = args.case_id
    data_root = args.data_root
    gt_root = args.gt_root or os.path.join(data_root, "original", "segmentations")
    
    # Load GT
    gt_candidates = [
        os.path.join(gt_root, f"{case_id}.nii.gz"),
        os.path.join(gt_root, f"{case_id}.nii.img.nii.gz"),
    ]
    gt_path = next((p for p in gt_candidates if os.path.exists(p)), None)
    if gt_path is None:
        raise FileNotFoundError(f"Could not find GT for case {case_id} under {gt_root}")
    gt_data = nib.load(gt_path).get_fdata()
    print(f"GT shape: {gt_data.shape}")
    
    print(f"\n=== Dice computation for case {case_id} ===")
    print(f"{'Class':<12} {'P1_Vol':>10} {'GT_Vol':>10} {'Dice':>8}")
    print("-" * 45)
    
    for cls_id, cls_name in CLASS_NAMES.items():
        cls_lower = cls_name.lower()
        
        # Load P1
        p1_path = os.path.join(data_root, f"{cls_lower}_global", f"{case_id}_mask.npy")
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
    
    print("\nIf these Dice values do not match the evaluation output, evaluate_repair_quality.py likely has a bug.")

if __name__ == "__main__":
    main()
