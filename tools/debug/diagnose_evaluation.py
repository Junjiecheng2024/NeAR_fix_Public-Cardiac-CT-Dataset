#!/usr/bin/env python3
"""
Diagnostic script: inspect the actual class-ID distribution in a GT file.
"""
import os
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(os.environ.get("NEAR_DATA_ROOT", REPO_ROOT / "dataset"))

# Class-name mapping used by this project
CLASS_NAMES = {
    1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
    6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
}

def check_gt_labels(case_id, gt_root):
    """Inspect all labels present in a single GT file."""
    candidates = [
        os.path.join(gt_root, f"{case_id}.nii.gz"),
        os.path.join(gt_root, f"{case_id}.nii.img.nii.gz"),
    ]
    gt_path = next((p for p in candidates if os.path.exists(p)), None)
    
    if not gt_path:
        print(f"[ERROR] Cannot find GT for case {case_id}")
        return
    
    print(f"[OK] Found GT: {gt_path}")
    gt_data = nib.load(gt_path).get_fdata()
    print(f"  Shape: {gt_data.shape}")
    
    unique_labels = np.unique(gt_data).astype(int)
    print(f"\n=== All labels present in GT ===")
    for label in unique_labels:
        count = (gt_data == label).sum()
        name = CLASS_NAMES.get(label, "UNKNOWN")
        print(f"  Label {label:2d}: {count:>10,} voxels  ({name})")
    
    print(f"\n=== Compare against CLASS_NAMES ===")
    for cls_id, cls_name in CLASS_NAMES.items():
        gt_vol = (gt_data == cls_id).sum()
        status = "OK" if gt_vol > 0 else "MISSING"
        print(f"  Class {cls_id:2d} ({cls_name:12s}): {gt_vol:>10,} voxels  {status}")

def check_p1_files(data_root, case_id):
    """Check whether Phase1 output files exist."""
    print(f"\n=== Checking Phase1 files ===")
    for cls_id, cls_name in CLASS_NAMES.items():
        cls_lower = cls_name.lower()
        p1_file = os.path.join(data_root, f"{cls_lower}_global", f"{case_id}_mask.npy")
        
        if os.path.exists(p1_file):
            data = np.load(p1_file)
            vol = (data > 0.5).sum()
            print(f"  {cls_name:12s}: Found, Volume = {vol:>10,}")
        else:
            print(f"  {cls_name:12s}: NOT FOUND")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", default="1", help="Case ID to inspect")
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT), help="Dataset root")
    parser.add_argument("--gt_root", default=None, help="GT root (default: data_root/original/segmentations)")
    args = parser.parse_args()

    gt_root = args.gt_root or os.path.join(args.data_root, "original", "segmentations")

    print("=" * 60)
    print("NeAR Evaluation Diagnostic Script")
    print("=" * 60)
    check_gt_labels(case_id=args.case_id, gt_root=gt_root)
    check_p1_files(data_root=args.data_root, case_id=args.case_id)
    print("\nDiagnostics complete")
