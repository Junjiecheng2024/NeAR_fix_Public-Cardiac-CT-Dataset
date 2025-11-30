import os
import numpy as np
import nibabel as nib
import argparse
from phase3 import load_masks, resolve_priority_overlap, enforce_anatomical_constraints, CLASS_NAMES

def save_nii(mask, path):
    affine = np.eye(4)
    nii = nib.Nifti1Image(mask, affine)
    nib.save(nii, path)
    print(f"Saved {path}")

def run_debug_case(case_id, phase2_dir, output_dir):
    print(f"--- Debugging Case {case_id} ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Phase 2 Inputs
    print("Loading Phase 2 masks...")
    masks = load_masks(case_id, phase2_dir)
    
    # Save combined Phase 2 (naive max? or just separate?)
    # Let's save a naive "max" just to see overlaps
    naive = np.zeros((256, 256, 256), dtype=np.uint8)
    for c, m in masks.items():
        if m is not None:
            # Just overwrite to see raw geometry
            naive[m > 0] = c
    save_nii(naive, os.path.join(output_dir, f"{case_id}_phase2_raw_overlay.nii.gz"))
    
    # 2. Priority Fusion
    print("Applying Priority Fusion...")
    fused = resolve_priority_overlap(masks)
    save_nii(fused, os.path.join(output_dir, f"{case_id}_step1_priority_fused.nii.gz"))
    
    # Check difference
    diff_p = (fused != naive).astype(np.uint8)
    if diff_p.sum() > 0:
        print(f"Priority Fusion changed {diff_p.sum()} voxels (Overlap resolution).")
    
    # 3. Anatomy Rules
    print("Applying Anatomy Rules...")
    final = enforce_anatomical_constraints(fused.copy())
    save_nii(final, os.path.join(output_dir, f"{case_id}_step2_anatomy_fixed.nii.gz"))
    
    # Check difference
    diff_a = (final != fused).astype(np.uint8)
    if diff_a.sum() > 0:
        print(f"Anatomy Rules changed {diff_a.sum()} voxels.")
        # Save difference mask
        save_nii(diff_a, os.path.join(output_dir, f"{case_id}_diff_anatomy.nii.gz"))
    else:
        print("Anatomy Rules triggered no changes.")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--phase2_dir", default="/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/phase2")
    parser.add_argument("--output_dir", default="debug_output")
    args = parser.parse_args()
    
    run_debug_case(args.case_id, args.phase2_dir, args.output_dir)
