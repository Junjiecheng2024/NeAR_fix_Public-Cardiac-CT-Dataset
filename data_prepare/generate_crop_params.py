#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_crop_params.py
-----------------------
Generate crop_params.json for all cases without creating full training data.
This is useful when training data was deleted but crop parameters are needed
for coordinate mapping during inference.

Usage:
    python generate_crop_params.py --class_name Coronary
    python generate_crop_params.py --all
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

try:
    import nibabel as nib
except ImportError:
    raise ImportError("Please install nibabel: pip install nibabel")


# Class definitions
CLASS_IDS = {
    "Myocardium": 1, "LA": 2, "LV": 3, "RA": 4, "RV": 5,
    "Aorta": 6, "PA": 7, "LAA": 8, "Coronary": 9, "PV": 10
}

# Margin settings (same as prepare_all_classes_tier2.py)
MARGIN_SETTINGS = {
    "Myocardium": 10, "LA": 10, "LV": 10, "RA": 10, "RV": 10,
    "Aorta": 10, "PA": 10, "LAA": 5, "Coronary": 5, "PV": 5,
}


def load_nifti(path: str) -> np.ndarray:
    """Load NIfTI file."""
    img = nib.load(str(path))
    return img.get_fdata()


def compute_class_bbox(mask: np.ndarray, class_id: int, margin: int = 10):
    """Compute bounding box for a specific class."""
    fg = (mask == class_id)
    
    if fg.sum() < 10:
        return None
    
    coords = np.argwhere(fg)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    
    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, np.array(mask.shape))
    
    return {
        "origin": mins.tolist(),
        "size": (maxs - mins).tolist()
    }


def process_single_case(args):
    """Process a single case to generate crop_params.json."""
    case_id, config = args
    target_class = config["target_class"]
    target_id = CLASS_IDS[target_class]
    margin = MARGIN_SETTINGS[target_class]
    
    try:
        # Load segmentation
        seg_path = Path(config["labels_dir"]) / f"{case_id}.nii.img.nii.gz"
        if not seg_path.exists():
            return {"case_id": case_id, "status": "missing_seg"}
        
        seg_data = load_nifti(seg_path)
        seg_data = np.rint(seg_data).astype(np.int16)
        
        # Check if target class exists
        target_voxels = np.sum(seg_data == target_id)
        if target_voxels < 100:
            return {"case_id": case_id, "status": "no_target"}
        
        # Compute bbox
        bbox = compute_class_bbox(seg_data, target_id, margin=margin)
        if bbox is None:
            return {"case_id": case_id, "status": "empty_bbox"}
        
        # Create output directory
        output_dir = Path(config["output_dir"]) / case_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save crop parameters
        crop_params = {
            "case_id": case_id,
            "target_class": target_class,
            "target_id": target_id,
            "origin": bbox["origin"],
            "size": bbox["size"],
            "original_shape": list(seg_data.shape),
            "resize_applied": True,  # Assume resize was applied during training
            "target_resolution": config.get("target_resolution", 128),
            "margin": margin,
        }
        
        with open(output_dir / "crop_params.json", "w") as f:
            json.dump(crop_params, f, indent=2)
        
        return {"case_id": case_id, "status": "success"}
        
    except Exception as e:
        return {"case_id": case_id, "status": "error", "error": str(e)}


def get_case_ids(labels_dir: str):
    """Get all case IDs from labels directory."""
    labels_path = Path(labels_dir)
    case_ids = []
    
    for f in labels_path.glob("*.nii.img.nii.gz"):
        name = f.name.replace(".nii.img.nii.gz", "")
        case_ids.append(name)
    
    return sorted(set(case_ids))


def main():
    parser = argparse.ArgumentParser(description="Generate crop_params.json only")
    parser.add_argument("--labels_dir", type=str,
                        default="/scratch/project_2016517/JunjieCheng/dataset/original/segmentations",
                        help="Directory of segmentation labels")
    parser.add_argument("--output_base", type=str,
                        default="/scratch/project_2016517/JunjieCheng/dataset",
                        help="Base output directory")
    parser.add_argument("--class_name", type=str, default=None,
                        help="Specific class to process (e.g., 'Coronary')")
    parser.add_argument("--all", action="store_true",
                        help="Process all 10 classes")
    parser.add_argument("--target_resolution", type=int, default=128,
                        help="Target resolution used during training")
    parser.add_argument("--n_workers", type=int, default=16,
                        help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Determine classes to process
    if args.all:
        classes = list(CLASS_IDS.keys())
    elif args.class_name:
        if args.class_name not in CLASS_IDS:
            print(f"Error: Unknown class '{args.class_name}'")
            print(f"Available: {list(CLASS_IDS.keys())}")
            sys.exit(1)
        classes = [args.class_name]
    else:
        print("Error: Specify --all or --class_name")
        sys.exit(1)
    
    # Get case IDs
    case_ids = get_case_ids(args.labels_dir)
    print(f"Found {len(case_ids)} cases")
    
    for class_name in classes:
        print(f"\n{'='*60}")
        print(f"Generating crop_params for: {class_name}")
        print(f"{'='*60}")
        
        output_dir = Path(args.output_base) / f"{class_name.lower()}_tier2"
        
        config = {
            "labels_dir": args.labels_dir,
            "output_dir": str(output_dir),
            "target_class": class_name,
            "target_resolution": args.target_resolution,
        }
        
        process_args = [(cid, config) for cid in case_ids]
        
        success = 0
        failed = 0
        
        with Pool(args.n_workers) as pool:
            for result in tqdm(pool.imap(process_single_case, process_args),
                             total=len(case_ids), desc=class_name):
                if result["status"] == "success":
                    success += 1
                else:
                    failed += 1
        
        print(f"{class_name}: {success} success, {failed} failed")
    
    print(f"\n{'='*60}")
    print("Done! crop_params.json files generated.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
