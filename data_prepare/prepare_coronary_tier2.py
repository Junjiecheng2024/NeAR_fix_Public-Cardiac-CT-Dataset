#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_coronary_tier2.py
-------------------------
Prepare Coronary-specific cropped data for NeAR v2.0 Tier2 training.

Key Features:
1. Computes class-aware bounding box: Coronary + Myocardium + Aorta
2. Saves cropped CT (appearance) and masks (shape)
3. Stores crop parameters for later coordinate mapping back to global space
4. Optionally upsamples to higher resolution for finer boundary learning

Usage:
    python prepare_coronary_tier2.py \
        --images_dir /scratch/project_2016517/junjie/dataset/original/images \
        --labels_dir /scratch/project_2016517/junjie/dataset/original/segmentations \
        --output_dir /scratch/project_2016517/junjie/dataset/coronary_tier2 \
        --margin 20 \
        --target_resolution 256 \
        --n_workers 8
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import nibabel as nib
except ImportError:
    raise ImportError("Please install nibabel: pip install nibabel")

from scipy.ndimage import binary_dilation
from skimage.transform import resize as sk_resize


# Class definitions
CLASS_IDS = {
    "Background": 0,
    "Myocardium": 1,
    "LA": 2,
    "LV": 3,
    "RA": 4,
    "RV": 5,
    "Aorta": 6,
    "PA": 7,
    "LAA": 8,
    "Coronary": 9,
    "PV": 10
}

# Coronary context: include Myocardium and Aorta
CORONARY_TARGET_CLASS = 9
CORONARY_CONTEXT_CLASSES = [1, 6]  # Myocardium, Aorta

# CT normalization parameters (cardiac CT window)
HU_MIN = -100
HU_MAX = 700


def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, object]:
    """Load NIfTI file, return data, affine, header."""
    img = nib.load(str(path))
    data = img.get_fdata()
    affine = img.affine.copy()
    header = img.header
    return data, affine, header


def compute_class_bbox(mask: np.ndarray, class_ids: List[int], margin: int = 20) -> Optional[Dict]:
    """
    Compute bounding box for specified classes.
    
    Returns:
        Dict with 'origin' (z,y,x), 'size' (d,h,w), 'valid' flag
        or None if no foreground.
    """
    # Create combined foreground mask
    fg = np.zeros(mask.shape, dtype=bool)
    for cid in class_ids:
        fg |= (mask == cid)
    
    if fg.sum() < 10:  # minimum voxels threshold
        return None
    
    # Get bounding box coordinates
    coords = np.argwhere(fg)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    
    # Apply margin
    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, np.array(mask.shape))
    
    origin = mins.tolist()
    size = (maxs - mins).tolist()
    
    return {
        "origin": origin,  # [z, y, x]
        "size": size,      # [d, h, w]
        "mins": mins,
        "maxs": maxs
    }


def crop_volume(vol: np.ndarray, bbox: Dict) -> np.ndarray:
    """Crop 3D volume using bbox dict."""
    mins = bbox["mins"]
    maxs = bbox["maxs"]
    return vol[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]].copy()


def resize_volume(vol: np.ndarray, target_size: Tuple[int, int, int], is_label: bool = False) -> np.ndarray:
    """Resize 3D volume to target size."""
    if is_label:
        # Use nearest neighbor for labels
        resized = sk_resize(vol, target_size, order=0, 
                           preserve_range=True, anti_aliasing=False)
        return resized.astype(vol.dtype)
    else:
        # Use cubic interpolation for CT
        resized = sk_resize(vol, target_size, order=3, 
                           preserve_range=True)
        return resized.astype(np.float32)


def normalize_ct(ct: np.ndarray, hu_min: int = HU_MIN, hu_max: int = HU_MAX) -> np.ndarray:
    """Normalize CT to [0, 1] range with HU clipping."""
    ct = np.clip(ct, hu_min, hu_max).astype(np.float32)
    ct = (ct - hu_min) / (hu_max - hu_min + 1e-6)
    return ct


def compute_voxel_ratio(mask: np.ndarray, class_id: int) -> float:
    """Compute voxel ratio for a specific class."""
    total = mask.size
    class_voxels = np.sum(mask == class_id)
    return class_voxels / total if total > 0 else 0.0


def process_single_case(args) -> Dict:
    """Process a single case for multiprocessing."""
    case_id, config = args
    
    try:
        # Construct file paths
        img_path = Path(config["images_dir"]) / f"{case_id}.nii.img.nii.gz"
        seg_path = Path(config["labels_dir"]) / f"{case_id}.nii.img.nii.gz"
        
        # Check file existence
        if not img_path.exists():
            return {"case_id": case_id, "status": "missing_img", "error": str(img_path)}
        if not seg_path.exists():
            return {"case_id": case_id, "status": "missing_seg", "error": str(seg_path)}
        
        # Load data
        ct_data, ct_affine, ct_header = load_nifti(img_path)
        seg_data, seg_affine, _ = load_nifti(seg_path)
        seg_data = np.rint(seg_data).astype(np.int16)
        
        # Check if coronary exists
        coronary_voxels = np.sum(seg_data == CORONARY_TARGET_CLASS)
        if coronary_voxels < 100:
            return {"case_id": case_id, "status": "no_coronary", 
                    "error": f"Only {coronary_voxels} coronary voxels"}
        
        # Compute combined bbox (Coronary + Myocardium + Aorta)
        all_classes = [CORONARY_TARGET_CLASS] + CORONARY_CONTEXT_CLASSES
        bbox = compute_class_bbox(seg_data, all_classes, margin=config["margin"])
        
        if bbox is None:
            return {"case_id": case_id, "status": "empty_bbox", "error": "No foreground found"}
        
        # Crop volumes
        ct_cropped = crop_volume(ct_data, bbox)
        seg_cropped = crop_volume(seg_data, bbox)
        
        # Compute voxel ratio BEFORE crop (for comparison)
        ratio_before = compute_voxel_ratio(seg_data, CORONARY_TARGET_CLASS)
        
        # Compute voxel ratio AFTER crop
        ratio_after = compute_voxel_ratio(seg_cropped, CORONARY_TARGET_CLASS)
        
        # Store original crop size before any resizing
        original_crop_size = list(ct_cropped.shape)
        
        # Optional: resize to target resolution
        resize_applied = False
        if config["target_resolution"] is not None:
            target_size = (config["target_resolution"],) * 3
            ct_cropped = resize_volume(ct_cropped, target_size, is_label=False)
            seg_cropped = resize_volume(seg_cropped, target_size, is_label=True)
            resize_applied = True
        
        # Normalize CT
        ct_normalized = normalize_ct(ct_cropped)
        
        # Extract individual masks
        mask_coronary = (seg_cropped == CORONARY_TARGET_CLASS).astype(np.uint8)
        mask_myo = (seg_cropped == CLASS_IDS["Myocardium"]).astype(np.uint8)
        mask_aorta = (seg_cropped == CLASS_IDS["Aorta"]).astype(np.uint8)
        mask_context = ((mask_myo > 0) | (mask_aorta > 0)).astype(np.uint8)
        
        # Create output directory for this case
        case_out_dir = Path(config["output_dir"]) / case_id
        case_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        np.save(case_out_dir / "ct.npy", ct_normalized)
        np.save(case_out_dir / "mask_coronary.npy", mask_coronary)
        np.save(case_out_dir / "mask_context.npy", mask_context)
        np.save(case_out_dir / "seg_full.npy", seg_cropped.astype(np.int16))
        
        # Save crop parameters for coordinate mapping
        crop_params = {
            "case_id": case_id,
            "origin": bbox["origin"],  # [z, y, x] in original space
            "size": original_crop_size,  # [d, h, w] before resize
            "original_shape": list(seg_data.shape),  # Full volume shape
            "cropped_shape": list(ct_cropped.shape),  # After resize
            "resize_applied": resize_applied,
            "target_resolution": config["target_resolution"],
            "margin": config["margin"],
            "voxel_ratio_before": float(ratio_before),
            "voxel_ratio_after": float(ratio_after),
            "ratio_improvement": float(ratio_after / ratio_before) if ratio_before > 0 else 0
        }
        
        with open(case_out_dir / "crop_params.json", "w") as f:
            json.dump(crop_params, f, indent=2)
        
        return {
            "case_id": case_id,
            "status": "success",
            "crop_size": original_crop_size,
            "final_size": list(ct_cropped.shape),
            "voxel_ratio_before": ratio_before,
            "voxel_ratio_after": ratio_after,
            "improvement": ratio_after / ratio_before if ratio_before > 0 else 0
        }
        
    except Exception as e:
        import traceback
        return {"case_id": case_id, "status": "error", "error": str(e), 
                "traceback": traceback.format_exc()}


def get_case_ids(labels_dir: str) -> List[str]:
    """Extract case IDs from labels directory."""
    labels_path = Path(labels_dir)
    case_ids = []
    
    # Try different patterns
    patterns = ["*.nii.gz", "*.nii.img.nii.gz"]
    
    for pattern in patterns:
        files = list(labels_path.glob(pattern))
        if files:
            for f in files:
                # Remove all nifti extensions
                name = f.name
                for ext in [".nii.img.nii.gz", ".nii.gz", ".nii"]:
                    if name.endswith(ext):
                        name = name[:-len(ext)]
                        break
                case_ids.append(name)
            break
    
    return sorted(set(case_ids))


def main():
    parser = argparse.ArgumentParser(description="Prepare Coronary Tier2 data for NeAR v2.0")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory of original CT images (.nii.gz)")
    parser.add_argument("--labels_dir", type=str, required=True,
                        help="Directory of segmentation labels (.nii.gz)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed data")
    parser.add_argument("--margin", type=int, default=20,
                        help="Margin (voxels) to expand bbox on each side (default: 20)")
    parser.add_argument("--target_resolution", type=int, default=None,
                        help="Target resolution for resizing (e.g., 256). None = keep original size")
    parser.add_argument("--n_workers", type=int, default=8,
                        help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get case IDs
    case_ids = get_case_ids(args.labels_dir)
    print(f"\n{'='*70}")
    print("NeAR v2.0 - Coronary Tier2 Data Preparation")
    print(f"{'='*70}")
    print(f"Found {len(case_ids)} cases")
    print(f"Images dir: {args.images_dir}")
    print(f"Labels dir: {args.labels_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Margin: {args.margin} voxels")
    print(f"Target resolution: {args.target_resolution or 'Keep original'}")
    print(f"Workers: {args.n_workers}")
    print(f"{'='*70}\n")
    
    if len(case_ids) == 0:
        print("[ERROR] No cases found! Check your labels_dir path.")
        sys.exit(1)
    
    # Build config
    config = {
        "images_dir": args.images_dir,
        "labels_dir": args.labels_dir,
        "output_dir": args.output_dir,
        "margin": args.margin,
        "target_resolution": args.target_resolution,
    }
    
    # Process cases
    process_args = [(cid, config) for cid in case_ids]
    
    results = []
    success_count = 0
    fail_count = 0
    
    if args.n_workers > 1:
        with Pool(args.n_workers) as pool:
            for result in tqdm(pool.imap(process_single_case, process_args), 
                             total=len(case_ids), desc="Processing"):
                results.append(result)
                if result["status"] == "success":
                    success_count += 1
                else:
                    fail_count += 1
                    print(f"\n[WARN] {result['case_id']}: {result['status']} - {result.get('error', '')}")
    else:
        for args_item in tqdm(process_args, desc="Processing"):
            result = process_single_case(args_item)
            results.append(result)
            if result["status"] == "success":
                success_count += 1
            else:
                fail_count += 1
                print(f"\n[WARN] {result['case_id']}: {result['status']} - {result.get('error', '')}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Success: {success_count}/{len(case_ids)}")
    print(f"Failed:  {fail_count}/{len(case_ids)}")
    
    # Compute average improvement
    successful = [r for r in results if r["status"] == "success"]
    if successful:
        avg_ratio_before = np.mean([r["voxel_ratio_before"] for r in successful])
        avg_ratio_after = np.mean([r["voxel_ratio_after"] for r in successful])
        avg_improvement = np.mean([r["improvement"] for r in successful])
        
        print(f"\nVoxel Ratio Statistics (Coronary):")
        print(f"  Before crop: {avg_ratio_before*100:.4f}%")
        print(f"  After crop:  {avg_ratio_after*100:.4f}%")
        print(f"  Improvement: {avg_improvement:.1f}x")
    
    # Save summary
    summary_path = output_dir / "processing_summary.json"
    summary = {
        "total_cases": len(case_ids),
        "success": success_count,
        "failed": fail_count,
        "config": config,
        "results": results
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nSummary saved to: {summary_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
