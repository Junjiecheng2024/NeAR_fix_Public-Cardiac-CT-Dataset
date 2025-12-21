#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_all_classes_tier2.py
----------------------------
Prepare cropped data for ALL cardiac classes for NeAR v2.0 Tier2 training.

Supports 10 classes:
- Myocardium (1), LA (2), LV (3), RA (4), RV (5)
- Aorta (6), PA (7), LAA (8), Coronary (9), PV (10)

Each class has its own context definition for anatomical guidance.

Usage:
    # Process all classes
    python prepare_all_classes_tier2.py --all

    # Process specific class
    python prepare_all_classes_tier2.py --class_name Aorta

    # Dry run (show what would be processed)
    python prepare_all_classes_tier2.py --all --dry_run
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import nibabel as nib
except ImportError:
    raise ImportError("Please install nibabel: pip install nibabel")

from scipy.ndimage import binary_dilation
from skimage.transform import resize as sk_resize


# ==============================================================================
# Class Definitions and Context Relationships
# ==============================================================================

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

# Reverse mapping
ID_TO_CLASS = {v: k for k, v in CLASS_IDS.items()}

# Context classes for each target (anatomically related structures)
# These provide spatial guidance during training
CONTEXT_DEFINITIONS = {
    "Myocardium": [2, 3, 4, 5],  # All chambers (LA, LV, RA, RV)
    "LA": [1, 8, 10],            # Myocardium, LAA, PV
    "LV": [1, 6],                # Myocardium, Aorta
    "RA": [1, 5, 7],             # Myocardium, RV, PA
    "RV": [1, 4, 7],             # Myocardium, RA, PA
    "Aorta": [1, 3, 9],          # Myocardium, LV, Coronary
    "PA": [4, 5],                # RA, RV
    "LAA": [2],                  # LA
    "Coronary": [1, 6],          # Myocardium, Aorta
    "PV": [2],                   # LA
}

# Margin settings (voxels) - larger for smaller structures
MARGIN_SETTINGS = {
    "Myocardium": 10,
    "LA": 10,
    "LV": 10,
    "RA": 10,
    "RV": 10,
    "Aorta": 10,
    "PA": 10,
    "LAA": 10,  # Small structure
    "Coronary": 10,  # Small structure
    "PV": 10,  # Small structure
}

# CT normalization parameters (cardiac CT window)
HU_MIN = -100
HU_MAX = 700


# ==============================================================================
# Helper Functions
# ==============================================================================

def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, object]:
    """Load NIfTI file, return data, affine, header."""
    img = nib.load(str(path))
    data = img.get_fdata()
    affine = img.affine.copy()
    header = img.header
    return data, affine, header


def compute_class_bbox(mask: np.ndarray, class_ids: List[int], margin: int = 20) -> Optional[Dict]:
    """Compute bounding box for specified classes."""
    fg = np.zeros(mask.shape, dtype=bool)
    for cid in class_ids:
        fg |= (mask == cid)
    
    if fg.sum() < 10:
        return None
    
    coords = np.argwhere(fg)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    
    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, np.array(mask.shape))
    
    origin = mins.tolist()
    size = (maxs - mins).tolist()
    
    return {
        "origin": origin,
        "size": size,
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
        resized = sk_resize(vol, target_size, order=0, 
                           preserve_range=True, anti_aliasing=False)
        return resized.astype(vol.dtype)
    else:
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


# ==============================================================================
# Processing Functions
# ==============================================================================

def process_single_case(args) -> Dict:
    """Process a single case for a specific class."""
    case_id, config = args
    target_class = config["target_class"]
    target_id = CLASS_IDS[target_class]
    context_ids = CONTEXT_DEFINITIONS[target_class]
    margin = MARGIN_SETTINGS[target_class]
    
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
        
        # Check if target class exists
        target_voxels = np.sum(seg_data == target_id)
        if target_voxels < 100:
            return {"case_id": case_id, "status": "no_target", 
                    "error": f"Only {target_voxels} {target_class} voxels"}
        
        # Compute bbox based on TARGET CLASS ONLY
        bbox = compute_class_bbox(seg_data, [target_id], margin=margin)
        
        if bbox is None:
            return {"case_id": case_id, "status": "empty_bbox", "error": "No foreground found"}
        
        # Crop volumes
        ct_cropped = crop_volume(ct_data, bbox)
        seg_cropped = crop_volume(seg_data, bbox)
        
        # Compute voxel ratios
        ratio_before = compute_voxel_ratio(seg_data, target_id)
        ratio_after = compute_voxel_ratio(seg_cropped, target_id)
        
        # Store original crop size
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
        
        # Extract target mask
        mask_target = (seg_cropped == target_id).astype(np.uint8)
        
        # Extract context mask (union of all context classes)
        mask_context = np.zeros_like(seg_cropped, dtype=np.uint8)
        for ctx_id in context_ids:
            mask_context |= (seg_cropped == ctx_id).astype(np.uint8)
        
        # Create output directory
        case_out_dir = Path(config["output_dir"]) / case_id
        case_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data (use generic names for compatibility)
        np.save(case_out_dir / "ct.npy", ct_normalized)
        np.save(case_out_dir / f"mask_{target_class.lower()}.npy", mask_target)
        np.save(case_out_dir / "mask_target.npy", mask_target)  # Alias for training script
        np.save(case_out_dir / "mask_context.npy", mask_context)
        np.save(case_out_dir / "seg_full.npy", seg_cropped.astype(np.int16))
        
        # Save crop parameters
        crop_params = {
            "case_id": case_id,
            "target_class": target_class,
            "target_id": target_id,
            "context_classes": [ID_TO_CLASS[i] for i in context_ids],
            "context_ids": context_ids,
            "origin": bbox["origin"],
            "size": original_crop_size,
            "original_shape": list(seg_data.shape),
            "cropped_shape": list(ct_cropped.shape),
            "resize_applied": resize_applied,
            "target_resolution": config["target_resolution"],
            "margin": margin,
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
    
    patterns = ["*.nii.gz", "*.nii.img.nii.gz"]
    
    for pattern in patterns:
        files = list(labels_path.glob(pattern))
        if files:
            for f in files:
                name = f.name
                for ext in [".nii.img.nii.gz", ".nii.gz", ".nii"]:
                    if name.endswith(ext):
                        name = name[:-len(ext)]
                        break
                case_ids.append(name)
            break
    
    return sorted(set(case_ids))


def process_class(class_name: str, config: Dict, case_ids: List[str], n_workers: int = 8) -> Dict:
    """Process all cases for a single class."""
    
    class_output_dir = Path(config["base_output_dir"]) / f"{class_name.lower()}_tier2"
    class_output_dir.mkdir(parents=True, exist_ok=True)
    
    class_config = {
        **config,
        "output_dir": str(class_output_dir),
        "target_class": class_name,
    }
    
    print(f"\n{'='*70}")
    print(f"Processing: {class_name}")
    print(f"{'='*70}")
    print(f"Output: {class_output_dir}")
    print(f"Context: {CONTEXT_DEFINITIONS[class_name]}")
    print(f"Margin: {MARGIN_SETTINGS[class_name]} voxels")
    
    process_args = [(cid, class_config) for cid in case_ids]
    
    results = []
    success_count = 0
    fail_count = 0
    
    if n_workers > 1:
        with Pool(n_workers) as pool:
            for result in tqdm(pool.imap(process_single_case, process_args), 
                             total=len(case_ids), desc=f"{class_name}"):
                results.append(result)
                if result["status"] == "success":
                    success_count += 1
                else:
                    fail_count += 1
    else:
        for args_item in tqdm(process_args, desc=f"{class_name}"):
            result = process_single_case(args_item)
            results.append(result)
            if result["status"] == "success":
                success_count += 1
            else:
                fail_count += 1
    
    # Summary
    print(f"\n{class_name} Results: {success_count}/{len(case_ids)} success, {fail_count} failed")
    
    # Compute statistics
    successful = [r for r in results if r["status"] == "success"]
    stats = {}
    if successful:
        stats = {
            "avg_ratio_before": np.mean([r["voxel_ratio_before"] for r in successful]),
            "avg_ratio_after": np.mean([r["voxel_ratio_after"] for r in successful]),
            "avg_improvement": np.mean([r["improvement"] for r in successful]),
        }
        print(f"  Voxel ratio: {stats['avg_ratio_before']*100:.4f}% -> {stats['avg_ratio_after']*100:.4f}%")
        print(f"  Improvement: {stats['avg_improvement']:.1f}x")
    
    # Save summary
    summary = {
        "class_name": class_name,
        "total_cases": len(case_ids),
        "success": success_count,
        "failed": fail_count,
        "config": class_config,
        "statistics": stats,
        "results": results
    }
    
    with open(class_output_dir / "processing_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare Tier2 data for ALL cardiac classes")
    parser.add_argument("--images_dir", type=str, 
                       default="/scratch/project_2016517/JunjieCheng/dataset/original/images",
                       help="Directory of original CT images")
    parser.add_argument("--labels_dir", type=str,
                       default="/scratch/project_2016517/JunjieCheng/dataset/original/segmentations",
                       help="Directory of segmentation labels")
    parser.add_argument("--output_dir", type=str,
                       default="/scratch/project_2016517/JunjieCheng/dataset",
                       help="Base output directory (class subdirs will be created)")
    parser.add_argument("--target_resolution", type=int, default=128,
                       help="Target resolution for resizing (default: 128)")
    parser.add_argument("--n_workers", type=int, default=8,
                       help="Number of parallel workers")
    parser.add_argument("--class_name", type=str, default=None,
                       help="Specific class to process (e.g., 'Aorta')")
    parser.add_argument("--all", action="store_true",
                       help="Process all 10 classes")
    parser.add_argument("--dry_run", action="store_true",
                       help="Only show what would be processed")
    parser.add_argument("--skip_coronary", action="store_true",
                       help="Skip Coronary (already processed)")
    
    args = parser.parse_args()
    
    # Determine which classes to process
    if args.all:
        classes_to_process = list(CLASS_IDS.keys())
        classes_to_process.remove("Background")
        if args.skip_coronary:
            classes_to_process.remove("Coronary")
    elif args.class_name:
        if args.class_name not in CLASS_IDS:
            print(f"Error: Unknown class '{args.class_name}'")
            print(f"Available: {list(CLASS_IDS.keys())}")
            sys.exit(1)
        classes_to_process = [args.class_name]
    else:
        print("Error: Specify --all or --class_name")
        parser.print_help()
        sys.exit(1)
    
    # Get case IDs
    case_ids = get_case_ids(args.labels_dir)
    
    print(f"\n{'='*70}")
    print("NeAR v2.0 - Multi-Class Tier2 Data Preparation")
    print(f"{'='*70}")
    print(f"Found {len(case_ids)} cases")
    print(f"Classes to process: {classes_to_process}")
    print(f"Target resolution: {args.target_resolution}")
    print(f"Output base: {args.output_dir}")
    print(f"{'='*70}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for cls in classes_to_process:
            output_dir = Path(args.output_dir) / f"{cls.lower()}_tier2"
            print(f"  - {cls}: {output_dir}")
        print("\nRun without --dry_run to actually process.")
        return
    
    if len(case_ids) == 0:
        print("[ERROR] No cases found! Check your labels_dir path.")
        sys.exit(1)
    
    # Process each class
    config = {
        "images_dir": args.images_dir,
        "labels_dir": args.labels_dir,
        "base_output_dir": args.output_dir,
        "target_resolution": args.target_resolution,
    }
    
    all_summaries = {}
    for class_name in classes_to_process:
        summary = process_class(class_name, config, case_ids, args.n_workers)
        all_summaries[class_name] = {
            "success": summary["success"],
            "failed": summary["failed"],
            "statistics": summary.get("statistics", {})
        }
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Class':<15} {'Success':>10} {'Failed':>10} {'Ratio After':>15}")
    print("-" * 55)
    for cls, data in all_summaries.items():
        ratio = data.get("statistics", {}).get("avg_ratio_after", 0) * 100
        print(f"{cls:<15} {data['success']:>10} {data['failed']:>10} {ratio:>14.2f}%")
    print(f"{'='*70}\n")
    
    # Save global summary
    global_summary_path = Path(args.output_dir) / "all_classes_summary.json"
    with open(global_summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Global summary saved to: {global_summary_path}")


if __name__ == "__main__":
    main()
