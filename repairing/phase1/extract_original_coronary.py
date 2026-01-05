#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_original_coronary.py
----------------------------
Extract Coronary (class_id=9) from original GT segmentations and resize to 256³.
This bypasses Phase 1 NeAR model to test Phase 2/3 with GT masks.

Usage:
    python extract_original_coronary.py \
        --input_dir /path/to/original/segmentations \
        --output_dir /path/to/coronary_global_gt \
        --target_size 256
"""

import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm
import glob


CORONARY_CLASS_ID = 9


def resize_mask(mask: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize 3D binary mask to target_size³ using nearest neighbor interpolation.
    
    Args:
        mask: Binary mask (D, H, W)
        target_size: Target dimension (e.g., 256)
    
    Returns:
        Resized mask (target_size, target_size, target_size)
    """
    current_shape = mask.shape
    zoom_factors = [target_size / s for s in current_shape]
    
    # Use order=0 (nearest neighbor) to preserve binary values
    resized = zoom(mask.astype(np.float32), zoom_factors, order=0)
    
    return (resized > 0.5).astype(np.float32)


def extract_coronary_from_segmentation(seg_path: str, target_size: int) -> np.ndarray:
    """
    Load segmentation NIfTI, extract Coronary class, and resize.
    
    Args:
        seg_path: Path to segmentation .nii.gz file
        target_size: Target dimension
    
    Returns:
        Binary coronary mask resized to target_size³
    """
    # Load NIfTI
    img = nib.load(seg_path)
    data = img.get_fdata().astype(np.int32)
    
    # Extract Coronary (class_id = 9)
    coronary_mask = (data == CORONARY_CLASS_ID).astype(np.float32)
    
    # Check if coronary exists
    if coronary_mask.sum() == 0:
        print(f"  Warning: No coronary found in {os.path.basename(seg_path)}")
        return np.zeros((target_size, target_size, target_size), dtype=np.float32)
    
    # Resize to target size
    resized = resize_mask(coronary_mask, target_size)
    
    return resized


def process_directory(args):
    """Process all segmentation files in input directory."""
    input_dir = args.input_dir
    output_dir = args.output_dir
    target_size = args.target_size
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .nii.gz files
    seg_files = glob.glob(os.path.join(input_dir, "*.nii.gz"))
    
    if not seg_files:
        print(f"No .nii.gz files found in {input_dir}")
        return
    
    print(f"Found {len(seg_files)} segmentation files")
    print(f"Target size: {target_size}³")
    print(f"Output dir: {output_dir}")
    print("-" * 50)
    
    stats = {
        'processed': 0,
        'with_coronary': 0,
        'without_coronary': 0
    }
    
    for seg_path in tqdm(seg_files, desc="Extracting Coronary"):
        # Extract case ID from filename (e.g., "123.nii.gz" -> "123")
        filename = os.path.basename(seg_path)
        case_id = filename.replace(".nii.gz", "").replace(".nii.img", "")
        
        # Extract and resize coronary
        coronary = extract_coronary_from_segmentation(seg_path, target_size)
        
        # Save as numpy array in Phase 1 output format: {case_id}_mask.npy
        output_path = os.path.join(output_dir, f"{case_id}_mask.npy")
        np.save(output_path, coronary)
        
        stats['processed'] += 1
        if coronary.sum() > 0:
            stats['with_coronary'] += 1
        else:
            stats['without_coronary'] += 1
    
    print("-" * 50)
    print(f"Processing complete!")
    print(f"  Total processed: {stats['processed']}")
    print(f"  With coronary: {stats['with_coronary']}")
    print(f"  Without coronary: {stats['without_coronary']}")
    print(f"\nOutput saved to: {output_dir}")
    print(f"Files are named: {{case_id}}_mask.npy")
    print("\nNext steps:")
    print("  1. Run Phase 2: python perform_morphology_v2.py --input_dir {output_dir} --output_dir coronary_morph_gt --target_class 9")
    print("  2. Temporarily swap coronary_morph with coronary_morph_gt")
    print("  3. Run Phase 3: python phase3.py ...")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Coronary from original GT segmentations"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to original segmentations directory (contains .nii.gz files)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for extracted coronary masks"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=256,
        help="Target volume size (default: 256)"
    )
    
    args = parser.parse_args()
    process_directory(args)


if __name__ == "__main__":
    main()
