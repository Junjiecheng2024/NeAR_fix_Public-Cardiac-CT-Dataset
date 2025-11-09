#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crop_dataset_with_bbox.py
-------------------------
Batch-crop CT volumes and their label masks to a tight 3D bounding box around
the foreground (mask > 0). Saves cropped images/masks and writes an info.csv
summarizing bbox, shapes, and spacings.

Usage (example):
    python crop_dataset_with_bbox.py \
        --images_dir /path/to/images_nii \
        --labels_dir /path/to/labels_nii \
        --out_images_dir /path/to/out_images_nii \
        --out_labels_dir /path/to/out_labels_nii \
        --margin 10 \
        --pattern ".nii.gz"

Notes:
- The bbox is computed from the *label* volume (mask > 0). You can
  optionally restrict foreground to a subset of class IDs via --include_classes.
- The affine is updated so that world coordinates stay consistent for the cropped region.
- If a label is empty (no foreground), the case is skipped by default unless --keep_empty is set.
- No resampling is performed here; you can resize later with your own pipeline (e.g., to 128^3).
"""

import os
import sys
import csv
import argparse
from typing import List, Optional, Tuple
import numpy as np

try:
    import nibabel as nib
except ImportError:
    raise ImportError("Please install nibabel: pip install nibabel")

def parse_args():
    p = argparse.ArgumentParser(description="Crop NIfTI images/masks to ROI bbox.")
    p.add_argument("--images_dir", type=str, required=True, help="Directory of input image NIfTI files")
    p.add_argument("--labels_dir", type=str, required=True, help="Directory of input label NIfTI files")
    p.add_argument("--out_images_dir", type=str, required=True, help="Directory to save cropped images")
    p.add_argument("--out_labels_dir", type=str, required=True, help="Directory to save cropped labels")
    p.add_argument("--pattern", type=str, default=".nii.gz", help="Filename suffix/pattern (default: .nii.gz)")
    p.add_argument("--margin", type=int, default=10, help="Margin (in voxels) to expand bbox on each side")
    p.add_argument("--min_foreground_voxels", type=int, default=10, help="Skip case if foreground voxels < this")
    p.add_argument("--include_classes", type=int, nargs="*", default=None,
                   help="Optional class IDs to define foreground. If not set, foreground is (label>0).")
    p.add_argument("--keep_empty", action="store_true",
                   help="If set, will still copy volumes with empty masks (no crop).")
    p.add_argument("--csv_path", type=str, default=None, help="Where to write info.csv (default: out_labels_dir/info.csv)")
    return p.parse_args()

def list_cases(labels_dir: str, pattern: str) -> List[str]:
    files = []
    for f in os.listdir(labels_dir):
        if f.endswith(pattern):
            files.append(f)
    files.sort()
    return files

def load_nifti(path: str):
    img = nib.load(path)
    data = img.get_fdata()  # float64 by default
    affine = img.affine.copy()
    header = img.header.copy()
    return data, affine, header

def save_nifti(data: np.ndarray, affine: np.ndarray, header, out_path: str):
    # use original header to keep pixdim etc.; nib will update dimension automatically
    new_img = nib.Nifti1Image(data, affine, header=header)
    # ensure sform/qform set
    new_img.set_sform(affine, code=1)
    new_img.set_qform(affine, code=1)
    nib.save(new_img, out_path)

def compute_bbox_from_mask(mask: np.ndarray,
                           include_classes: Optional[List[int]] = None,
                           margin: int = 10,
                           min_foreground_voxels: int = 10) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns (start_idx[3], end_idx[3]) inclusive-exclusive bbox in index space,
    or None if not enough foreground.
    """
    if include_classes is None:
        fg = (mask > 0)
    else:
        fg = np.isin(mask, np.array(include_classes, dtype=mask.dtype))

    if fg.sum() < min_foreground_voxels:
        return None

    coords = np.argwhere(fg)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1  # make end exclusive
    # apply margins per axis
    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, np.array(mask.shape))
    return mins, maxs

def crop_with_bbox(vol: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    # Slices as [i0:i1, j0:j1, k0:k1] consistent with data array axes
    return vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

def adjust_affine(affine: np.ndarray, start: np.ndarray) -> np.ndarray:
    """
    Update affine so that world coords remain correct after cropping.
    new_affine[:3,3] = old_affine[:3,3] + old_affine[:3,:3] @ start_idx
    """
    new_aff = affine.copy()
    offset = affine[:3, :3].dot(start.astype(float))
    new_aff[:3, 3] = affine[:3, 3] + offset
    return new_aff

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def main():
    args = parse_args()
    ensure_dirs(args.out_images_dir, args.out_labels_dir)
    csv_path = args.csv_path or os.path.join(args.out_labels_dir, "info.csv")

    label_files = list_cases(args.labels_dir, args.pattern)
    if not label_files:
        print(f"[ERROR] No label files found in {args.labels_dir} with pattern {args.pattern}")
        sys.exit(1)

    rows = []
    for i, lf in enumerate(label_files, 1):
        label_in = os.path.join(args.labels_dir, lf)
        image_in = os.path.join(args.images_dir, lf)  # assume same filename
        if not os.path.exists(image_in):
            print(f"[WARN] Missing image for {lf}, skip.")
            continue

        try:
            label_data, label_aff, label_hdr = load_nifti(label_in)
            image_data, image_aff, image_hdr = load_nifti(image_in)
        except Exception as e:
            print(f"[ERROR] Failed to read {lf}: {e}")
            continue

        # Sanity checks
        if label_data.shape != image_data.shape:
            print(f"[ERROR] Shape mismatch for {lf}: label{label_data.shape} vs image{image_data.shape}")
            continue

        bbox = compute_bbox_from_mask(
            label_data,
            include_classes=args.include_classes,
            margin=args.margin,
            min_foreground_voxels=args.min_foreground_voxels
        )

        if bbox is None:
            if args.keep_empty:
                # copy without crop
                img_out = os.path.join(args.out_images_dir, lf)
                lab_out = os.path.join(args.out_labels_dir, lf)
                save_nifti(image_data, image_aff, image_hdr, img_out)
                save_nifti(label_data, label_aff, label_hdr, lab_out)
                rows.append({
                    "id": os.path.splitext(os.path.splitext(lf)[0])[0],
                    "image_in": image_in, "label_in": label_in,
                    "image_out": img_out, "label_out": lab_out,
                    "bbox_start": [0, 0, 0], "bbox_end": list(label_data.shape),
                    "orig_shape": list(label_data.shape), "new_shape": list(label_data.shape),
                    "spacing": list(label_hdr.get_zooms()[:3]),
                    "note": "empty_foreground_kept"
                })
                print(f"[{i}/{len(label_files)}] {lf}: empty mask -> copied w/o crop.")
                continue
            else:
                print(f"[{i}/{len(label_files)}] {lf}: empty foreground -> skipped.")
                continue

        start, end = bbox
        cropped_label = crop_with_bbox(label_data, start, end)
        cropped_image = crop_with_bbox(image_data, start, end)

        new_aff = adjust_affine(label_aff, start)
        # Use same affine for image (assumed same as label initially)
        new_aff_img = adjust_affine(image_aff, start)

        img_out = os.path.join(args.out_images_dir, lf)
        lab_out = os.path.join(args.out_labels_dir, lf)

        save_nifti(cropped_image, new_aff_img, image_hdr, img_out)
        save_nifti(cropped_label, new_aff, label_hdr, lab_out)

        rows.append({
            "id": os.path.splitext(os.path.splitext(lf)[0])[0],
            "image_in": image_in, "label_in": label_in,
            "image_out": img_out, "label_out": lab_out,
            "bbox_start": start.tolist(), "bbox_end": end.tolist(),
            "orig_shape": list(label_data.shape), "new_shape": list(cropped_label.shape),
            "spacing": list(label_hdr.get_zooms()[:3]),
            "note": ""
        })

        print(f"[{i}/{len(label_files)}] {lf}: cropped -> shape {tuple(cropped_label.shape)} bbox {start.tolist()}->{end.tolist()}")

    # Write CSV
    if rows:
        fieldnames = ["id", "image_in", "label_in", "image_out", "label_out",
                      "bbox_start", "bbox_end", "orig_shape", "new_shape", "spacing", "note"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[OK] Wrote info.csv with {len(rows)} rows: {csv_path}")
    else:
        print("[WARN] No rows written. Nothing processed.")

if __name__ == "__main__":
    main()
