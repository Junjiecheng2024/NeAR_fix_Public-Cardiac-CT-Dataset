#!/usr/bin/env python3
"""
Visualize Coronary Tier2 preprocessed data.
Shows CT, coronary mask, and context mask side by side.

Usage:
    python visualize_tier2_sample.py --sample_dir /path/to/coronary_tier2/1
    python visualize_tier2_sample.py --sample_dir /path/to/coronary_tier2/1 --slice_idx 128
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_sample(sample_dir):
    """Load all data for a sample."""
    sample_dir = Path(sample_dir)
    
    data = {}
    
    # Load CT
    ct_path = sample_dir / "ct.npy"
    if ct_path.exists():
        data["ct"] = np.load(ct_path)
        print(f"CT shape: {data['ct'].shape}, range: [{data['ct'].min():.3f}, {data['ct'].max():.3f}]")
    
    # Load coronary mask
    mask_path = sample_dir / "mask_coronary.npy"
    if mask_path.exists():
        data["mask_coronary"] = np.load(mask_path)
        voxel_count = data["mask_coronary"].sum()
        print(f"Coronary mask shape: {data['mask_coronary'].shape}, voxels: {voxel_count}")
    
    # Load context mask
    context_path = sample_dir / "mask_context.npy"
    if context_path.exists():
        data["mask_context"] = np.load(context_path)
        print(f"Context mask shape: {data['mask_context'].shape}")
    
    # Load full segmentation
    seg_path = sample_dir / "seg_full.npy"
    if seg_path.exists():
        data["seg_full"] = np.load(seg_path)
        unique_labels = np.unique(data["seg_full"])
        print(f"Full seg shape: {data['seg_full'].shape}, labels: {unique_labels}")
    
    # Load crop params
    params_path = sample_dir / "crop_params.json"
    if params_path.exists():
        with open(params_path) as f:
            data["crop_params"] = json.load(f)
        print(f"Crop origin: {data['crop_params']['origin']}")
        print(f"Original shape: {data['crop_params']['original_shape']}")
        print(f"Voxel ratio improvement: {data['crop_params']['ratio_improvement']:.1f}x")
    
    return data


def visualize_slices(data, slice_idx=None, save_path=None):
    """Visualize axial, coronal, and sagittal slices."""
    ct = data.get("ct")
    mask = data.get("mask_coronary")
    context = data.get("mask_context")
    
    if ct is None:
        print("No CT data found!")
        return
    
    d, h, w = ct.shape
    
    # Default to middle slice
    if slice_idx is None:
        slice_idx = d // 2
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f"Coronary Tier2 Sample Visualization (Slice {slice_idx})", fontsize=14)
    
    # Row 1: Axial views (different slices)
    for i, z in enumerate([slice_idx - d//4, slice_idx, slice_idx + d//4]):
        z = max(0, min(z, d-1))
        ax = axes[0, i]
        ax.imshow(ct[z], cmap='gray', vmin=0, vmax=1)
        if mask is not None:
            ax.contour(mask[z], levels=[0.5], colors='red', linewidths=1)
        if context is not None:
            ax.contour(context[z], levels=[0.5], colors='blue', linewidths=0.5, alpha=0.5)
        ax.set_title(f"Axial z={z}")
        ax.axis('off')
    
    # Legend
    axes[0, 3].text(0.1, 0.5, "Red: Coronary\nBlue: Context (Myo+Aorta)", 
                     fontsize=12, transform=axes[0, 3].transAxes)
    axes[0, 3].axis('off')
    
    # Row 2: Coronal views
    for i, y in enumerate([h//4, h//2, 3*h//4]):
        ax = axes[1, i]
        ax.imshow(ct[:, y, :], cmap='gray', vmin=0, vmax=1, aspect='auto')
        if mask is not None:
            ax.contour(mask[:, y, :], levels=[0.5], colors='red', linewidths=1)
        if context is not None:
            ax.contour(context[:, y, :], levels=[0.5], colors='blue', linewidths=0.5, alpha=0.5)
        ax.set_title(f"Coronal y={y}")
        ax.axis('off')
    
    # Stats
    if mask is not None:
        voxel_ratio = mask.sum() / mask.size * 100
        axes[1, 3].text(0.1, 0.5, f"Coronary voxels: {mask.sum()}\n"
                         f"Voxel ratio: {voxel_ratio:.2f}%\n"
                         f"Volume shape: {ct.shape}",
                         fontsize=11, transform=axes[1, 3].transAxes)
    axes[1, 3].axis('off')
    
    # Row 3: Sagittal views
    for i, x in enumerate([w//4, w//2, 3*w//4]):
        ax = axes[2, i]
        ax.imshow(ct[:, :, x], cmap='gray', vmin=0, vmax=1, aspect='auto')
        if mask is not None:
            ax.contour(mask[:, :, x], levels=[0.5], colors='red', linewidths=1)
        if context is not None:
            ax.contour(context[:, :, x], levels=[0.5], colors='blue', linewidths=0.5, alpha=0.5)
        ax.set_title(f"Sagittal x={x}")
        ax.axis('off')
    
    # 3D preview (max projection)
    if mask is not None:
        mip = np.max(mask, axis=0)
        axes[2, 3].imshow(mip, cmap='hot')
        axes[2, 3].set_title("Coronary MIP (axial)")
        axes[2, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize Coronary Tier2 sample")
    parser.add_argument("--sample_dir", type=str, required=True,
                        help="Path to sample directory (e.g., coronary_tier2/1)")
    parser.add_argument("--slice_idx", type=int, default=None,
                        help="Slice index for visualization (default: middle)")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save figure (e.g., sample_1.png)")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Loading sample: {args.sample_dir}")
    print(f"{'='*60}\n")
    
    data = load_sample(args.sample_dir)
    
    if not data:
        print("No data found!")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Generating visualization...")
    print(f"{'='*60}\n")
    
    visualize_slices(data, args.slice_idx, args.save)


if __name__ == "__main__":
    main()
