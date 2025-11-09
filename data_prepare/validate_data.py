#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data validation script for preprocessed datasets."""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from near.datasets.refine_dataset import CardiacMultiClassDataset

def validate_dataset(data_root, num_samples_to_check=5):
    """Validate preprocessed dataset integrity and statistics.
    
    Args:
        data_root: Root directory of preprocessed data
        num_samples_to_check: Number of random samples to validate
    """
    print("=" * 80)
    print("Dataset Validation")
    print("=" * 80)
    
    data_root = Path(data_root)
    print(f"\nData root: {data_root}")
    
    required_dirs = ["appearance", "shape"]
    required_files = ["info.csv", "class_statistics.csv", "class_weights.json"]
    
    print("\nChecking directory structure:")
    for d in required_dirs:
        exists = (data_root / d).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {d}/")
    
    for f in required_files:
        exists = (data_root / f).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {f}")
    
    print("\nLoading dataset...")
    try:
        dataset = CardiacMultiClassDataset(
            root=str(data_root),
            resolution=None,
            normalize=True
        )
        print(f"  ✓ Dataset loaded successfully: {len(dataset)} samples")
    except Exception as e:
        print(f"  ✗ Dataset loading failed: {e}")
        return
    
    print("\nClass statistics:")
    stats_path = data_root / "class_statistics.csv"
    if stats_path.exists():
        stats_df = pd.read_csv(stats_path)
        print(stats_df.to_string(index=False))
    
    print(f"\nValidating {num_samples_to_check} random samples:")
    indices = np.random.choice(len(dataset), min(num_samples_to_check, len(dataset)), replace=False)
    
    for idx in indices:
        try:
            _, app, seg = dataset[idx]
            case_id = dataset.info.loc[idx, dataset.id_key]
            
            app_shape = app.shape
            seg_shape = seg.shape
            
            app_min, app_max = app.min().item(), app.max().item()
            seg_min, seg_max = seg.min().item(), seg.max().item()
            seg_unique = np.unique(seg.numpy())
            
            print(f"\n  Sample {idx} (ID: {case_id}):")
            print(f"    - Appearance shape: {app_shape}, range: [{app_min:.4f}, {app_max:.4f}]")
            print(f"    - Segmentation shape: {seg_shape}, range: [{seg_min}, {seg_max}]")
            print(f"    - Classes present: {seg_unique.tolist()}")
            
            if seg_max >= 11:
                print(f"    ⚠ Warning: Segmentation contains out-of-range class {seg_max}")
            if seg_min < 0:
                print(f"    ⚠ Warning: Segmentation contains negative values {seg_min}")
            
        except Exception as e:
            print(f"\n  Sample {idx} validation failed: {e}")
    
    print("\nGenerating visualization...")
    try:
        idx = 0
        _, app, seg = dataset[idx]
        case_id = dataset.info.loc[idx, dataset.id_key]
        
        app_np = app.numpy()[0]
        seg_np = seg.numpy()[0]
        
        mid_slice = app_np.shape[0] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(app_np[mid_slice], cmap='gray')
        axes[0].set_title(f"CT Image (Case {case_id})")
        axes[0].axis('off')
        
        axes[1].imshow(app_np[mid_slice], cmap='gray', alpha=0.7)
        axes[1].imshow(seg_np[mid_slice], cmap='tab20', alpha=0.5, vmin=0, vmax=10)
        axes[1].set_title("Segmentation Overlay")
        axes[1].axis('off')
        
        axes[2].imshow(seg_np[mid_slice], cmap='tab20', vmin=0, vmax=10)
        axes[2].set_title("Segmentation Only")
        axes[2].axis('off')
        
        class_names = [
            "Background", "Myocardium", "LA", "LV", "RA", "RV",
            "Aorta", "PA", "LAA", "Coronary", "PV"
        ]
        legend_text = "\n".join([f"{i}: {name}" for i, name in enumerate(class_names)])
        fig.text(1.02, 0.5, legend_text, fontsize=9, verticalalignment='center')
        
        plt.tight_layout()
        
        vis_path = data_root / "dataset_validation.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Visualization saved to: {vis_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"  ✗ Visualization failed: {e}")
    
    print("\n" + "=" * 80)
    print("Validation complete!")
    print("=" * 80)
    print("\nIf all checks passed, you can start training:")
    print("  cd repairing/near_repairing")
    print("  python3 near_repair.py")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset validation script")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of preprocessed data")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to check")
    
    args = parser.parse_args()
    
    validate_dataset(args.data_root, args.num_samples)
