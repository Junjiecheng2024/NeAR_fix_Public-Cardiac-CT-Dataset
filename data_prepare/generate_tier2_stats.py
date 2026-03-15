#!/usr/bin/env python3
"""
Generate statistics for Tier2 samples.
Computes voxel ratios before and after cropping.

Usage:
    python generate_tier2_stats.py --data_dir /path/to/class_tier2 --output stats.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool


def process_single_sample(sample_dir):
    """Process a single sample and return stats."""
    sample_dir = Path(sample_dir)
    case_id = sample_dir.name
    
    try:
        # Load crop params (contains pre-computed ratios)
        params_path = sample_dir / "crop_params.json"
        if not params_path.exists():
            return None
        
        with open(params_path) as f:
            params = json.load(f)
        
        # Load masks to verify (generic current format, with legacy fallback)
        mask_path = sample_dir / "mask_target.npy"
        if not mask_path.exists():
            mask_path = sample_dir / "mask_coronary.npy"
        mask_target = np.load(mask_path)
        mask_context = np.load(sample_dir / "mask_context.npy")
        
        # Calculate current voxel counts
        target_voxels = int(mask_target.sum())
        context_voxels = int(mask_context.sum())
        total_voxels = mask_target.size
        
        # Current ratio (after crop + resize)
        current_ratio = target_voxels / total_voxels
        
        # Original ratio (from params)
        original_ratio = params.get("voxel_ratio_before", 0)
        
        # Improvement factor
        improvement = params.get("ratio_improvement", 0)
        
        return {
            "case_id": case_id,
            "original_shape": "x".join(map(str, params.get("original_shape", []))),
            "cropped_shape": "x".join(map(str, params.get("cropped_shape", []))),
            "crop_origin": params.get("origin", []),
            "original_ratio_pct": original_ratio * 100,
            "current_ratio_pct": current_ratio * 100,
            "improvement_factor": improvement,
            "target_voxels": target_voxels,
            "context_voxels": context_voxels,
            "context_ratio_pct": context_voxels / total_voxels * 100
        }
    
    except Exception as e:
        print(f"[ERROR] {case_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate Tier2 statistics")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to a class-specific Tier2 directory")
    parser.add_argument("--output", type=str, default="tier2_stats.csv",
                        help="Output CSV file path")
    parser.add_argument("--n_workers", type=int, default=8,
                        help="Number of parallel workers")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Find all sample directories
    sample_dirs = sorted([d for d in data_dir.iterdir() 
                          if d.is_dir() and (d / "crop_params.json").exists()])
    
    print(f"\n{'='*70}")
    print("Generating Tier2 Statistics")
    print(f"{'='*70}")
    print(f"Data directory: {data_dir}")
    print(f"Found {len(sample_dirs)} samples")
    print(f"{'='*70}\n")
    
    # Process all samples
    results = []
    
    if args.n_workers > 1:
        with Pool(args.n_workers) as pool:
            for result in tqdm(pool.imap(process_single_sample, sample_dirs), 
                              total=len(sample_dirs), desc="Processing"):
                if result is not None:
                    results.append(result)
    else:
        for sample_dir in tqdm(sample_dirs, desc="Processing"):
            result = process_single_sample(sample_dir)
            if result is not None:
                results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by case_id (numeric sort)
    df["case_id_num"] = df["case_id"].astype(int)
    df = df.sort_values("case_id_num").drop(columns=["case_id_num"])
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Total samples: {len(df)}")
    print(f"\nOriginal Target Voxel Ratio (%):")
    print(f"  Mean:   {df['original_ratio_pct'].mean():.4f}%")
    print(f"  Std:    {df['original_ratio_pct'].std():.4f}%")
    print(f"  Min:    {df['original_ratio_pct'].min():.4f}%")
    print(f"  Max:    {df['original_ratio_pct'].max():.4f}%")
    
    print(f"\nCurrent Target Voxel Ratio (after crop+resize) (%):")
    print(f"  Mean:   {df['current_ratio_pct'].mean():.4f}%")
    print(f"  Std:    {df['current_ratio_pct'].std():.4f}%")
    print(f"  Min:    {df['current_ratio_pct'].min():.4f}%")
    print(f"  Max:    {df['current_ratio_pct'].max():.4f}%")
    
    print(f"\nImprovement Factor:")
    print(f"  Mean:   {df['improvement_factor'].mean():.2f}x")
    print(f"  Std:    {df['improvement_factor'].std():.2f}x")
    print(f"  Min:    {df['improvement_factor'].min():.2f}x")
    print(f"  Max:    {df['improvement_factor'].max():.2f}x")
    
    print(f"\nContext Mask (Myo+Aorta) Voxel Ratio (%):")
    print(f"  Mean:   {df['context_ratio_pct'].mean():.2f}%")
    print(f"  Samples with context > 0: {(df['context_voxels'] > 0).sum()}/{len(df)}")
    
    print(f"\nTarget Voxel Count:")
    print(f"  Mean:   {df['target_voxels'].mean():.0f}")
    print(f"  Min:    {df['target_voxels'].min():.0f}")
    print(f"  Max:    {df['target_voxels'].max():.0f}")
    
    print(f"\n{'='*70}")
    
    # Also create a summary file
    summary_path = args.output.replace(".csv", "_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Tier2 Statistics Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Total samples: {len(df)}\n\n")
        f.write(f"Original Target Ratio: {df['original_ratio_pct'].mean():.4f}% ± {df['original_ratio_pct'].std():.4f}%\n")
        f.write(f"Current Target Ratio:  {df['current_ratio_pct'].mean():.4f}% ± {df['current_ratio_pct'].std():.4f}%\n")
        f.write(f"Improvement Factor:      {df['improvement_factor'].mean():.2f}x\n\n")
        f.write(f"Context mask coverage:   {(df['context_voxels'] > 0).sum()}/{len(df)} samples\n")
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
