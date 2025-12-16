#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
map_tier2_to_global.py
-----------------------
Map Tier2 predictions from cropped space back to global 256³ space.
This is Phase 1.5 in the NeAR v2.0 pipeline.

Usage:
    python map_tier2_to_global.py \
        --tier2_dir /path/to/tier2_predictions \
        --crop_params_dir /path/to/coronary_tier2 \
        --output_dir /path/to/global_predictions \
        --global_shape 256
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.ndimage import zoom


def load_crop_params(crop_params_path: str) -> dict:
    """Load crop parameters from JSON file."""
    with open(crop_params_path) as f:
        return json.load(f)


def map_to_global(
    pred_crop: np.ndarray, 
    crop_params: dict, 
    global_shape: tuple = (256, 256, 256),
    is_prob: bool = True
) -> np.ndarray:
    """
    Map cropped prediction back to global coordinate space.
    
    Args:
        pred_crop: Prediction in crop space (D_crop, H_crop, W_crop)
        crop_params: Crop parameters dict with 'origin', 'size', 'cropped_shape'
        global_shape: Target global shape
        is_prob: If True, use linear interpolation; else nearest neighbor
    
    Returns:
        pred_global: Prediction mapped to global space
    """
    origin = np.array(crop_params['origin'])  # [z, y, x] in original space
    original_crop_size = np.array(crop_params['size'])  # [d, h, w] before resize
    
    # Check if resize was applied during preprocessing
    if crop_params.get('resize_applied', False):
        # pred_crop is at resized resolution, need to resize back to original crop size
        zoom_factors = original_crop_size / np.array(pred_crop.shape)
        order = 1 if is_prob else 0
        pred_original_size = zoom(pred_crop, zoom_factors, order=order)
    else:
        pred_original_size = pred_crop
    
    # Create global volume
    pred_global = np.zeros(global_shape, dtype=np.float32)
    
    # Compute end coordinates
    end = origin + np.array(pred_original_size.shape)
    
    # Clip to global bounds
    valid_start = np.maximum(origin, 0)
    valid_end = np.minimum(end, np.array(global_shape))
    
    # Compute offsets in crop space
    crop_start = valid_start - origin
    crop_end = crop_start + (valid_end - valid_start)
    
    # Place crop into global volume
    pred_global[
        valid_start[0]:valid_end[0],
        valid_start[1]:valid_end[1],
        valid_start[2]:valid_end[2]
    ] = pred_original_size[
        crop_start[0]:crop_end[0],
        crop_start[1]:crop_end[1],
        crop_start[2]:crop_end[2]
    ]
    
    return pred_global


def process_single_case(args) -> dict:
    """Process a single case."""
    case_id, config = args
    
    try:
        # Load crop parameters
        crop_params_path = Path(config['crop_params_dir']) / case_id / "crop_params.json"
        if not crop_params_path.exists():
            return {"case_id": case_id, "status": "missing_params"}
        
        crop_params = load_crop_params(crop_params_path)
        
        # Load prediction (probability map from inference)
        pred_path = Path(config['tier2_dir']) / f"{case_id}_prob.npy"
        if not pred_path.exists():
            # Try alternative naming
            pred_path = Path(config['tier2_dir']) / case_id / "pred_prob.npy"
        if not pred_path.exists():
            return {"case_id": case_id, "status": "missing_pred", "error": str(pred_path)}
        
        pred_crop = np.load(pred_path)
        
        # If prediction has channel dimension, squeeze it
        if pred_crop.ndim == 4:
            pred_crop = pred_crop.squeeze(0)
        
        # Map to global space
        global_shape = (config['global_shape'],) * 3
        pred_global = map_to_global(
            pred_crop, 
            crop_params, 
            global_shape,
            is_prob=True
        )
        
        # Save mapped prediction
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / f"{case_id}_prob_global.npy", pred_global)
        
        # Also save binary mask (threshold at 0.5)
        mask_global = (pred_global > 0.5).astype(np.uint8)
        np.save(output_dir / f"{case_id}_mask_global.npy", mask_global)
        
        return {
            "case_id": case_id, 
            "status": "success",
            "crop_shape": list(pred_crop.shape),
            "global_shape": list(pred_global.shape),
            "coverage_ratio": float(pred_global.sum() / pred_global.size)
        }
        
    except Exception as e:
        import traceback
        return {
            "case_id": case_id, 
            "status": "error", 
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    parser = argparse.ArgumentParser(description="Map Tier2 predictions to global space")
    parser.add_argument("--tier2_dir", type=str, required=True,
                        help="Directory containing Tier2 predictions")
    parser.add_argument("--crop_params_dir", type=str, required=True,
                        help="Directory containing crop parameters (output of prepare_coronary_tier2.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for global-space predictions")
    parser.add_argument("--global_shape", type=int, default=256,
                        help="Global volume shape (default: 256)")
    parser.add_argument("--n_workers", type=int, default=8,
                        help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Find all cases
    crop_params_dir = Path(args.crop_params_dir)
    case_ids = [d.name for d in crop_params_dir.iterdir() 
                if d.is_dir() and (d / "crop_params.json").exists()]
    
    print(f"\n{'='*70}")
    print("NeAR v2.0 Phase 1.5: Map Tier2 to Global Space")
    print(f"{'='*70}")
    print(f"Found {len(case_ids)} cases")
    print(f"Tier2 predictions: {args.tier2_dir}")
    print(f"Crop params: {args.crop_params_dir}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*70}\n")
    
    if len(case_ids) == 0:
        print("[ERROR] No cases found!")
        sys.exit(1)
    
    config = {
        'tier2_dir': args.tier2_dir,
        'crop_params_dir': args.crop_params_dir,
        'output_dir': args.output_dir,
        'global_shape': args.global_shape
    }
    
    process_args = [(cid, config) for cid in case_ids]
    
    results = []
    if args.n_workers > 1:
        with Pool(args.n_workers) as pool:
            for result in tqdm(pool.imap(process_single_case, process_args),
                             total=len(case_ids), desc="Mapping"):
                results.append(result)
                if result["status"] != "success":
                    print(f"\n[WARN] {result['case_id']}: {result['status']}")
    else:
        for item in tqdm(process_args, desc="Mapping"):
            result = process_single_case(item)
            results.append(result)
            if result["status"] != "success":
                print(f"\n[WARN] {result['case_id']}: {result['status']}")
    
    # Summary
    success = sum(1 for r in results if r['status'] == 'success')
    print(f"\n{'='*70}")
    print(f"Mapping complete: {success}/{len(case_ids)} successful")
    print(f"Output saved to: {args.output_dir}")
    print(f"{'='*70}\n")
    
    # Save summary
    summary_path = Path(args.output_dir) / "mapping_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({"results": results, "config": config}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
