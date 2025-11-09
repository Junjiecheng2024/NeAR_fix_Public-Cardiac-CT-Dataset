"""
Quick evaluation script using pre-computed inference results.
"""

import os
import sys
import numpy as np
import argparse
from scipy.ndimage import label as connected_components
from skimage.transform import resize
import matplotlib.pyplot as plt

# Add NeAR to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance
    HAS_SURFACE_DISTANCE = True
except ImportError:
    HAS_SURFACE_DISTANCE = False
    print("Warning: surface_distance not available, skipping surface metrics")


def dice_coefficient(mask1, mask2):
    """Calculate Dice coefficient between two binary masks."""
    intersection = (mask1 * mask2).sum()
    return 2.0 * intersection / (mask1.sum() + mask2.sum() + 1e-6)


def analyze_connected_components(mask, min_size=10):
    """Analyze connected components."""
    labeled_mask, num_features = connected_components(mask)
    
    component_sizes = []
    for i in range(1, num_features + 1):
        size = (labeled_mask == i).sum()
        if size >= min_size:
            component_sizes.append(size)
    
    component_sizes = sorted(component_sizes, reverse=True)
    return len(component_sizes), component_sizes


def surface_dice_metric(mask_gt, mask_pred, spacing_mm=(1.0, 1.0, 1.0), tolerance_mm=1.0):
    """Calculate Surface Dice at specified tolerance."""
    if not HAS_SURFACE_DISTANCE:
        return None, (None, None)
    
    mask_gt_bool = mask_gt.astype(bool)
    mask_pred_bool = mask_pred.astype(bool)
    
    if not mask_gt_bool.any() or not mask_pred_bool.any():
        return 0.0, (float('inf'), float('inf'))
    
    surface_distances = compute_surface_distances(
        mask_gt_bool, mask_pred_bool, spacing_mm
    )
    
    surf_dice = compute_surface_dice_at_tolerance(surface_distances, tolerance_mm)
    
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    
    avg_dist_gt_to_pred = (
        np.sum(distances_gt_to_pred * surfel_areas_gt) / np.sum(surfel_areas_gt)
        if len(surfel_areas_gt) > 0 else float('inf')
    )
    avg_dist_pred_to_gt = (
        np.sum(distances_pred_to_gt * surfel_areas_pred) / np.sum(surfel_areas_pred)
        if len(surfel_areas_pred) > 0 else float('inf')
    )
    
    return surf_dice, (avg_dist_gt_to_pred, avg_dist_pred_to_gt)


def evaluate_single_sample(sample_id, inference_dir, data_root, class_index=9):
    """Evaluate a single sample using pre-computed inference result."""
    
    print(f"\n{'='*70}")
    print(f"Evaluating Sample {sample_id}")
    print(f"{'='*70}")
    
    # Load refined mask from inference
    refined_path = os.path.join(inference_dir, f'{sample_id}.npy')
    if not os.path.exists(refined_path):
        print(f"Error: Refined mask not found at {refined_path}")
        return None
    
    refined_seg = np.load(refined_path)
    print(f"Loaded refined mask: shape {refined_seg.shape}")
    
    # Binarize if needed (inference outputs probabilities)
    if refined_seg.max() <= 1.0:
        refined_seg = (refined_seg > 0.5).astype(np.uint8)
    
    # Load original segmentation
    shape_path = os.path.join(data_root, 'shape', f'{sample_id}.npy')
    original_multi = np.load(shape_path)
    original_seg = (original_multi == class_index).astype(np.uint8)
    
    print(f"Original shape: {original_seg.shape}")
    
    # Resize refined to match original if needed
    if refined_seg.shape != original_seg.shape:
        print(f"Resizing refined from {refined_seg.shape} to {original_seg.shape}")
        refined_seg = resize(refined_seg, original_seg.shape, 
                            order=0, preserve_range=True, anti_aliasing=False)
        refined_seg = (refined_seg > 0.5).astype(np.uint8)
    
    # Connected Components Analysis
    num_cc_orig, sizes_orig = analyze_connected_components(original_seg)
    num_cc_refined, sizes_refined = analyze_connected_components(refined_seg)
    
    print(f"\nConnected Components:")
    print(f"  Original:  {num_cc_orig} CCs, largest: {sizes_orig[0] if sizes_orig else 0}")
    print(f"  Refined:   {num_cc_refined} CCs, largest: {sizes_refined[0] if sizes_refined else 0}")
    print(f"  Reduction: {num_cc_orig - num_cc_refined}")
    
    # Dice Score
    dice = dice_coefficient(original_seg, refined_seg)
    print(f"\nVolumetric Dice: {dice:.4f}")
    
    # Surface Dice
    if HAS_SURFACE_DISTANCE:
        print(f"\nSurface Dice (1mm tolerance):")
        surf_dice_1mm, (avg_gt_to_pred, avg_pred_to_gt) = surface_dice_metric(
            original_seg, refined_seg, spacing_mm=(1.0, 1.0, 1.0), tolerance_mm=1.0
        )
        print(f"  Surface Dice: {surf_dice_1mm:.4f}")
        if avg_gt_to_pred is not None:
            avg_dist = (avg_gt_to_pred + avg_pred_to_gt) / 2
            print(f"  Avg surface distance: {avg_dist:.3f} mm")
    
    # Volume Analysis
    orig_positive = original_seg.sum()
    refined_positive = refined_seg.sum()
    volume_change = (refined_positive - orig_positive) / (orig_positive + 1e-6) * 100
    
    print(f"\nVolume:")
    print(f"  Original: {orig_positive:,} voxels")
    print(f"  Refined:  {refined_positive:,} voxels")
    print(f"  Change:   {volume_change:+.2f}%")
    
    return {
        'sample_id': sample_id,
        'dice': dice,
        'surface_dice_1mm': surf_dice_1mm if HAS_SURFACE_DISTANCE else None,
        'num_cc_orig': num_cc_orig,
        'num_cc_refined': num_cc_refined,
        'volume_orig': orig_positive,
        'volume_refined': refined_positive,
        'volume_change_pct': volume_change
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate using pre-computed inference results')
    parser.add_argument('--inference_dir', type=str, required=True,
                       help='Directory containing inference results (.npy files)')
    parser.add_argument('--data_root', type=str, 
                       default='../../../../dataset/near_format_data',
                       help='Path to dataset root')
    parser.add_argument('--sample_ids', type=int, nargs='+',
                       help='Sample IDs to evaluate (default: first 10 available)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to evaluate if --sample_ids not specified')
    parser.add_argument('--class_index', type=int, default=9,
                       help='Class index (9 for Coronary)')
    parser.add_argument('--output_summary', type=str,
                       help='Path to save summary CSV (optional)')
    
    args = parser.parse_args()
    
    # Find available samples
    if args.sample_ids:
        sample_ids = args.sample_ids
    else:
        inference_files = sorted([f for f in os.listdir(args.inference_dir) if f.endswith('.npy')])
        # Filter out non-numeric filenames
        sample_ids = []
        for f in inference_files:
            try:
                sample_id = int(f.replace('.npy', ''))
                sample_ids.append(sample_id)
                if len(sample_ids) >= args.num_samples:
                    break
            except ValueError:
                continue  # Skip non-numeric filenames
    
    print(f"\n{'='*70}")
    print(f"Evaluating {len(sample_ids)} samples from inference results")
    print(f"Inference directory: {args.inference_dir}")
    print(f"{'='*70}")
    
    # Evaluate samples
    results = []
    for sample_id in sample_ids:
        result = evaluate_single_sample(
            sample_id, args.inference_dir, args.data_root, args.class_index
        )
        if result:
            results.append(result)
    
    # Summary statistics
    if results:
        print(f"\n\n{'='*70}")
        print(f"SUMMARY STATISTICS ({len(results)} samples)")
        print(f"{'='*70}")
        
        dices = [r['dice'] for r in results]
        cc_reductions = [r['num_cc_orig'] - r['num_cc_refined'] for r in results]
        volume_changes = [r['volume_change_pct'] for r in results]
        
        print(f"\nVolumetric Dice:")
        print(f"  Mean:   {np.mean(dices):.4f} ± {np.std(dices):.4f}")
        print(f"  Median: {np.median(dices):.4f}")
        print(f"  Range:  [{np.min(dices):.4f}, {np.max(dices):.4f}]")
        
        if HAS_SURFACE_DISTANCE:
            surf_dices = [r['surface_dice_1mm'] for r in results if r['surface_dice_1mm'] is not None]
            if surf_dices:
                print(f"\nSurface Dice (1mm):")
                print(f"  Mean:   {np.mean(surf_dices):.4f} ± {np.std(surf_dices):.4f}")
                print(f"  Median: {np.median(surf_dices):.4f}")
                print(f"  Range:  [{np.min(surf_dices):.4f}, {np.max(surf_dices):.4f}]")
        
        print(f"\nConnected Components Reduction:")
        print(f"  Mean:   {np.mean(cc_reductions):.1f}")
        print(f"  Median: {np.median(cc_reductions):.0f}")
        
        print(f"\nVolume Change:")
        print(f"  Mean:   {np.mean(volume_changes):+.2f}%")
        print(f"  Median: {np.median(volume_changes):+.2f}%")
        
        # Save summary if requested
        if args.output_summary:
            import csv
            with open(args.output_summary, 'w', newline='') as f:
                fieldnames = ['sample_id', 'dice', 'surface_dice_1mm', 
                            'num_cc_orig', 'num_cc_refined', 'cc_reduction',
                            'volume_orig', 'volume_refined', 'volume_change_pct']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    r['cc_reduction'] = r['num_cc_orig'] - r['num_cc_refined']
                    writer.writerow(r)
            print(f"\nSummary saved to: {args.output_summary}")
        
        print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
