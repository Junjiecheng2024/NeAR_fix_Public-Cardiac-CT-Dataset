"""
Evaluation script for comparing original and refined masks.
Computes surface distance metrics and visual quality assessment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import binary_erosion, binary_dilation
from collections import defaultdict


def compute_dice(pred, gt):
    """Compute Dice score."""
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    if union == 0:
        return 1.0 if np.sum(pred) == 0 else 0.0
    return 2.0 * intersection / union


def compute_hausdorff_distance(pred, gt, percentile=95):
    """Compute Hausdorff distance (simplified version)."""
    # Get surface points
    pred_surface = pred ^ binary_erosion(pred)
    gt_surface = gt ^ binary_erosion(gt)
    
    if np.sum(pred_surface) == 0 or np.sum(gt_surface) == 0:
        return float('inf')
    
    # Get coordinates
    pred_coords = np.argwhere(pred_surface)
    gt_coords = np.argwhere(gt_surface)
    
    # Compute distances (using sampling for efficiency)
    max_samples = 10000
    if len(pred_coords) > max_samples:
        indices = np.random.choice(len(pred_coords), max_samples, replace=False)
        pred_coords = pred_coords[indices]
    if len(gt_coords) > max_samples:
        indices = np.random.choice(len(gt_coords), max_samples, replace=False)
        gt_coords = gt_coords[indices]
    
    # Compute minimum distances
    from scipy.spatial.distance import cdist
    dist_pred_to_gt = cdist(pred_coords, gt_coords, metric='euclidean')
    dist_gt_to_pred = cdist(gt_coords, pred_coords, metric='euclidean')
    
    min_dist_pred_to_gt = np.min(dist_pred_to_gt, axis=1)
    min_dist_gt_to_pred = np.min(dist_gt_to_pred, axis=1)
    
    # Percentile Hausdorff distance
    hd = max(
        np.percentile(min_dist_pred_to_gt, percentile),
        np.percentile(min_dist_gt_to_pred, percentile)
    )
    
    return hd


def compute_surface_smoothness(mask):
    """Compute surface smoothness (lower is smoother)."""
    # Compute surface
    surface = mask ^ binary_erosion(mask)
    
    if np.sum(surface) == 0:
        return 0.0
    
    # Compute gradient magnitude on surface
    from scipy.ndimage import sobel
    grad_x = sobel(mask.astype(float), axis=0)
    grad_y = sobel(mask.astype(float), axis=1)
    grad_z = sobel(mask.astype(float), axis=2)
    
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    # Average gradient magnitude on surface
    smoothness = np.mean(grad_mag[surface > 0])
    
    return smoothness


def compute_connected_components(mask):
    """Count number of connected components."""
    from scipy.ndimage import label
    labeled, n_components = label(mask)
    return n_components


def evaluate_sample(original_mask, refined_mask):
    """Evaluate quality metrics for a single sample."""
    metrics = {}
    
    # Dice score (should be high, indicating refinement preserves structure)
    metrics['dice'] = compute_dice(refined_mask, original_mask)
    
    # Surface smoothness (lower is better for refined)
    metrics['original_smoothness'] = compute_surface_smoothness(original_mask)
    metrics['refined_smoothness'] = compute_surface_smoothness(refined_mask)
    metrics['smoothness_improvement'] = metrics['original_smoothness'] - metrics['refined_smoothness']
    
    # Connected components (should decrease after refinement)
    metrics['original_cc'] = compute_connected_components(original_mask)
    metrics['refined_cc'] = compute_connected_components(refined_mask)
    
    # Volume preservation (should be close to 1.0)
    original_vol = np.sum(original_mask)
    refined_vol = np.sum(refined_mask)
    if original_vol > 0:
        metrics['volume_ratio'] = refined_vol / original_vol
    else:
        metrics['volume_ratio'] = 0.0 if refined_vol == 0 else float('inf')
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate refinement quality')
    parser.add_argument('--original_dir', type=str, required=True,
                       help='Directory with original masks')
    parser.add_argument('--refined_dir', type=str, required=True,
                       help='Directory with refined masks')
    parser.add_argument('--data_path', type=str,
                       default='../../../dataset/near_format_data',
                       help='Path to dataset')
    parser.add_argument('--class_name', type=str, default='Coronary',
                       help='Class name')
    parser.add_argument('--output_csv', type=str, default='evaluation_results.csv',
                       help='Output CSV file for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for refined probability maps')
    
    args = parser.parse_args()
    
    # Load sample info
    info_path = os.path.join(args.data_path, 'info.csv')
    info_df = pd.read_csv(info_path)
    sample_ids = info_df['sample_id'].values
    
    print(f"\n{'='*70}")
    print(f"Evaluating refinement for {args.class_name}")
    print(f"Original masks: {args.original_dir}")
    print(f"Refined masks: {args.refined_dir}")
    print(f"Total samples: {len(sample_ids)}")
    print(f"{'='*70}\n")
    
    results = []
    
    for sample_id in tqdm(sample_ids):
        # Load original mask
        original_path = os.path.join(args.original_dir, f'{sample_id}.npy')
        if not os.path.exists(original_path):
            print(f"Warning: Original mask not found for {sample_id}")
            continue
        original_mask = np.load(original_path)
        
        # Load refined mask
        refined_path = os.path.join(args.refined_dir, f'{sample_id}_refined.npy')
        if not os.path.exists(refined_path):
            # Try probability map
            refined_path = os.path.join(args.refined_dir, f'{sample_id}_prob.npy')
            if os.path.exists(refined_path):
                refined_prob = np.load(refined_path)
                refined_mask = (refined_prob > args.threshold).astype(np.uint8)
            else:
                print(f"Warning: Refined mask not found for {sample_id}")
                continue
        else:
            refined_mask = np.load(refined_path)
        
        # Compute metrics
        metrics = evaluate_sample(original_mask, refined_mask)
        metrics['sample_id'] = sample_id
        
        results.append(metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Compute summary statistics
    print(f"\n{'='*70}")
    print("Summary Statistics:")
    print(f"{'='*70}")
    
    metric_names = ['dice', 'original_smoothness', 'refined_smoothness',
                   'smoothness_improvement', 'original_cc', 'refined_cc', 'volume_ratio']
    
    for metric in metric_names:
        if metric in results_df.columns:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            print(f"{metric:30s}: {mean_val:8.4f} ± {std_val:8.4f}")
    
    # Count CC improvements
    cc_improved = np.sum(results_df['refined_cc'] < results_df['original_cc'])
    cc_same = np.sum(results_df['refined_cc'] == results_df['original_cc'])
    cc_worse = np.sum(results_df['refined_cc'] > results_df['original_cc'])
    
    print(f"\nConnected Components Analysis:")
    print(f"  Improved (CC reduced): {cc_improved} ({100*cc_improved/len(results_df):.1f}%)")
    print(f"  Unchanged:             {cc_same} ({100*cc_same/len(results_df):.1f}%)")
    print(f"  Worse (CC increased):  {cc_worse} ({100*cc_worse/len(results_df):.1f}%)")
    
    print(f"\n{'='*70}\n")
    
    # Save results
    results_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to: {args.output_csv}\n")


if __name__ == "__main__":
    main()
