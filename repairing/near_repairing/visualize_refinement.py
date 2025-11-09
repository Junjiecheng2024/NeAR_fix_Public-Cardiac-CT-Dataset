"""
Visualization script for comparing original and refined masks.
Generates side-by-side comparison images for quality inspection.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd


def visualize_comparison(original, refined, sample_id, slice_idx=None, axis=2):
    """Create side-by-side comparison visualization."""
    # Select middle slice if not specified
    if slice_idx is None:
        slice_idx = original.shape[axis] // 2
    
    # Extract slices
    if axis == 0:
        orig_slice = original[slice_idx, :, :]
        ref_slice = refined[slice_idx, :, :]
    elif axis == 1:
        orig_slice = original[:, slice_idx, :]
        ref_slice = refined[:, slice_idx, :]
    else:  # axis == 2
        orig_slice = original[:, :, slice_idx]
        ref_slice = refined[:, :, slice_idx]
    
    # Create difference map
    diff = np.zeros_like(orig_slice, dtype=np.int8)
    diff[orig_slice & ~ref_slice] = -1  # Removed pixels
    diff[~orig_slice & ref_slice] = 1   # Added pixels
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original mask
    axes[0].imshow(orig_slice, cmap='gray', interpolation='nearest')
    axes[0].set_title(f'Original Mask\nSum: {np.sum(original)}')
    axes[0].axis('off')
    
    # Refined mask
    axes[1].imshow(ref_slice, cmap='gray', interpolation='nearest')
    axes[1].set_title(f'Refined Mask\nSum: {np.sum(refined)}')
    axes[1].axis('off')
    
    # Overlay
    overlay = np.zeros((*orig_slice.shape, 3))
    overlay[orig_slice > 0] = [0, 1, 0]  # Original: green
    overlay[ref_slice > 0] = [1, 0, 0]   # Refined: red
    overlay[orig_slice & ref_slice] = [1, 1, 0]  # Overlap: yellow
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay\n(Green: Orig, Red: Ref, Yellow: Both)')
    axes[2].axis('off')
    
    # Difference map
    cmap_diff = ListedColormap(['red', 'white', 'green'])
    im = axes[3].imshow(diff, cmap=cmap_diff, vmin=-1, vmax=1, interpolation='nearest')
    axes[3].set_title('Difference\n(Red: Removed, Green: Added)')
    axes[3].axis('off')
    
    plt.suptitle(f'Sample: {sample_id} | Slice: {slice_idx} (axis={axis})', fontsize=14)
    plt.tight_layout()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize refinement comparison')
    parser.add_argument('--original_dir', type=str, required=True,
                       help='Directory with original masks')
    parser.add_argument('--refined_dir', type=str, required=True,
                       help='Directory with refined masks')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for visualization images')
    parser.add_argument('--data_path', type=str,
                       default='../../../dataset/near_format_data',
                       help='Path to dataset')
    parser.add_argument('--sample_ids', type=str, nargs='+',
                       help='Specific sample IDs to visualize (default: first 10)')
    parser.add_argument('--n_samples', type=int, default=10,
                       help='Number of samples to visualize if sample_ids not specified')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for refined probability maps')
    parser.add_argument('--axis', type=int, default=2,
                       help='Axis for slicing (0, 1, or 2)')
    parser.add_argument('--slice_idx', type=int, default=None,
                       help='Specific slice index (default: middle slice)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get sample IDs
    if args.sample_ids:
        sample_ids = args.sample_ids
    else:
        info_path = os.path.join(args.data_path, 'info.csv')
        info_df = pd.read_csv(info_path)
        sample_ids = info_df['sample_id'].values[:args.n_samples]
    
    print(f"\n{'='*70}")
    print(f"Visualizing {len(sample_ids)} samples")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*70}\n")
    
    for sample_id in sample_ids:
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
        
        # Create visualization
        fig = visualize_comparison(
            original_mask, refined_mask, sample_id,
            slice_idx=args.slice_idx, axis=args.axis
        )
        
        # Save figure
        output_path = os.path.join(args.output_dir, f'{sample_id}_comparison.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved visualization for {sample_id}")
    
    print(f"\n{'='*70}")
    print(f"Visualization completed!")
    print(f"Images saved to: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
