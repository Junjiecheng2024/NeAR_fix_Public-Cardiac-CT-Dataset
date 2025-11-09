"""
Evaluation script for Coronary refinement results.

Features:
1. Visualize comparison: Original CT + Original Segmentation + Refined Segmentation
2. Connected Components (CC) analysis before/after refinement
3. Dice score calculation
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import label as connected_components
from skimage.transform import resize
import argparse

# Add NeAR to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from near.models.nn3d.model_shape_only import EmbeddingDecoderShapeOnly
from near.utils.misc import to_var
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load trained NeAR model."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Model config (should match training config)
    model = EmbeddingDecoderShapeOnly(
        latent_dimension=256,
        decoder_channels=[64, 48, 32, 16]
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def inference_single_sample(model, sample_idx, resolution=256, device='cuda'):
    """
    Inference for a single sample.
    
    Args:
        model: trained NeAR model
        sample_idx: sample index (0-997)
        resolution: output resolution (128 or 256)
    
    Returns:
        refined_mask: (D, H, W) binary mask
    """
    print(f"Running inference for sample {sample_idx} at resolution {resolution}³...")
    
    # Generate 3D grid
    d = torch.linspace(-1, 1, resolution)
    meshx, meshy, meshz = torch.meshgrid(d, d, d, indexing='ij')
    grid = torch.stack([meshz, meshy, meshx], -1)  # (D, H, W, 3)
    grid = grid.unsqueeze(0).to(device)  # (1, D, H, W, 3)
    
    # Inference in chunks to avoid OOM
    chunk_size = 64
    refined_mask = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    
    with torch.no_grad():
        indices = torch.tensor([sample_idx], device=device)
        
        # Process in slices along Z axis
        for z_start in range(0, resolution, chunk_size):
            z_end = min(z_start + chunk_size, resolution)
            grid_chunk = grid[:, z_start:z_end, :, :, :]  # (1, chunk, H, W, 3)
            
            pred_logits, _ = model(indices, grid_chunk)
            pred_probs = torch.sigmoid(pred_logits).cpu().numpy()
            
            refined_mask[z_start:z_end, :, :] = pred_probs[0, 0]
    
    # Binarize
    refined_mask = (refined_mask > 0.5).astype(np.uint8)
    
    print(f"Inference complete. Positive voxels: {refined_mask.sum()}")
    return refined_mask


def analyze_connected_components(mask, min_size=10):
    """
    Analyze connected components in a binary mask.
    
    Args:
        mask: binary mask (D, H, W)
        min_size: minimum size (voxels) to count as a component
    
    Returns:
        num_components: number of connected components
        component_sizes: list of component sizes (sorted, largest first)
    """
    labeled_mask, num_features = connected_components(mask)
    
    # Count sizes
    component_sizes = []
    for i in range(1, num_features + 1):
        size = (labeled_mask == i).sum()
        if size >= min_size:
            component_sizes.append(size)
    
    component_sizes = sorted(component_sizes, reverse=True)
    num_components = len(component_sizes)
    
    return num_components, component_sizes


def dice_coefficient(mask1, mask2):
    """Calculate Dice coefficient between two binary masks."""
    intersection = (mask1 * mask2).sum()
    return 2.0 * intersection / (mask1.sum() + mask2.sum() + 1e-6)


def surface_dice_metric(mask_gt, mask_pred, spacing_mm=(1.0, 1.0, 1.0), tolerance_mm=1.0):
    """
    Calculate Surface Dice at specified tolerance.
    
    Args:
        mask_gt: ground truth binary mask (bool or 0/1)
        mask_pred: predicted binary mask (bool or 0/1)
        spacing_mm: voxel spacing in mm (z, y, x)
        tolerance_mm: tolerance for surface overlap in mm
    
    Returns:
        surface_dice: Surface Dice coefficient [0.0, 1.0]
        avg_surf_dist: tuple of (gt_to_pred, pred_to_gt) average surface distances
    """
    # Convert to bool if needed
    mask_gt_bool = mask_gt.astype(bool)
    mask_pred_bool = mask_pred.astype(bool)
    
    # Check if masks are empty
    if not mask_gt_bool.any() or not mask_pred_bool.any():
        print("  Warning: One or both masks are empty!")
        return 0.0, (float('inf'), float('inf'))
    
    # Compute surface distances
    surface_distances = compute_surface_distances(
        mask_gt_bool, mask_pred_bool, spacing_mm
    )
    
    # Calculate Surface Dice at tolerance
    surf_dice = compute_surface_dice_at_tolerance(surface_distances, tolerance_mm)
    
    # Calculate average surface distances
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


def visualize_comparison(ct_image, original_seg, refined_seg, sample_id, save_path):
    """
    Create visualization comparing original and refined segmentation.
    
    Args:
        ct_image: CT image (D, H, W) or None
        original_seg: original segmentation (D, H, W)
        refined_seg: refined segmentation (D, H, W)
        sample_id: sample ID for title
        save_path: path to save figure
    """
    # Select middle slices
    z_mid = original_seg.shape[0] // 2
    y_mid = original_seg.shape[1] // 2
    x_mid = original_seg.shape[2] // 2
    
    # Create figure
    if ct_image is not None:
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        titles = ['Axial (Z)', 'Coronal (Y)', 'Sagittal (X)']
        
        for i, (slice_idx, axis_name) in enumerate([(z_mid, 'Z'), (y_mid, 'Y'), (x_mid, 'X')]):
            # CT image
            if axis_name == 'Z':
                ct_slice = ct_image[slice_idx, :, :]
                orig_slice = original_seg[slice_idx, :, :]
                ref_slice = refined_seg[slice_idx, :, :]
            elif axis_name == 'Y':
                ct_slice = ct_image[:, slice_idx, :]
                orig_slice = original_seg[:, slice_idx, :]
                ref_slice = refined_seg[:, slice_idx, :]
            else:  # X
                ct_slice = ct_image[:, :, slice_idx]
                orig_slice = original_seg[:, :, slice_idx]
                ref_slice = refined_seg[:, :, slice_idx]
            
            # Row 1: CT with original overlay
            axes[0, i].imshow(ct_slice, cmap='gray')
            axes[0, i].imshow(orig_slice, cmap='Reds', alpha=0.3 * (orig_slice > 0))
            axes[0, i].set_title(f'{titles[i]} - CT + Original')
            axes[0, i].axis('off')
            
            # Row 2: CT with refined overlay
            axes[1, i].imshow(ct_slice, cmap='gray')
            axes[1, i].imshow(ref_slice, cmap='Greens', alpha=0.3 * (ref_slice > 0))
            axes[1, i].set_title(f'{titles[i]} - CT + Refined')
            axes[1, i].axis('off')
            
            # Row 3: Difference (added in green, removed in red)
            axes[2, i].imshow(ct_slice, cmap='gray')
            added = (ref_slice > 0) & (orig_slice == 0)
            removed = (orig_slice > 0) & (ref_slice == 0)
            axes[2, i].imshow(added, cmap='Greens', alpha=0.5 * added)
            axes[2, i].imshow(removed, cmap='Reds', alpha=0.5 * removed)
            axes[2, i].set_title(f'{titles[i]} - Difference')
            axes[2, i].axis('off')
    else:
        # Without CT image
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for i, (slice_idx, axis_name) in enumerate([(z_mid, 'Z'), (y_mid, 'Y'), (x_mid, 'X')]):
            if axis_name == 'Z':
                orig_slice = original_seg[slice_idx, :, :]
                ref_slice = refined_seg[slice_idx, :, :]
            elif axis_name == 'Y':
                orig_slice = original_seg[:, slice_idx, :]
                ref_slice = refined_seg[:, slice_idx, :]
            else:
                orig_slice = original_seg[:, :, slice_idx]
                ref_slice = refined_seg[:, :, slice_idx]
            
            # Original
            axes[0, i].imshow(orig_slice, cmap='gray')
            axes[0, i].set_title(f'Original - {axis_name} slice')
            axes[0, i].axis('off')
            
            # Refined
            axes[1, i].imshow(ref_slice, cmap='gray')
            axes[1, i].set_title(f'Refined - {axis_name} slice')
            axes[1, i].axis('off')
    
    plt.suptitle(f'Sample {sample_id} - Coronary Refinement Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Coronary refinement')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (best.pth)')
    parser.add_argument('--data_root', type=str, 
                        default='../../../dataset/near_format_data',
                        help='Path to dataset root')
    parser.add_argument('--sample_id', type=int, default=1,
                        help='Sample ID to evaluate (1-998)')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Inference resolution (128 or 256)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--class_index', type=int, default=9,
                        help='Class index (9 for Coronary)')
    parser.add_argument('--ct_image_dir', type=str, 
                        default='../../../dataset/near_format_data/images_256',
                        help='Directory containing resized CT images (256^3 .npy files)')
    parser.add_argument('--use_ct_overlay', action='store_true',
                        help='Overlay segmentation on CT image in visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = load_checkpoint(args.checkpoint, device)
    
    # Load original segmentation (256³, multi-class)
    shape_path = os.path.join(args.data_root, 'shape', f'{args.sample_id}.npy')
    print(f"\nLoading original segmentation from: {shape_path}")
    original_multi = np.load(shape_path)
    print(f"Original shape: {original_multi.shape}, unique classes: {np.unique(original_multi)}")
    
    # Extract Coronary class (class 9)
    original_seg = (original_multi == args.class_index).astype(np.uint8)
    print(f"Coronary voxels in original: {original_seg.sum()}")
    
    # Inference
    sample_idx = args.sample_id - 1  # Convert to 0-indexed
    refined_seg = inference_single_sample(model, sample_idx, args.resolution, device)
    
    # Resize if needed
    if refined_seg.shape != original_seg.shape:
        print(f"Resizing refined segmentation from {refined_seg.shape} to {original_seg.shape}")
        refined_seg_resized = resize(refined_seg, original_seg.shape, 
                                      order=0, preserve_range=True, anti_aliasing=False)
        refined_seg_resized = (refined_seg_resized > 0.5).astype(np.uint8)
    else:
        refined_seg_resized = refined_seg
    
    print(f"Coronary voxels in refined: {refined_seg_resized.sum()}")
    
    # === Analysis 1: Connected Components ===
    print("\n" + "="*70)
    print("Connected Components Analysis")
    print("="*70)
    
    num_cc_orig, sizes_orig = analyze_connected_components(original_seg)
    num_cc_refined, sizes_refined = analyze_connected_components(refined_seg_resized)
    
    print(f"\nOriginal Segmentation:")
    print(f"  Number of CCs: {num_cc_orig}")
    print(f"  Largest CC size: {sizes_orig[0] if sizes_orig else 0} voxels")
    print(f"  Top 5 CC sizes: {sizes_orig[:5]}")
    
    print(f"\nRefined Segmentation:")
    print(f"  Number of CCs: {num_cc_refined}")
    print(f"  Largest CC size: {sizes_refined[0] if sizes_refined else 0} voxels")
    print(f"  Top 5 CC sizes: {sizes_refined[:5]}")
    
    print(f"\nChange:")
    print(f"  CC reduction: {num_cc_orig - num_cc_refined} ({num_cc_orig} → {num_cc_refined})")
    
    # === Analysis 2: Dice Score (Volumetric) ===
    dice = dice_coefficient(original_seg, refined_seg_resized)
    print(f"\n" + "="*70)
    print(f"Volumetric Dice Coefficient: {dice:.4f}")
    print("="*70)
    
    # === Analysis 3: Surface Dice (更适合评估边界质量) ===
    print(f"\n" + "="*70)
    print("Surface Dice Analysis (Better for boundary quality)")
    print("="*70)
    
    # Voxel spacing (assume isotropic 1mm, adjust if needed)
    spacing_mm = (1.0, 1.0, 1.0)
    
    # Calculate Surface Dice at different tolerances
    tolerances = [0.5, 1.0, 2.0, 3.0]  # mm
    print(f"\nVoxel spacing: {spacing_mm} mm")
    print(f"\nSurface Dice at different tolerances:")
    
    for tol in tolerances:
        surf_dice, avg_dists = surface_dice_metric(
            original_seg, refined_seg_resized, 
            spacing_mm=spacing_mm, 
            tolerance_mm=tol
        )
        print(f"  Tolerance {tol:.1f}mm: Surface Dice = {surf_dice:.4f}")
    
    # Detailed analysis at 1mm tolerance (standard)
    surf_dice_1mm, (avg_gt_to_pred, avg_pred_to_gt) = surface_dice_metric(
        original_seg, refined_seg_resized, 
        spacing_mm=spacing_mm, 
        tolerance_mm=1.0
    )
    
    print(f"\nAverage Surface Distances (at 1mm tolerance):")
    print(f"  Original → Refined: {avg_gt_to_pred:.3f} mm")
    print(f"  Refined → Original: {avg_pred_to_gt:.3f} mm")
    print(f"  Mean: {(avg_gt_to_pred + avg_pred_to_gt) / 2:.3f} mm")
    
    # Interpretation
    print(f"\nInterpretation:")
    if surf_dice_1mm >= 0.90:
        print(f"  ✓ Excellent boundary preservation (Surface Dice ≥ 0.90)")
    elif surf_dice_1mm >= 0.80:
        print(f"  ✓ Good boundary quality (Surface Dice ≥ 0.80)")
    elif surf_dice_1mm >= 0.70:
        print(f"  ⚠ Fair boundary quality (Surface Dice ≥ 0.70)")
    else:
        print(f"  ✗ Poor boundary quality (Surface Dice < 0.70)")
    
    avg_dist = (avg_gt_to_pred + avg_pred_to_gt) / 2
    if avg_dist <= 0.5:
        print(f"  ✓ Excellent surface alignment (avg dist ≤ 0.5mm)")
    elif avg_dist <= 1.0:
        print(f"  ✓ Good surface alignment (avg dist ≤ 1.0mm)")
    elif avg_dist <= 2.0:
        print(f"  ⚠ Fair surface alignment (avg dist ≤ 2.0mm)")
    else:
        print(f"  ✗ Poor surface alignment (avg dist > 2.0mm)")
    
    print("="*70)
    print("="*70)
    
    # === Analysis 3: Volume/Voxel Ratio ===
    total_voxels = original_seg.size
    orig_positive = original_seg.sum()
    refined_positive = refined_seg_resized.sum()
    
    orig_ratio = orig_positive / total_voxels * 100
    refined_ratio = refined_positive / total_voxels * 100
    ratio_change = refined_ratio - orig_ratio
    ratio_change_pct = (refined_positive - orig_positive) / (orig_positive + 1e-6) * 100
    
    print(f"\n" + "="*70)
    print("Voxel Occupancy Analysis")
    print("="*70)
    print(f"Total voxels: {total_voxels:,}")
    print(f"\nOriginal Segmentation:")
    print(f"  Positive voxels: {orig_positive:,}")
    print(f"  Occupancy ratio: {orig_ratio:.4f}%")
    print(f"\nRefined Segmentation:")
    print(f"  Positive voxels: {refined_positive:,}")
    print(f"  Occupancy ratio: {refined_ratio:.4f}%")
    print(f"\nChange:")
    print(f"  Absolute change: {refined_positive - orig_positive:+,} voxels")
    print(f"  Ratio change: {ratio_change:+.4f}%")
    print(f"  Relative change: {ratio_change_pct:+.2f}%")
    
    if abs(ratio_change_pct) > 20:
        print(f"\n⚠️  Warning: Volume changed by more than 20%! Check refinement quality.")
    elif abs(ratio_change_pct) < 5:
        print(f"\n✓  Good: Volume change is minimal (<5%).")
    print("="*70)
    
    # === Visualization ===
    print(f"\nGenerating visualization...")
    
    # Load CT image if requested
    ct_image = None
    if args.use_ct_overlay:
        ct_path = os.path.join(args.ct_image_dir, f'{args.sample_id}.npy')
        if os.path.exists(ct_path):
            ct_image = np.load(ct_path)
            print(f"Loaded CT image from: {ct_path}")
        else:
            print(f"Warning: CT image not found at {ct_path}, using segmentation only")
    
    vis_path = os.path.join(args.output_dir, f'sample_{args.sample_id}_comparison.png')
    visualize_comparison(ct_image, original_seg, refined_seg_resized, args.sample_id, vis_path)
    
    # Save refined mask
    refined_path = os.path.join(args.output_dir, f'sample_{args.sample_id}_refined.npy')
    np.save(refined_path, refined_seg_resized)
    print(f"Refined mask saved to: {refined_path}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, f'sample_{args.sample_id}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Coronary Refinement Evaluation - Sample {args.sample_id}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Resolution: {args.resolution}³\n\n")
        
        f.write("Connected Components:\n")
        f.write(f"  Original: {num_cc_orig} CCs\n")
        f.write(f"  Refined:  {num_cc_refined} CCs\n")
        f.write(f"  Change:   {num_cc_orig - num_cc_refined}\n\n")
        
        f.write(f"Volumetric Dice Coefficient: {dice:.4f}\n\n")
        
        f.write(f"Surface Dice (at different tolerances):\n")
        for tol in tolerances:
            surf_dice_t, _ = surface_dice_metric(
                original_seg, refined_seg_resized, 
                spacing_mm=spacing_mm, 
                tolerance_mm=tol
            )
            f.write(f"  {tol:.1f}mm: {surf_dice_t:.4f}\n")
        
        f.write(f"\nAverage Surface Distances (1mm tolerance):\n")
        f.write(f"  Original → Refined: {avg_gt_to_pred:.3f} mm\n")
        f.write(f"  Refined → Original: {avg_pred_to_gt:.3f} mm\n")
        f.write(f"  Mean:               {avg_dist:.3f} mm\n\n")
        
        f.write(f"Voxel Counts:\n")
        f.write(f"  Original: {original_seg.sum():,}\n")
        f.write(f"  Refined:  {refined_seg_resized.sum():,}\n\n")
        
        f.write(f"Volume Occupancy:\n")
        f.write(f"  Original ratio: {orig_ratio:.4f}%\n")
        f.write(f"  Refined ratio:  {refined_ratio:.4f}%\n")
        f.write(f"  Change:         {ratio_change:+.4f}% ({ratio_change_pct:+.2f}%)\n")
    
    print(f"Summary saved to: {summary_path}")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
