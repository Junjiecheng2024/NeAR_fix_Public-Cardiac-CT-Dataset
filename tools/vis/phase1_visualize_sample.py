#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_sample.py
-------------------
Visualize a single sample comparing:
  1. Original CT image (cropped)
  2. Ground truth segmentation overlay
  3. Predicted segmentation overlay
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

REPO_ROOT = Path(project_root)
DEFAULT_DATA_ROOT = Path(os.environ.get("NEAR_DATA_ROOT", REPO_ROOT / "dataset"))

from near.datasets.coronary_tier2_dataset import CoronaryTier2Dataset
from near.models.nn3d.model_shape_appearance import EmbeddingDecoderShapeAppearanceWithContext


def create_full_grid(shape, device):
    """Create a full-resolution sampling grid."""
    d, h, w = shape
    z = torch.linspace(-1, 1, d)
    y = torch.linspace(-1, 1, h)
    x = torch.linspace(-1, 1, w)
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    return grid.unsqueeze(0).to(device)


def get_prediction(model, dataset, index, device, resolution=128):
    """Run inference on a single sample and return prediction."""
    model.eval()
    
    batch = dataset[index]
    
    # Get data
    appearance = batch['appearance'].unsqueeze(0).to(device)  # (1, 1, D, H, W)
    context = batch.get('context')
    if context is not None:
        context = context.unsqueeze(0).to(device)
    
    indices = torch.tensor([index], dtype=torch.long, device=device)
    
    # Create grid for target resolution
    grid = create_full_grid((resolution, resolution, resolution), device)
    
    with torch.no_grad():
        pred_logit, _ = model(indices, grid, appearance, context)
        pred_prob = torch.sigmoid(pred_logit)
    
    return pred_prob.squeeze().cpu().numpy()


def create_overlay_image(ct_slice, mask_slice, alpha=0.4, color='red'):
    """Create an overlay of mask on CT slice."""
    # Normalize CT to 0-1
    ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
    
    # Create RGB image from grayscale CT
    rgb = np.stack([ct_norm, ct_norm, ct_norm], axis=-1)
    
    # Create colored mask overlay
    color_rgb = mcolors.to_rgb(color)
    mask_binary = (mask_slice > 0.5).astype(np.float32)
    
    for i, c in enumerate(color_rgb):
        rgb[:, :, i] = rgb[:, :, i] * (1 - alpha * mask_binary) + c * alpha * mask_binary
    
    return np.clip(rgb, 0, 1)


def visualize_sample(
    ct,           # (D, H, W)
    gt_mask,      # (D, H, W)
    pred_mask,    # (D, H, W)
    slice_indices=None,
    output_path=None,
    case_id=None
):
    """
    Create a 3x3 or 3x5 comparison figure.
    
    Rows: Different slices (axial views at different depths)
    Cols: 1. CT only, 2. CT + GT overlay, 3. CT + Pred overlay
    """
    # Determine slice indices
    d = ct.shape[0]
    if slice_indices is None:
        # Pick 3 representative slices
        slice_indices = [d // 4, d // 2, 3 * d // 4]
    
    n_slices = len(slice_indices)
    fig, axes = plt.subplots(n_slices, 3, figsize=(12, 4 * n_slices))
    
    if n_slices == 1:
        axes = axes.reshape(1, -1)
    
    for row, z_idx in enumerate(slice_indices):
        z_idx = min(z_idx, d - 1)  # Clamp to valid range
        
        ct_slice = ct[z_idx]
        gt_slice = gt_mask[z_idx]
        pred_slice = pred_mask[z_idx]
        
        # Column 1: CT only
        ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
        axes[row, 0].imshow(ct_norm, cmap='gray')
        axes[row, 0].set_title(f'CT (z={z_idx})')
        axes[row, 0].axis('off')
        
        # Column 2: CT + GT overlay (green)
        gt_overlay = create_overlay_image(ct_slice, gt_slice, alpha=0.5, color='lime')
        axes[row, 1].imshow(gt_overlay)
        axes[row, 1].set_title(f'Ground Truth (z={z_idx})')
        axes[row, 1].axis('off')
        
        # Column 3: CT + Pred overlay (red)
        pred_overlay = create_overlay_image(ct_slice, pred_slice, alpha=0.5, color='red')
        axes[row, 2].imshow(pred_overlay)
        axes[row, 2].set_title(f'Prediction (z={z_idx})')
        axes[row, 2].axis('off')
    
    # Calculate metrics
    gt_binary = (gt_mask > 0.5).astype(np.float32)
    pred_binary = (pred_mask > 0.5).astype(np.float32)
    
    intersection = (gt_binary * pred_binary).sum()
    union = gt_binary.sum() + pred_binary.sum()
    dice = 2 * intersection / (union + 1e-8)
    
    title = f"Case: {case_id}" if case_id else "Sample Comparison"
    title += f"  |  Dice: {dice:.4f}  |  GT voxels: {int(gt_binary.sum())}  |  Pred voxels: {int(pred_binary.sum())}"
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return dice


def main():
    parser = argparse.ArgumentParser(description="Visualize single-class Phase1 segmentation results")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--data_path", type=str, 
                       default=str(DEFAULT_DATA_ROOT / "coronary_tier2"),
                       help="Dataset path")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to visualize")
    parser.add_argument("--case_id", type=str, default=None, help="Case ID to visualize (overrides sample_idx)")
    parser.add_argument("--output", type=str, default=None, help="Optional output image path")
    parser.add_argument("--slices", type=int, nargs='+', default=None, help="Slice indices to visualize")
    parser.add_argument("--resolution", type=int, default=128, help="Inference resolution")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    dataset = CoronaryTier2Dataset(
        root=args.data_path,
        resolution=args.resolution,
        use_appearance=True,
        boundary_bias_ratio=0.0,
        augment=False
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Find sample index by case_id if specified
    sample_idx = args.sample_idx
    if args.case_id:
        for i, case_dir in enumerate(dataset.case_dirs):
            if case_dir.name == args.case_id:
                sample_idx = i
                break
        else:
            print(f"Error: Case ID '{args.case_id}' not found!")
            print(f"Available cases: {[d.name for d in dataset.case_dirs[:10]]}...")
            return
    
    print(f"Visualizing sample {sample_idx}: {dataset.case_dirs[sample_idx].name}")
    
    # Load sample
    batch = dataset[sample_idx]
    case_id = batch['case_id']
    
    # Get ground truth and CT from batch
    gt_mask = batch['shape'].squeeze().numpy()  # (D, H, W)
    ct = batch['appearance'].squeeze().numpy()  # (D, H, W)
    
    print(f"CT shape: {ct.shape}")
    print(f"GT mask shape: {gt_mask.shape}, positive voxels: {(gt_mask > 0.5).sum()}")
    
    # Build model
    model = EmbeddingDecoderShapeAppearanceWithContext(
        latent_dimension=256,
        n_samples=len(dataset),
        decoder_channels=[64, 48, 32, 16],
        appearance_channels=64,
        use_context=True
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() 
                     if k.startswith('model.')}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print("Model loaded successfully!")
    
    # Get prediction
    print("Running inference...")
    pred_mask = get_prediction(model, dataset, sample_idx, device, resolution=args.resolution)
    print(f"Pred mask shape: {pred_mask.shape}, positive voxels: {(pred_mask > 0.5).sum()}")
    
    # Determine output path
    output_path = args.output
    if output_path is None:
        output_dir = Path(args.checkpoint).parent
        output_path = output_dir / f"vis_{case_id}.png"
    
    # Visualize
    dice = visualize_sample(
        ct=ct,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        slice_indices=args.slices,
        output_path=str(output_path),
        case_id=case_id
    )
    
    print(f"\nDice score: {dice:.4f}")


if __name__ == "__main__":
    main()
