#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_3d.py
---------------
Generate 3D surface visualizations comparing:
  1. Ground truth segmentation (green)
  2. Predicted segmentation (red)
  
Uses marching cubes to extract isosurfaces and renders them side by side.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from skimage import measure
except ImportError:
    print("Please install scikit-image: pip install scikit-image")
    sys.exit(1)

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
    """Run inference on a single sample."""
    model.eval()
    batch = dataset[index]
    
    appearance = batch['appearance'].unsqueeze(0).to(device)
    context = batch.get('context')
    if context is not None:
        context = context.unsqueeze(0).to(device)
    
    indices = torch.tensor([index], dtype=torch.long, device=device)
    grid = create_full_grid((resolution, resolution, resolution), device)
    
    with torch.no_grad():
        pred_logit, _ = model(indices, grid, appearance, context)
        pred_prob = torch.sigmoid(pred_logit)
    
    return pred_prob.squeeze().cpu().numpy()


def plot_3d_mesh(ax, mask, color='green', alpha=0.7, title=""):
    """Plot a 3D surface mesh from binary mask."""
    # Marching cubes to extract surface
    if mask.sum() < 10:
        ax.set_title(f"{title}\n(Empty: {int(mask.sum())} voxels)")
        return
    
    try:
        # Pad to avoid edge artifacts
        mask_padded = np.pad(mask, 1, mode='constant', constant_values=0)
        verts, faces, _, _ = measure.marching_cubes(mask_padded, level=0.5)
        # Adjust for padding
        verts = verts - 1
        
        # Create mesh
        mesh = Poly3DCollection(verts[faces], alpha=alpha)
        mesh.set_facecolor(color)
        mesh.set_edgecolor('none')
        ax.add_collection3d(mesh)
        
        # Set limits
        ax.set_xlim(0, mask.shape[2])
        ax.set_ylim(0, mask.shape[1])
        ax.set_zlim(0, mask.shape[0])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    except Exception as e:
        print(f"Warning: Could not create mesh for {title}: {e}")
    
    n_voxels = int(mask.sum())
    ax.set_title(f"{title}\n({n_voxels} voxels)")


def visualize_3d(gt_mask, pred_mask, output_path=None, case_id=None,
                 elev=30, azim=45):
    """
    Create side-by-side 3D visualization.
    """
    # Binarize
    gt_binary = (gt_mask > 0.5).astype(np.float32)
    pred_binary = (pred_mask > 0.5).astype(np.float32)
    
    # Calculate Dice
    intersection = (gt_binary * pred_binary).sum()
    union = gt_binary.sum() + pred_binary.sum()
    dice = 2 * intersection / (union + 1e-8)
    
    # Create figure with two 3D subplots
    fig = plt.figure(figsize=(16, 7))
    
    # Ground truth
    ax1 = fig.add_subplot(121, projection='3d')
    plot_3d_mesh(ax1, gt_binary, color='limegreen', alpha=0.6, title="Ground Truth")
    ax1.view_init(elev=elev, azim=azim)
    
    # Prediction
    ax2 = fig.add_subplot(122, projection='3d')
    plot_3d_mesh(ax2, pred_binary, color='red', alpha=0.6, title="Prediction")
    ax2.view_init(elev=elev, azim=azim)
    
    # Title
    title = f"Case: {case_id}" if case_id else "3D Comparison"
    title += f"  |  Dice: {dice:.4f}"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    return dice


def visualize_3d_overlay(gt_mask, pred_mask, output_path=None, case_id=None,
                         elev=30, azim=45):
    """
    Create overlaid 3D visualization (GT green, Pred red, overlap yellow).
    """
    gt_binary = (gt_mask > 0.5).astype(np.float32)
    pred_binary = (pred_mask > 0.5).astype(np.float32)
    
    # Calculate regions
    overlap = gt_binary * pred_binary  # Both
    gt_only = gt_binary * (1 - pred_binary)  # GT only (FN)
    pred_only = pred_binary * (1 - gt_binary)  # Pred only (FP)
    
    # Calculate Dice
    intersection = overlap.sum()
    union = gt_binary.sum() + pred_binary.sum()
    dice = 2 * intersection / (union + 1e-8)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each region with different colors
    if overlap.sum() > 10:
        plot_3d_mesh(ax, overlap, color='yellow', alpha=0.8, title="")
    if gt_only.sum() > 10:
        try:
            mask_padded = np.pad(gt_only, 1, mode='constant', constant_values=0)
            verts, faces, _, _ = measure.marching_cubes(mask_padded, level=0.5)
            verts = verts - 1
            mesh = Poly3DCollection(verts[faces], alpha=0.5)
            mesh.set_facecolor('green')
            mesh.set_edgecolor('none')
            ax.add_collection3d(mesh)
        except:
            pass
    if pred_only.sum() > 10:
        try:
            mask_padded = np.pad(pred_only, 1, mode='constant', constant_values=0)
            verts, faces, _, _ = measure.marching_cubes(mask_padded, level=0.5)
            verts = verts - 1
            mesh = Poly3DCollection(verts[faces], alpha=0.5)
            mesh.set_facecolor('red')
            mesh.set_edgecolor('none')
            ax.add_collection3d(mesh)
        except:
            pass
    
    # Set limits
    ax.set_xlim(0, gt_mask.shape[2])
    ax.set_ylim(0, gt_mask.shape[1])
    ax.set_zlim(0, gt_mask.shape[0])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='yellow', label=f'Overlap (TP): {int(overlap.sum())}'),
        Patch(facecolor='green', label=f'GT only (FN): {int(gt_only.sum())}'),
        Patch(facecolor='red', label=f'Pred only (FP): {int(pred_only.sum())}'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    title = f"Case: {case_id}" if case_id else "3D Overlay"
    title += f"  |  Dice: {dice:.4f}"
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
    parser = argparse.ArgumentParser(description="3D visualization of single-class Phase1 segmentation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, 
                       default=str(DEFAULT_DATA_ROOT / "coronary_tier2"))
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--case_id", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Optional output image path")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--elev", type=float, default=30, help="Elevation angle")
    parser.add_argument("--azim", type=float, default=45, help="Azimuth angle")
    parser.add_argument("--overlay", action="store_true", help="Create overlay visualization instead")
    
    args = parser.parse_args()
    
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
    
    # Find sample
    sample_idx = args.sample_idx
    if args.case_id:
        for i, case_dir in enumerate(dataset.case_dirs):
            if case_dir.name == args.case_id:
                sample_idx = i
                break
        else:
            print(f"Error: Case ID '{args.case_id}' not found!")
            return
    
    print(f"Visualizing sample {sample_idx}: {dataset.case_dirs[sample_idx].name}")
    
    # Load data
    batch = dataset[sample_idx]
    case_id = batch['case_id']
    gt_mask = batch['shape'].squeeze().numpy()
    
    print(f"GT mask shape: {gt_mask.shape}, positive voxels: {(gt_mask > 0.5).sum()}")
    
    # Build and load model
    model = EmbeddingDecoderShapeAppearanceWithContext(
        latent_dimension=256,
        n_samples=len(dataset),
        decoder_channels=[64, 48, 32, 16],
        appearance_channels=64,
        use_context=True
    )
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() 
                     if k.startswith('model.')}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print("Model loaded!")
    
    # Get prediction
    print("Running inference...")
    pred_mask = get_prediction(model, dataset, sample_idx, device, resolution=args.resolution)
    print(f"Pred mask shape: {pred_mask.shape}, positive voxels: {(pred_mask > 0.5).sum()}")
    
    # Output path
    output_path = args.output
    if output_path and os.path.isdir(output_path):
        suffix = "_overlay" if args.overlay else "_3d"
        output_path = os.path.join(output_path, f"vis_{case_id}{suffix}.png")
    
    # Visualize
    if args.overlay:
        dice = visualize_3d_overlay(
            gt_mask, pred_mask, output_path=output_path, case_id=case_id,
            elev=args.elev, azim=args.azim
        )
    else:
        dice = visualize_3d(
            gt_mask, pred_mask, output_path=output_path, case_id=case_id,
            elev=args.elev, azim=args.azim
        )
    
    print(f"\nDice score: {dice:.4f}")


if __name__ == "__main__":
    main()
