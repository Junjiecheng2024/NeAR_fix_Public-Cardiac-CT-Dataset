#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_nifti.py
---------------
Export prediction and ground truth masks as NIfTI (.nii.gz) for viewing in 3D Slicer.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

import torch

try:
    import nibabel as nib
except ImportError:
    print("Please install nibabel: pip install nibabel")
    sys.exit(1)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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


def save_nifti(data, output_path, affine=None):
    """Save numpy array as NIfTI file."""
    if affine is None:
        # Default identity affine with 1mm spacing
        affine = np.eye(4)
    
    nii = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(nii, output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export segmentation masks as NIfTI")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, 
                       default="/scratch/project_2016517/JunjieCheng/dataset/coronary_tier2")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--case_id", type=str, default=None)
    parser.add_argument("--output_dir", type=str, 
                       default="/scratch/project_2016517/JunjieCheng/dataset/nifti_output")
    parser.add_argument("--resolution", type=int, default=128)
    
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
    
    print(f"Processing sample {sample_idx}: {dataset.case_dirs[sample_idx].name}")
    
    # Load data
    batch = dataset[sample_idx]
    case_id = batch['case_id']
    gt_mask = batch['shape'].squeeze().numpy()
    ct = batch['appearance'].squeeze().numpy()
    
    print(f"CT shape: {ct.shape}")
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
    pred_prob = get_prediction(model, dataset, sample_idx, device, resolution=args.resolution)
    pred_mask = (pred_prob > 0.5).astype(np.float32)
    print(f"Pred mask shape: {pred_mask.shape}, positive voxels: {pred_mask.sum()}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save NIfTI files
    print(f"\nExporting to: {output_dir}")
    
    # CT image
    save_nifti(ct, output_dir / f"{case_id}_ct.nii.gz")
    
    # Ground truth mask
    save_nifti(gt_mask, output_dir / f"{case_id}_gt.nii.gz")
    
    # Prediction probability
    save_nifti(pred_prob, output_dir / f"{case_id}_pred_prob.nii.gz")
    
    # Prediction binary mask
    save_nifti(pred_mask, output_dir / f"{case_id}_pred_mask.nii.gz")
    
    # Calculate Dice
    gt_binary = (gt_mask > 0.5).astype(np.float32)
    intersection = (gt_binary * pred_mask).sum()
    union = gt_binary.sum() + pred_mask.sum()
    dice = 2 * intersection / (union + 1e-8)
    
    print(f"\n{'='*50}")
    print(f"Case: {case_id}")
    print(f"Dice score: {dice:.4f}")
    print(f"GT voxels: {int(gt_binary.sum())}")
    print(f"Pred voxels: {int(pred_mask.sum())}")
    print(f"{'='*50}")
    print(f"\nFiles saved:")
    print(f"  - {case_id}_ct.nii.gz (CT image)")
    print(f"  - {case_id}_gt.nii.gz (Ground truth)")
    print(f"  - {case_id}_pred_mask.nii.gz (Prediction binary)")
    print(f"  - {case_id}_pred_prob.nii.gz (Prediction probability)")


if __name__ == "__main__":
    main()
