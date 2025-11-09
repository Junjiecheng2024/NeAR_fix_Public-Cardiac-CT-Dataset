"""
Inference script for Coronary (class 9) refinement.
Generates M^{ref}_{i,9} (refined Coronary masks) for all 998 samples.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from near.models.nn3d.model_shape_only import EmbeddingDecoderShapeOnly
from near.utils.misc import to_device, to_var


def load_checkpoint(checkpoint_path, n_samples, latent_dimension, decoder_channels):
    """Load trained model from checkpoint."""
    model = to_device(
        EmbeddingDecoderShapeOnly(
            n_samples=n_samples,
            latent_dimension=latent_dimension,
            decoder_channels=decoder_channels
        )
    )
    
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def generate_grid_coordinates(resolution):
    """Generate dense grid coordinates for full volume."""
    # Create coordinate grid [-1, 1] in each dimension
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    z = torch.linspace(-1, 1, resolution)
    
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Shape: (resolution, resolution, resolution, 3)
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    
    return grid


def inference_single_sample(model, sample_idx, resolution, batch_size=8192):
    """Generate refined mask for a single sample."""
    # Generate grid coordinates
    grid = generate_grid_coordinates(resolution)  # (R, R, R, 3)
    
    # Flatten grid for batch processing
    grid_flat = grid.reshape(-1, 3)  # (R^3, 3)
    n_points = grid_flat.shape[0]
    
    # Prepare sample index
    indices = to_var(torch.LongTensor([sample_idx]))
    
    # Process in batches
    predictions = []
    
    with torch.no_grad():
        for start_idx in range(0, n_points, batch_size):
            end_idx = min(start_idx + batch_size, n_points)
            batch_grid = to_var(grid_flat[start_idx:end_idx].unsqueeze(0))  # (1, B, 3)
            
            # Forward pass
            pred_logit, _ = model(indices, batch_grid)  # (1, B, 1)
            pred_prob = torch.sigmoid(pred_logit)
            
            predictions.append(pred_prob.squeeze().cpu())
    
    # Concatenate and reshape
    predictions = torch.cat(predictions, dim=0)  # (R^3,)
    refined_mask = predictions.reshape(resolution, resolution, resolution).numpy()
    
    return refined_mask


def main():
    parser = argparse.ArgumentParser(description='Inference for Coronary refinement')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (best.pth or latest.pth)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for refined masks')
    parser.add_argument('--data_path', type=str,
                       default='../../../dataset/near_format_data',
                       help='Path to dataset')
    parser.add_argument('--class_name', type=str, default='Coronary',
                       help='Class name')
    parser.add_argument('--resolution', type=int, default=128,
                       help='Target resolution for refined masks')
    parser.add_argument('--latent_dimension', type=int, default=256,
                       help='Latent dimension')
    parser.add_argument('--decoder_channels', type=int, nargs='+',
                       default=[64, 48, 32, 16],
                       help='Decoder channels')
    parser.add_argument('--batch_size', type=int, default=8192,
                       help='Batch size for grid point processing')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary mask')
    parser.add_argument('--save_prob', action='store_true',
                       help='Save probability maps instead of binary masks')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load sample info
    info_path = os.path.join(args.data_path, 'info.csv')
    info_df = pd.read_csv(info_path)
    sample_ids = info_df['sample_id'].values
    n_samples = len(sample_ids)
    
    print(f"\n{'='*70}")
    print(f"Inference for {args.class_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Total samples: {n_samples}")
    print(f"Resolution: {args.resolution}")
    print(f"Threshold: {args.threshold}")
    print(f"{'='*70}\n")
    
    # Load model
    print("Loading model...")
    model = load_checkpoint(
        args.checkpoint,
        n_samples=n_samples,
        latent_dimension=args.latent_dimension,
        decoder_channels=args.decoder_channels
    )
    print("Model loaded successfully!\n")
    
    # Process each sample
    print("Generating refined masks...")
    for idx, sample_id in enumerate(tqdm(sample_ids)):
        # Generate refined mask
        refined_prob = inference_single_sample(
            model=model,
            sample_idx=idx,
            resolution=args.resolution,
            batch_size=args.batch_size
        )
        
        # Save result
        if args.save_prob:
            # Save probability map
            output_path = os.path.join(args.output_dir, f'{sample_id}_prob.npy')
            np.save(output_path, refined_prob.astype(np.float32))
        else:
            # Save binary mask
            refined_mask = (refined_prob > args.threshold).astype(np.uint8)
            output_path = os.path.join(args.output_dir, f'{sample_id}_refined.npy')
            np.save(output_path, refined_mask)
    
    print(f"\n{'='*70}")
    print(f"Inference completed!")
    print(f"Refined masks saved to: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
