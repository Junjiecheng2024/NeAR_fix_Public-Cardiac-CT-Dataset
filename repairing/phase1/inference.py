#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_tier2.py
------------------
Run inference on trained Tier2 model to generate probability maps.
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import importlib.util

import torch
import torch.nn.functional as F

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from near.datasets.coronary_tier2_dataset import CoronaryTier2Dataset
from near.models.nn3d.model_shape_appearance import EmbeddingDecoderShapeAppearanceWithContext


def load_config(path):
    """Load config from Python file."""
    spec = importlib.util.spec_from_file_location("config", path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    return cfg_module.cfg


def create_full_grid(shape, device):
    """Create a full-resolution sampling grid."""
    d, h, w = shape
    # Create normalized grid [-1, 1]
    z = torch.linspace(-1, 1, d)
    y = torch.linspace(-1, 1, h)
    x = torch.linspace(-1, 1, w)
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (D, H, W, 3), xyz order for grid_sample
    return grid.unsqueeze(0).to(device)  # (1, D, H, W, 3)


def run_inference(
    model,
    dataset,
    device,
    output_dir,
    chunk_size=64,
    use_sliding_window=True
):
    """
    Run inference on all samples.
    
    Args:
        model: Trained model
        dataset: CoronaryTier2Dataset
        device: torch device
        output_dir: Output directory
        chunk_size: Chunk size for sliding window inference
        use_sliding_window: Whether to use sliding window for large volumes
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Inference"):
            batch = dataset[idx]
            case_id = batch['case_id']
            
            # Get data
            shape = batch['shape'].unsqueeze(0).to(device)  # (1, 1, D, H, W)
            appearance = batch['appearance'].unsqueeze(0).to(device)
            context = batch.get('context')
            if context is not None:
                context = context.unsqueeze(0).to(device)
            
            # Get indices
            indices = torch.tensor([idx], dtype=torch.long, device=device)
            
            # Get volume shape
            vol_shape = shape.shape[2:]  # (D, H, W)
            
            if not use_sliding_window or max(vol_shape) <= chunk_size:
                # Full volume inference
                grid = create_full_grid(vol_shape, device)
                pred_logit, _ = model(indices, grid, appearance, context)
                pred_prob = torch.sigmoid(pred_logit)
            else:
                # Sliding window inference for large volumes
                pred_prob = sliding_window_inference(
                    model, indices, appearance, context, vol_shape, 
                    chunk_size, device
                )
            
            # Save probability map
            pred_np = pred_prob.squeeze().cpu().numpy()
            
            case_output_dir = output_dir / case_id
            case_output_dir.mkdir(exist_ok=True)
            
            np.save(case_output_dir / "pred_prob.npy", pred_np.astype(np.float32))
            
            # Also save binary mask
            mask = (pred_np > 0.5).astype(np.uint8)
            np.save(case_output_dir / "pred_mask.npy", mask)
            
            # Save metadata
            metadata = {
                "case_id": case_id,
                "pred_shape": list(pred_np.shape),
                "positive_voxels": int(mask.sum()),
                "positive_ratio": float(mask.mean())
            }
            with open(case_output_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)


def sliding_window_inference(
    model, indices, appearance, context, vol_shape, 
    chunk_size, device, overlap=16
):
    """
    Sliding window inference for large volumes.
    """
    d, h, w = vol_shape
    
    # Initialize output
    pred_sum = torch.zeros((1, 1) + vol_shape, device=device)
    count = torch.zeros((1, 1) + vol_shape, device=device)
    
    # Calculate steps
    step = chunk_size - overlap
    
    for z_start in range(0, d, step):
        z_end = min(z_start + chunk_size, d)
        for y_start in range(0, h, step):
            y_end = min(y_start + chunk_size, h)
            for x_start in range(0, w, step):
                x_end = min(x_start + chunk_size, w)
                
                # Extract chunks
                app_chunk = appearance[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
                ctx_chunk = context[:, :, z_start:z_end, y_start:y_end, x_start:x_end] if context is not None else None
                
                chunk_shape = (z_end - z_start, y_end - y_start, x_end - x_start)
                grid = create_full_grid(chunk_shape, device)
                
                # Inference on chunk
                pred_logit, _ = model(indices, grid, app_chunk, ctx_chunk)
                pred_prob = torch.sigmoid(pred_logit)
                
                # Accumulate
                pred_sum[:, :, z_start:z_end, y_start:y_end, x_start:x_end] += pred_prob
                count[:, :, z_start:z_end, y_start:y_end, x_start:x_end] += 1
    
    # Average overlapping regions
    pred_avg = pred_sum / count.clamp(min=1)
    return pred_avg


def main():
    parser = argparse.ArgumentParser(description="NeAR v2.0 Tier2 Inference")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for sliding window")
    parser.add_argument("--no_sliding_window", action="store_true", help="Disable sliding window")
    parser.add_argument("--inference_resolution", type=int, default=None, 
                        help="Inference resolution (default: use config's target_resolution)")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    print(f"\n{'='*70}")
    print("NeAR v2.0 Tier2 Inference")
    print(f"{'='*70}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*70}\n")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine inference resolution
    inference_res = args.inference_resolution if args.inference_resolution else cfg.get('target_resolution')
    print(f"Inference resolution: {inference_res}³")
    
    # Load dataset (no augmentation for inference)
    dataset = CoronaryTier2Dataset(
        root=cfg['data_path'],
        resolution=inference_res,
        n_samples=cfg.get('n_training_samples'),
        use_appearance=cfg.get('use_appearance', True),
        boundary_bias_ratio=0.0,
        augment=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Build model
    model = EmbeddingDecoderShapeAppearanceWithContext(
        latent_dimension=cfg.get('latent_dimension', 256),
        n_samples=len(dataset),
        decoder_channels=cfg.get('decoder_channels', [64, 48, 32, 16]),
        appearance_channels=cfg.get('appearance_channels', 64),
        use_context=cfg.get('use_context', True)
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        # Lightning checkpoint
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() 
                     if k.startswith('model.')}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print("Model loaded successfully!")
    
    # Run inference
    run_inference(
        model=model,
        dataset=dataset,
        device=device,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        use_sliding_window=not args.no_sliding_window
    )
    
    print(f"\nInference complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
