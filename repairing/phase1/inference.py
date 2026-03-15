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
from scipy.ndimage import zoom

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from near.datasets.coronary_tier2_dataset import CoronaryTier2Dataset
from near.models.nn3d.model_shape_appearance import EmbeddingDecoderShapeAppearanceWithContext


def load_config(path, class_name=None):
    """Load config from Python file."""
    spec = importlib.util.spec_from_file_location("config", path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)

    if class_name is not None and hasattr(cfg_module, "get_config"):
        return cfg_module.get_config(class_name).to_dict()

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


def create_global_chunk_grid(chunk_shape, start, end, vol_shape, device):
    """
    Create a sampling grid for a chunk with GLOBAL normalized coordinates.
    
    This is critical for sliding window inference: each chunk's grid coordinates
    should map to their actual position in the full volume's [-1, 1] space,
    NOT be locally normalized to [-1, 1] within the chunk.
    
    Args:
        chunk_shape: (d, h, w) shape of the chunk
        start: (z_start, y_start, x_start) starting indices in full volume
        end: (z_end, y_end, x_end) ending indices in full volume
        vol_shape: (D, H, W) shape of the full volume
        device: torch device
    
    Returns:
        grid: (1, d, h, w, 3) sampling grid with global normalized coordinates
    """
    d, h, w = chunk_shape
    z_start, y_start, x_start = start
    z_end, y_end, x_end = end
    D, H, W = vol_shape
    
    # Convert voxel indices to global normalized coordinates [-1, 1]
    # For a volume of size D, voxel i should map to: 2 * i / (D - 1) - 1
    # But linspace is more elegant: create coords for the chunk's range in global space
    
    # Z coordinates: from z_start to z_end-1 in global [0, D-1] → [-1, 1]
    z = torch.linspace(2 * z_start / (D - 1) - 1, 2 * (z_end - 1) / (D - 1) - 1, d)
    y = torch.linspace(2 * y_start / (H - 1) - 1, 2 * (y_end - 1) / (H - 1) - 1, h)
    x = torch.linspace(2 * x_start / (W - 1) - 1, 2 * (x_end - 1) / (W - 1) - 1, w)
    
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (d, h, w, 3), xyz order
    return grid.unsqueeze(0).to(device)  # (1, d, h, w, 3)


def map_to_global(
    pred_crop: np.ndarray, 
    crop_params: dict, 
    global_shape: tuple = None
) -> np.ndarray:
    """
    Map cropped prediction back to global coordinate space.
    If global_shape is None, uses crop_params['original_shape'].
    If global_shape is provided and differs from original_shape, 
    first reconstructs in original_shape then resizes.
    """
    origin = np.array(crop_params['origin'])  # [z, y, x] in original space
    original_crop_size = np.array(crop_params['size'])  # [d, h, w] before resize
    original_full_shape = tuple(crop_params['original_shape'])
    
    # Check if resize was applied during preprocessing (from orig crop to model input)
    if crop_params.get('resize_applied', False):
        zoom_factors = original_crop_size / np.array(pred_crop.shape)
        pred_original_size = zoom(pred_crop.astype(np.float32), zoom_factors, order=0)
        pred_original_size = (pred_original_size > 0.5).astype(np.uint8)
    else:
        pred_original_size = pred_crop
    
    # Reconstruct in ORIGINAL full resolution first
    pred_global = np.zeros(original_full_shape, dtype=np.uint8)
    
    # Compute end coordinates
    end = origin + np.array(pred_original_size.shape)
    
    # Clip to global bounds (original full shape)
    valid_start = np.maximum(origin, 0).astype(int)
    valid_end = np.minimum(end, np.array(original_full_shape)).astype(int)
    
    # Ensure valid range
    valid_end = np.maximum(valid_end, valid_start)
    
    if np.any(valid_end - valid_start <= 0):
        # Crop is completely outside
        return pred_global
    
    # Compute offsets in crop space
    crop_start = (valid_start - origin).astype(int)
    crop_end = (crop_start + (valid_end - valid_start)).astype(int)
    
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
    
    # If a target global_shape is requested (e.g. 256^3) and differs from original
    if global_shape is not None and tuple(global_shape) != original_full_shape:
        factors = np.array(global_shape) / np.array(original_full_shape)
        # Use order=0 for nearest neighbor (binary mask)
        pred_global = zoom(pred_global, factors, order=0)
        
    return pred_global


def run_inference(
    model,
    dataset,
    device,
    output_dir,
    chunk_size=64,
    use_sliding_window=True,
    global_shape=256
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
            
            # Convert to numpy and threshold
            pred_np = pred_prob.squeeze().cpu().numpy()
            mask_crop = (pred_np > 0.5).astype(np.uint8)
            
            # Get crop params for coordinate mapping
            crop_params = dataset.get_crop_params(case_id)
            
            if crop_params is not None:
                # Map to global space (Reconstruct in Original -> Resize to Target Global)
                mask_global = map_to_global(mask_crop, crop_params, (global_shape,)*3)
                
                # Save global mask (flat structure for Phase 2/3 compatibility)
                np.save(output_dir / f"{case_id}_mask.npy", mask_global)
                
                positive_voxels = int(mask_global.sum())
                positive_ratio = float(mask_global.mean())
            else:
                # No crop params, save crop-space mask
                case_output_dir = output_dir / case_id
                case_output_dir.mkdir(exist_ok=True)
                np.save(case_output_dir / "pred_mask.npy", mask_crop)
                positive_voxels = int(mask_crop.sum())
                positive_ratio = float(mask_crop.mean())


def sliding_window_inference(
    model, indices, appearance, context, vol_shape, 
    chunk_size, device, overlap=16
):
    """
    Sliding window inference for large volumes.
    
    IMPORTANT: Grid coordinates are computed in GLOBAL normalized space [-1, 1]
    to match the training setup where the entire volume is normalized.
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
                
                # FIX: Create grid with GLOBAL normalized coordinates
                # Instead of normalizing each chunk to [-1, 1], we compute
                # the chunk's position in the global [-1, 1] space
                grid = create_global_chunk_grid(
                    chunk_shape, 
                    (z_start, y_start, x_start),
                    (z_end, y_end, x_end),
                    vol_shape,
                    device
                )
                
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
    parser.add_argument("--class_name", type=str, default=None,
                        help="Optional class name to load from config.py via get_config()")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for sliding window")
    parser.add_argument("--no_sliding_window", action="store_true", help="Disable sliding window")
    parser.add_argument("--inference_resolution", type=int, default=None, 
                        help="Inference resolution (default: use config's target_resolution)")
    parser.add_argument("--global_shape", type=int, default=256,
                        help="Global output shape for coordinate mapping (default: 256)")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config, args.class_name)
    
    print(f"\n{'='*70}")
    print("NeAR v2.0 Tier2 Inference")
    print(f"{'='*70}")
    print(f"Config: {args.config}")
    if args.class_name is not None:
        print(f"Class override: {args.class_name}")
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
    
    # Run inference with coordinate mapping
    print(f"Output global shape: {args.global_shape}³")
    run_inference(
        model=model,
        dataset=dataset,
        device=device,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        use_sliding_window=not args.no_sliding_window,
        global_shape=args.global_shape
    )
    
    print(f"\nInference complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
