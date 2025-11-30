"""
阶段一：冠状动脉推理脚本

功能说明：
1. 使用训练好的 Shape-only NeAR 模型生成精修后的冠状动脉掩膜
2. 输入：训练好的 checkpoint (F_c + z_i)
3. 输出：M^{ref}_{i,9} - 所有998个样本的精修冠状动脉二值掩膜
4. 推理策略：
   - 在目标分辨率（如256³）的网格上evaluate隐式函数
   - 使用阈值0.5将occupancy转为二值掩膜
   - 可选：CC清理（保留最大连通分量）
5. 保存格式：.npy文件，shape=(H,W,D)，dtype=uint8

Inference script for Coronary (class 9) refinement.
Generates M^{ref}_{i,9} (refined Coronary masks) for all 998 samples.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

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
    """Generate dense grid coordinates for full volume in format expected by model."""
    # Create coordinate grid [-1, 1] in each dimension
    d = torch.linspace(-1, 1, resolution)
    meshx, meshy, meshz = torch.meshgrid((d, d, d), indexing='ij')
    
    # Stack in (z, y, x) order to match grid_sample convention
    # Shape: (resolution, resolution, resolution, 3)
    grid = torch.stack((meshz, meshy, meshx), dim=-1)
    
    # Add batch dimension: (1, resolution, resolution, resolution, 3)
    grid = grid.unsqueeze(0)
    
    return grid


def inference_single_sample(model, sample_idx, resolution, batch_size=8192, chunk_size=128):
    """Generate refined mask for a single sample.
    
    Due to memory constraints, we process the volume in spatial chunks.
    Each chunk is of size (chunk_size, chunk_size, chunk_size).
    
    Args:
        model: The trained model
        sample_idx: Index of the sample
        resolution: Target resolution (e.g., 256)
        batch_size: Unused, kept for compatibility
        chunk_size: Size of spatial chunks to process (default: 128, balance memory/speed)
    """
    # Generate full volume grid
    full_grid = generate_grid_coordinates(resolution)  # (1, R, R, R, 3)
    full_grid = full_grid.squeeze(0)  # (R, R, R, 3)
    
    # Prepare sample index
    indices = to_var(torch.LongTensor([sample_idx]))
    
    # Initialize output volume and count for averaging
    output_volume = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    count_volume = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    
    # Process in spatial chunks without overlap (faster, slight quality loss at boundaries)
    stride = chunk_size  # No overlap for speed
    
    with torch.no_grad():
        for z_start in range(0, resolution, stride):
            for y_start in range(0, resolution, stride):
                for x_start in range(0, resolution, stride):
                    # Calculate chunk boundaries
                    z_end = min(z_start + chunk_size, resolution)
                    y_end = min(y_start + chunk_size, resolution)
                    x_end = min(x_start + chunk_size, resolution)
                    
                    # Extract chunk grid
                    chunk_grid = full_grid[z_start:z_end, 
                                         y_start:y_end, 
                                         x_start:x_end, :].unsqueeze(0)  # (1, D, H, W, 3)
                    chunk_grid = to_var(chunk_grid)
                    
                    # Forward pass
                    pred_logit, _ = model(indices, chunk_grid)  # (1, 1, D, H, W)
                    pred_prob = torch.sigmoid(pred_logit).squeeze().cpu().numpy()
                    
                    # Copy to output volume
                    actual_z_size = z_end - z_start
                    actual_y_size = y_end - y_start
                    actual_x_size = x_end - x_start
                    
                    output_volume[z_start:z_end, y_start:y_end, x_start:x_end] = \
                        pred_prob[:actual_z_size, :actual_y_size, :actual_x_size]
    
    return output_volume


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
    sample_ids = info_df['id'].values  # Column name is 'id', not 'sample_id'
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
