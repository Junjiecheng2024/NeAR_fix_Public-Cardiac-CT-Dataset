"""
Correct inference script for Coronary refinement.
Properly loads the model and performs inference at target resolution.
"""

import os
import sys
import numpy as np
import torch
import argparse
from tqdm import tqdm

# Add NeAR to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from near.models.nn3d.model_shape_only import EmbeddingDecoderShapeOnly
from near.datasets.cardiac_dataset import CardiacClassDatasetWithBiasedSampling


def load_model_and_data(checkpoint_path, data_root, class_index, device='cuda'):
    """Load trained model and dataset."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Model config (should match training config)
    model = EmbeddingDecoderShapeOnly(
        latent_dimension=256,
        decoder_channels=[64, 48, 32, 16]
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("✓ Model loaded successfully!")
    
    # Load dataset (to get proper sample indexing)
    dataset = CardiacClassDatasetWithBiasedSampling(
        root=data_root,
        class_name='Coronary',
        resolution=128,  # Dataset loads at this resolution
        n_samples=None,  # All samples
        sampling_bias_ratio=0.5,
        sampling_dilation_radius=2
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    return model, dataset


def inference_single_sample(model, dataset, sample_id, target_resolution=256, device='cuda'):
    """
    Perform inference for a single sample at specified resolution.
    
    Args:
        model: trained NeAR model
        dataset: CardiacClassDataset instance
        sample_id: 1-based sample ID (1-998)
        target_resolution: output resolution (128 or 256)
        device: cuda or cpu
    
    Returns:
        refined_mask: (D, H, W) binary mask at target_resolution
    """
    sample_idx = sample_id - 1  # Convert to 0-based
    print(f"\nInference for sample {sample_id} at resolution {target_resolution}³...")
    
    # Create dense query grid
    lin = torch.linspace(-1, 1, target_resolution, device=device)
    meshx, meshy, meshz = torch.meshgrid(lin, lin, lin, indexing='ij')
    grids = torch.stack([meshz, meshy, meshx], dim=-1)  # (D, H, W, 3)
    grids = grids.unsqueeze(0)  # (1, D, H, W, 3)
    
    # Process in chunks to avoid OOM
    chunk_size = 32  # Process 32 slices at a time
    refined_mask = np.zeros((target_resolution, target_resolution, target_resolution), dtype=np.float32)
    
    indices = torch.tensor([sample_idx], dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Process in slices along Z axis
        num_chunks = (target_resolution + chunk_size - 1) // chunk_size
        for i in tqdm(range(num_chunks), desc="Inference"):
            z_start = i * chunk_size
            z_end = min(z_start + chunk_size, target_resolution)
            
            grid_chunk = grids[:, z_start:z_end, :, :, :]  # (1, chunk, H, W, 3)
            
            # Model forward
            pred_logits, _ = model(indices, grid_chunk)  # (1, 1, chunk, H, W)
            pred_probs = torch.sigmoid(pred_logits)  # Apply sigmoid
            
            # Store results
            refined_mask[z_start:z_end, :, :] = pred_probs[0, 0].cpu().numpy()
    
    # Binarize with threshold 0.5
    refined_binary = (refined_mask > 0.5).astype(np.uint8)
    
    print(f"✓ Inference complete")
    print(f"  Positive voxels: {refined_binary.sum():,}")
    print(f"  Occupancy: {refined_binary.sum() / refined_binary.size * 100:.4f}%")
    
    return refined_binary


def main():
    parser = argparse.ArgumentParser(description='Coronary refinement inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best.pth checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset root (containing shape/ directory)')
    parser.add_argument('--sample_id', type=int, required=True,
                        help='Sample ID (1-998)')
    parser.add_argument('--resolution', type=int, default=256,
                        choices=[128, 256],
                        help='Output resolution')
    parser.add_argument('--class_index', type=int, default=9,
                        help='Class index (9 for Coronary)')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and dataset
    model, dataset = load_model_and_data(
        args.checkpoint,
        args.data_root,
        args.class_index,
        args.device
    )
    
    # Load original segmentation for comparison
    shape_path = os.path.join(args.data_root, 'shape', f'{args.sample_id}.npy')
    print(f"\nLoading original segmentation: {shape_path}")
    original_multi = np.load(shape_path)
    original_coronary = (original_multi == args.class_index).astype(np.uint8)
    print(f"  Original Coronary voxels: {original_coronary.sum():,}")
    
    # Perform inference
    refined_mask = inference_single_sample(
        model, dataset, args.sample_id,
        target_resolution=args.resolution,
        device=args.device
    )
    
    # Save results
    output_path = os.path.join(args.output_dir, f'sample_{args.sample_id}_refined.npy')
    np.save(output_path, refined_mask)
    print(f"\n✓ Refined mask saved to: {output_path}")
    
    # Quick comparison
    print(f"\nQuick Comparison:")
    print(f"  Original:  {original_coronary.sum():,} voxels")
    print(f"  Refined:   {refined_mask.sum():,} voxels")
    print(f"  Change:    {refined_mask.sum() - original_coronary.sum():+,} voxels")
    
    # Calculate Dice (resize if needed)
    if refined_mask.shape != original_coronary.shape:
        from skimage.transform import resize
        refined_resized = resize(refined_mask, original_coronary.shape, 
                                  order=0, preserve_range=True, anti_aliasing=False)
        refined_resized = (refined_resized > 0.5).astype(np.uint8)
    else:
        refined_resized = refined_mask
    
    intersection = (original_coronary * refined_resized).sum()
    dice = 2.0 * intersection / (original_coronary.sum() + refined_resized.sum() + 1e-6)
    print(f"  Volumetric Dice: {dice:.4f}")
    
    print("\n" + "="*70)
    print("Inference complete! Now run evaluate_coronary.py for detailed analysis.")
    print("="*70)


if __name__ == '__main__':
    main()
