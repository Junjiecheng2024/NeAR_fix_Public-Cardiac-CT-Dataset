"""
Integrated Inference and Evaluation Script for RV Refinement (Stage 1)

This script performs:
1. Inference: Generates 256-resolution masks using the trained implicit model.
2. Evaluation: Calculates Voxel Ratio and Connected Components (CC).
3. Visualization: Compares Original Image (if available) vs Original Mask vs Refined Mask.

Usage:
    python inference_and_evaluate.py \
        --checkpoint lightning_logs/version_X/checkpoints/best.pth \
        --output_dir results_256 \
        --resolution 256 \
        --data_path ../../../dataset/near_format_data
"""

import sys
import os
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import label
from collections import OrderedDict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from near.models.nn3d.model_shape_only import EmbeddingDecoderShapeOnly
from near.utils.misc import to_device, to_var

# -----------------------------------------------------------------------------
# Model & Inference Functions
# -----------------------------------------------------------------------------

def load_checkpoint(checkpoint_path, n_samples, latent_dimension, decoder_channels):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    model = to_device(
        EmbeddingDecoderShapeOnly(
            n_samples=n_samples,
            latent_dimension=latent_dimension,
            decoder_channels=decoder_channels
        )
    )
    
    # Load state dict
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Handle 'model.' prefix if saved from LightningModule
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('model.'):
                name = k[6:] # remove 'model.'
            elif k.startswith('module.'):
                name = k[7:] # remove 'module.'
            else:
                name = k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
        
    model.eval()
    return model

def generate_grid_coordinates(resolution):
    """Generate dense grid coordinates for full volume [-1, 1]."""
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    z = torch.linspace(-1, 1, resolution)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid

def inference_single_sample(model, sample_idx, resolution, batch_size=8192, verbose=True):
    """Generate refined mask for a single sample."""
    grid = generate_grid_coordinates(resolution)
    grid_flat = grid.reshape(-1, 3)
    n_points = grid_flat.shape[0]
    indices = to_var(torch.LongTensor([sample_idx]))
    
    predictions = []
    
    iterator = range(0, n_points, batch_size)
    if verbose:
        iterator = tqdm(iterator, desc=f"Inferring (Res {resolution})", leave=False)
        
    with torch.no_grad():
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, n_points)
            # Reshape grid to (1, B, 1, 1, 3) to match 5D expectation of grid_sample
            # The model expects (B, D, H, W, 3)
            # Here we treat the batch of points as a "line" or "volume" of shape (B, 1, 1)
            batch_grid = to_var(grid_flat[start_idx:end_idx].unsqueeze(0).unsqueeze(2).unsqueeze(3))
            
            pred_logit, _ = model(indices, batch_grid)
            pred_prob = torch.sigmoid(pred_logit)
            predictions.append(pred_prob.squeeze().cpu())
            
    predictions = torch.cat(predictions, dim=0)
    refined_mask = predictions.reshape(resolution, resolution, resolution).numpy()
    return refined_mask

# -----------------------------------------------------------------------------
# Evaluation Functions
# -----------------------------------------------------------------------------

def compute_connected_components(mask):
    """Count number of connected components."""
    labeled, n_components = label(mask)
    return n_components

def compute_metrics(original_mask, refined_mask):
    """Compute CC and Voxel Ratio (accounting for resolution differences)."""
    metrics = {}
    
    # Connected Components
    metrics['original_cc'] = compute_connected_components(original_mask)
    metrics['refined_cc'] = compute_connected_components(refined_mask)
    
    # Voxel Ratio (Volume Preservation)
    # We must account for resolution differences.
    # Assuming both masks represent the same physical volume [-1, 1]^3
    # Volume = Count * (Voxel_Size)^3
    # Voxel_Size = 2.0 / Resolution
    
    res_orig = original_mask.shape[0]
    res_ref = refined_mask.shape[0]
    
    # If resolutions are the same (e.g. both 256), this simplifies to count ratio
    vol_orig = np.sum(original_mask) * (2.0 / res_orig)**3
    vol_ref = np.sum(refined_mask) * (2.0 / res_ref)**3
    
    if vol_orig > 0:
        metrics['voxel_ratio'] = vol_ref / vol_orig
    else:
        metrics['voxel_ratio'] = 0.0 if vol_ref == 0 else float('inf')
        
    return metrics

# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------

def visualize_comparison(original_mask, refined_mask, original_image=None, sample_id="sample", output_path=None):
    """
    Visualize: Original Image vs Original Mask (Overlay) vs Refined Mask (Overlay).
    Shows the middle slice of the Z-axis (axis 2).
    """
    # Use the resolution of the refined mask (usually 256) as the target display resolution
    target_res = refined_mask.shape[0]
    slice_idx = target_res // 2
    
    # Prepare Refined Slice (256)
    # Assuming (D, H, W), slice along D (axis 0) or Z (axis 2)?
    # Usually medical volumes are (D, H, W) = (Z, Y, X) or (X, Y, Z).
    # Let's try slicing axis 0 (Depth/Z) which is common for D,H,W
    # BUT previous code used axis 2. Let's stick to axis 0 if we assume (D, H, W).
    # Wait, previous code used axis 2. Let's check if that was the issue.
    # If data is (D, H, W), axis 0 is Z.
    # If data is (H, W, D), axis 2 is Z.
    
    # Let's try to be smart. We want the view that looks like a standard CT slice (axial).
    # Usually that's the first dimension if (D, H, W).
    
    axis_to_slice = 0 # Try axis 0 (Depth)
    
    ref_slice = refined_mask[slice_idx, :, :]
    
    # Prepare Original Mask Slice
    orig_res = original_mask.shape[0]
    orig_slice_idx = int(slice_idx * (orig_res / target_res))
    
    # Handle potential transpose for Original Mask
    # If Original Mask is (H, W, D) and we want (D, H, W) behavior
    # We might need to slice differently or transpose.
    # Let's try to visualize Original Mask with same slicing as Refined.
    orig_slice = original_mask[orig_slice_idx, :, :]
    
    # Resize orig_slice
    from skimage.transform import resize
    if orig_res != target_res:
        orig_slice_resized = resize(orig_slice, (target_res, target_res), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
    else:
        orig_slice_resized = orig_slice

    # Prepare Image Slice
    if original_image is not None:
        img_res = original_image.shape[0]
        if len(original_image.shape) == 3:
            img_slice_idx = int(slice_idx * (img_res / target_res))
            img_slice = original_image[img_slice_idx, :, :] # Slice axis 0
        elif len(original_image.shape) == 2:
            img_slice = original_image
            img_res = img_slice.shape[0]
        
        if img_res != target_res:
            img_slice = resize(img_slice, (target_res, target_res), order=1, preserve_range=True, anti_aliasing=True)
    else:
        img_slice = np.zeros((target_res, target_res))

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original CT Image
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title(f'Original CT Image\nSlice Axis {axis_to_slice}')
    axes[0].axis('off')
    
    # 2. Original Mask Overlay
    axes[1].imshow(img_slice, cmap='gray')
    overlay_orig = np.zeros((target_res, target_res, 4))
    overlay_orig[orig_slice_resized > 0] = [1, 0, 0, 0.5] 
    axes[1].imshow(overlay_orig)
    axes[1].set_title(f'Original Mask (Overlay)\nSum: {np.sum(original_mask)}')
    axes[1].axis('off')
    
    # 3. Refined Mask Overlay
    axes[2].imshow(img_slice, cmap='gray')
    overlay_ref = np.zeros((target_res, target_res, 4))
    overlay_ref[ref_slice > 0] = [0, 1, 0, 0.5] 
    axes[2].imshow(overlay_ref)
    axes[2].set_title(f'Refined Mask (Overlay)\nSum: {np.sum(refined_mask)}')
    axes[2].axis('off')
    
    plt.suptitle(f'Sample: {sample_id} | Slice: {slice_idx}', fontsize=16)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Saving visualization to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Inference, Evaluate and Visualize RV Refinement')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--data_path', type=str, default='../../../dataset/near_format_data', help='Path to dataset root')
    parser.add_argument('--resolution', type=int, default=256, help='Inference resolution')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binarization threshold')
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of patients/cases to process (for demo)')
    parser.add_argument('--target_class', type=int, default=5, help='Target class index to extract from original mask (e.g., 5 for RV)')
    parser.add_argument('--batch_size', type=int, default=65536, help='Inference batch size (default: 65536)')
    parser.add_argument('--all_samples', action='store_true', help='Process all samples')
    
    args = parser.parse_args()
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Setup paths
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load Info
    info_path = os.path.join(args.data_path, 'info.csv')
    if not os.path.exists(info_path):
        print(f"Error: info.csv not found at {info_path}")
        return
        
    info_df = pd.read_csv(info_path)
    # Check for common column names for ID
    if 'sample_id' in info_df.columns:
        id_col = 'sample_id'
    elif 'id' in info_df.columns:
        id_col = 'id'
    else:
        # Fallback to first column
        id_col = info_df.columns[0]
        print(f"Warning: 'sample_id' or 'id' column not found. Using first column '{id_col}' as ID.")

    sample_ids = info_df[id_col].values
    
    if not args.all_samples:
        sample_ids = sample_ids[:args.n_samples]
        
    print(f"Processing {len(sample_ids)} samples...")
    
    # Load Model
    model = load_checkpoint(
        args.checkpoint,
        n_samples=len(info_df), # Total samples in training
        latent_dimension=args.latent_dim,
        decoder_channels=[64, 48, 32, 16] # Default architecture
    )
    
    results = []
    
    for i, sample_id in enumerate(tqdm(sample_ids)):
        sample_id = str(sample_id) # Ensure sample_id is string
        # 1. Inference
        # Note: sample_idx must match the training index. 
        # Assuming info.csv order matches training order.
        # We need the index from the full dataframe.
        # Use string comparison if needed, or ensure types match
        # info_df[id_col] might be int, sample_id is str. 
        # Let's rely on the original type for lookup, but use str for filenames.
        
        # Get original type sample_id for lookup
        lookup_id = sample_ids[i] 
        full_idx = info_df[info_df[id_col] == lookup_id].index[0]
        
        refined_prob = inference_single_sample(model, full_idx, args.resolution, batch_size=args.batch_size)
        refined_mask = (refined_prob > args.threshold).astype(np.uint8)
        
        # Save Refined Mask
        np.save(os.path.join(args.output_dir, f'{sample_id}_refined.npy'), refined_mask)
        
        # 2. Load Original Data
        # Assuming masks are in {data_path}/masks/{sample_id}.npy or {data_path}/shape/{sample_id}.npy
        # Adjust path structure if needed based on your dataset
        orig_mask_path = os.path.join(args.data_path, 'masks', f'{sample_id}.npy')
        if not os.path.exists(orig_mask_path):
            # Try 'shape' directory (common in this project)
            orig_mask_path = os.path.join(args.data_path, 'shape', f'{sample_id}.npy')
            
        if not os.path.exists(orig_mask_path):
            # Try searching recursively or standard locations if not found
            # Fallback to checking if it's just in data_path
            orig_mask_path = os.path.join(args.data_path, f'{sample_id}.npy')
            
        if os.path.exists(orig_mask_path):
            original_mask = np.load(orig_mask_path)
            
            # Handle multi-class original mask
            if original_mask.max() > 1:
                # print(f"Extracting class {args.target_class} from multi-class mask...")
                original_mask = (original_mask == args.target_class).astype(np.uint8)

            # AUTO-DETECT ORIENTATION
            # The Refined Mask is generated in (D, H, W) format (matching the model/training).
            # The Original Mask might be in (H, W, D) format (common in .npy).
            # If the volume is cubic (e.g. 256^3), shape checks won't detect this.
            # We compare overlap (intersection) with the Refined Mask to decide.
            
            overlap_asis = np.sum((refined_mask > 0) & (original_mask > 0))
            
            # Try Transpose (2, 0, 1): H,W,D -> D,H,W
            mask_trans = original_mask.transpose(2, 0, 1)
            overlap_trans = np.sum((refined_mask > 0) & (mask_trans > 0))
            
            if overlap_trans > overlap_asis:
                print(f"[{sample_id}] Auto-correcting orientation: Transposing Original Mask (2,0,1). Overlap improved: {overlap_asis} -> {overlap_trans}")
                original_mask = mask_trans
                mask_was_transposed = True
            else:
                mask_was_transposed = False
            
            # 3. Load Original Image (for visualization and shape check)
            img_path = os.path.join(args.data_path, 'appearance', f'{sample_id}.npy')
            if not os.path.exists(img_path):
                img_path = os.path.join(args.data_path, 'images', f'{sample_id}.npy')
            if not os.path.exists(img_path):
                img_path = os.path.join(args.data_path, 'images_256', f'{sample_id}.npy')
                
            original_image = None
            if os.path.exists(img_path):
                original_image = np.load(img_path)
                
                # Check for dimension mismatch between Image and Original Mask
                # If Image is (D, H, W) and Mask is (H, W, D), we need to transpose Mask.
                # NOTE: We already did auto-correction based on Refined Mask overlap above.
                # So we can skip the shape-based check or just keep it as a sanity check.
                # But since we trusted Refined Mask, let's trust the auto-correction.
                
                # FORCE TRANSPOSE IMAGE (X, Y, Z) -> (Z, Y, X)
                # Based on inspection, Image is (X, Y, Z) and Mask is (Z, Y, X).
                if len(original_image.shape) == 3:
                     print(f"[{sample_id}] Transposing Original Image (2, 1, 0) to match (Z, Y, X) mask.")
                     original_image = original_image.transpose(2, 1, 0)
            else:
                print(f"Warning: Original image not found for {sample_id}")

            # 4. Compute Metrics
            metrics = compute_metrics(original_mask, refined_mask)
            metrics['sample_id'] = sample_id
            results.append(metrics)
            
            # 5. Visualization
            visualize_comparison(
                original_mask, 
                refined_mask, 
                original_image, 
                sample_id, 
                output_path=os.path.join(vis_dir, f'{sample_id}_vis.png')
            )
        else:
            print(f"Warning: Original mask not found for {sample_id}")
            
    # Save Results
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(args.output_dir, 'evaluation_results.csv')
        df.to_csv(csv_path, index=False)
        
        print("\nSummary:")
        print(f"Mean Voxel Ratio: {df['voxel_ratio'].mean():.4f}")
        print(f"Mean Original CC: {df['original_cc'].mean():.2f}")
        print(f"Mean Refined CC: {df['refined_cc'].mean():.2f}")
        print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()
