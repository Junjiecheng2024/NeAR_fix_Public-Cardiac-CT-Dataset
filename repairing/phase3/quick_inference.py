"""
快速推理脚本 - 为特定样本运行Phase1 NeAR推理
只推理指定的case_ids，用于可视化
"""
import sys
import os
import argparse
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

# Add project root to path
sys.path.insert(0, '/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset')

from near.models.nn3d.model_shape_only import EmbeddingDecoderShapeOnly
from near.utils.misc import to_device, to_var

# 类别配置 - 每个类别最新的checkpoint
CLASS_CONFIG = {
    1: {
        'name': 'Myocardium',
        'checkpoint': 'Myocardium_class1_251122_225221/best.ckpt',
    },
    2: {
        'name': 'LA',
        'checkpoint': 'LA_class2_251124_155401/best.ckpt',
    },
    3: {
        'name': 'LV',
        'checkpoint': 'LV_class3_251124_153327/best.ckpt',
    },
    4: {
        'name': 'RA',
        'checkpoint': 'RA_class4_251124_151008/best.ckpt',
    },
    5: {
        'name': 'RV',
        'checkpoint': 'RV_class5_shape_only_251121_111501/best.ckpt',
    },
    6: {
        'name': 'Aorta',
        'checkpoint': 'Aorta_class6_shape_only_251123_040844/best.ckpt',
    },
    7: {
        'name': 'PA',
        'checkpoint': 'PA_class7_251122_015831/best.ckpt',
    },
    8: {
        'name': 'LAA',
        'checkpoint': 'LAA_class8_251122_222337/best.ckpt',
    },
    9: {
        'name': 'Coronary', 
        'checkpoint': 'Coronary_class9_251126_101417/best.ckpt',
    },
    10: {
        'name': 'PV',
        'checkpoint': 'PV_class10_251124_223846/best.ckpt',
    },
}

def load_checkpoint(checkpoint_path, n_samples, latent_dimension=256, decoder_channels=[64, 48, 32, 16]):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    model = to_device(
        EmbeddingDecoderShapeOnly(
            n_samples=n_samples,
            latent_dimension=latent_dimension,
            decoder_channels=decoder_channels
        )
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('model.'):
            name = k[6:]
        elif k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
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

def inference_single_sample(model, sample_idx, resolution=256, batch_size=8192):
    """Generate refined mask for a single sample."""
    grid = generate_grid_coordinates(resolution)
    grid_flat = grid.reshape(-1, 3)
    n_points = grid_flat.shape[0]
    
    # indices must be LongTensor
    indices = torch.LongTensor([sample_idx])
    if torch.cuda.is_available():
        indices = indices.cuda()
    
    predictions = []
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, n_points, batch_size), desc=f"Inferring", leave=False):
            end_idx = min(start_idx + batch_size, n_points)
            # Reshape grid to (1, N, 1, 1, 3) to match 5D expectation of grid_sample
            batch_grid = grid_flat[start_idx:end_idx].unsqueeze(0).unsqueeze(2).unsqueeze(3)
            if torch.cuda.is_available():
                batch_grid = batch_grid.cuda()
            
            # model signature: forward(indices, grid)
            pred_logit, _ = model(indices, batch_grid)
            pred_prob = torch.sigmoid(pred_logit)
            predictions.append(pred_prob.squeeze().cpu())
    
    predictions = torch.cat(predictions, dim=0)
    predictions = predictions.numpy().reshape(resolution, resolution, resolution)
    
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_ids', type=str, default='24,327,75,655,229,637,201',
                        help='Comma-separated case IDs to infer')
    parser.add_argument('--classes', type=str, default='1,2,3,4,5,6,7,8,9,10',
                        help='Classes to infer (1-10)')
    parser.add_argument('--data_path', type=str, 
                        default='/scratch/project_2016517/junjie/dataset/near_format_data',
                        help='Path to near_format_data')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/phase1/checkpoints',
                        help='Path to checkpoints')
    parser.add_argument('--output_dir', type=str,
                        default='/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/phase2',
                        help='Output directory')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    
    case_ids = [int(x.strip()) for x in args.case_ids.split(',')]
    classes = [int(x.strip()) for x in args.classes.split(',')]
    
    # Load info.csv to get sample indices
    info_path = os.path.join(args.data_path, 'info.csv')
    info_df = pd.read_csv(info_path)
    
    if 'sample_id' in info_df.columns:
        id_col = 'sample_id'
    elif 'id' in info_df.columns:
        id_col = 'id'
    else:
        id_col = info_df.columns[0]
    
    n_samples = len(info_df)
    print(f"Total samples in dataset: {n_samples}")
    print(f"Cases to infer: {case_ids}")
    print(f"Classes to infer: {classes}")
    
    for class_id in classes:
        if class_id not in CLASS_CONFIG:
            print(f"Warning: Class {class_id} not configured, skipping")
            continue
            
        cfg = CLASS_CONFIG[class_id]
        class_name = cfg['name']
        checkpoint_path = os.path.join(args.checkpoint_dir, cfg['checkpoint'])
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing Class {class_id}: {class_name}")
        print(f"{'='*60}")
        
        # Create output directory
        output_subdir = os.path.join(args.output_dir, f"class{class_id}_{class_name}", 
                                      f"class{class_id}_{class_name}_results_256")
        os.makedirs(output_subdir, exist_ok=True)
        
        # Load model
        model = load_checkpoint(checkpoint_path, n_samples)
        
        for case_id in case_ids:
            # Find sample index
            matches = info_df[info_df[id_col] == case_id]
            if len(matches) == 0:
                # Try string match
                matches = info_df[info_df[id_col].astype(str) == str(case_id)]
            
            if len(matches) == 0:
                print(f"  Warning: Case {case_id} not found in info.csv")
                continue
                
            sample_idx = matches.index[0]
            print(f"\n  Case {case_id} (index {sample_idx})")
            
            # Inference
            refined_prob = inference_single_sample(model, sample_idx, args.resolution)
            refined_mask = (refined_prob > args.threshold).astype(np.uint8)
            
            # Save
            output_path = os.path.join(output_subdir, f"{case_id}.npy")
            np.save(output_path, refined_mask)
            print(f"    Saved: {output_path}")
            print(f"    Volume: {refined_mask.sum()} voxels")
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
