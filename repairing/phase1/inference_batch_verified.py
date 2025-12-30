#!/usr/bin/env python3
"""
Batch Inference Script - 重新推理所有 case
直接使用验证通过的逻辑，不依赖 inference.py

Usage:
    python inference_batch_verified.py --class_name la --data_root /path/to/dataset
"""
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from near.models.nn3d.model_shape_appearance import EmbeddingDecoderShapeAppearanceWithContext
from near.datasets.coronary_tier2_dataset import CoronaryTier2Dataset
from repairing.phase1.inference import create_full_grid, map_to_global
import glob

CLASS_INFO = {
    "myocardium": {"class_id": 1, "run_flag": "Myocardium_Tier2_"},
    "la":         {"class_id": 2, "run_flag": "LA_Tier2_"},
    "lv":         {"class_id": 3, "run_flag": "LV_Tier2_"},
    "ra":         {"class_id": 4, "run_flag": "RA_Tier2_"},
    "rv":         {"class_id": 5, "run_flag": "RV_Tier2_"},
    "aorta":      {"class_id": 6, "run_flag": "Aorta_Tier2_"},
    "pa":         {"class_id": 7, "run_flag": "PA_Tier2_"},
    "laa":        {"class_id": 8, "run_flag": "LAA_Tier2_"},
    "coronary":   {"class_id": 9, "run_flag": "Coronary_Tier2_"},
    "pv":         {"class_id": 10, "run_flag": "PV_Tier2_"},
}

CKPT_BASE = "/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/checkpoints"

def find_checkpoint(run_flag):
    pattern = os.path.join(CKPT_BASE, f"{run_flag}*/best.ckpt")
    matches = sorted(glob.glob(pattern), reverse=True)
    return matches[0] if matches else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", type=str, required=True, 
                        choices=list(CLASS_INFO.keys()),
                        help="Class to process")
    parser.add_argument("--data_root", type=str, 
                        default="/scratch/project_2016517/JunjieCheng/dataset")
    args = parser.parse_args()
    
    info = CLASS_INFO[args.class_name]
    device = torch.device('cuda')
    
    tier2_dir = os.path.join(args.data_root, f"{args.class_name}_tier2")
    output_dir = os.path.join(args.data_root, f"{args.class_name}_global")
    
    print(f"Class: {args.class_name}")
    print(f"Tier2 Dir: {tier2_dir}")
    print(f"Output Dir: {output_dir}")
    
    # Load dataset
    dataset = CoronaryTier2Dataset(
        root=tier2_dir,
        resolution=128,
        use_appearance=True,
        boundary_bias_ratio=0.0,
        augment=False
    )
    print(f"Dataset: {len(dataset)} cases")
    
    # Find checkpoint
    ckpt_path = find_checkpoint(info["run_flag"])
    if not ckpt_path:
        print(f"Checkpoint not found for {info['run_flag']}")
        return
    print(f"Checkpoint: {ckpt_path}")
    
    # Load model
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
    n_samples = ckpt.get('hyper_parameters', {}).get('n_samples', len(dataset))
    
    model = EmbeddingDecoderShapeAppearanceWithContext(
        latent_dimension=256,
        n_samples=n_samples,
        decoder_channels=[64, 48, 32, 16],
        appearance_channels=64,
        use_context=True
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print("Model loaded!")
    
    # Create output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Grid (128³)
    grid = create_full_grid((128, 128, 128), device)
    
    # Process all cases
    for idx in tqdm(range(len(dataset)), desc=args.class_name):
        batch = dataset[idx]
        case_id = batch['case_id']
        
        appearance = batch['appearance'].unsqueeze(0).to(device)
        context = batch['context'].unsqueeze(0).to(device)
        indices = torch.tensor([idx], dtype=torch.long, device=device)
        
        with torch.no_grad():
            pred_logit, _ = model(indices, grid, appearance, context)
            pred_prob = torch.sigmoid(pred_logit)
        
        mask_crop = (pred_prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        
        # Map to global
        crop_params = dataset.get_crop_params(case_id)
        mask_global = map_to_global(mask_crop, crop_params, (256, 256, 256))
        
        # Save
        np.save(os.path.join(output_dir, f"{case_id}_mask.npy"), mask_global)
    
    print(f"\nDone! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
