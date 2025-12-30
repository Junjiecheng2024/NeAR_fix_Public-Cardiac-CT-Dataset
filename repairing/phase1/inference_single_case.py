#!/usr/bin/env python3
"""
Single Case Inference Script
推理单个 case 的所有 10 个类，并计算 Dice 验证

Usage:
    python inference_single_case.py --case_id 10 --data_root /path/to/dataset
"""
import os
import sys
import argparse
import numpy as np
import nibabel as nib
import torch
import json
from scipy.ndimage import zoom
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from near.models.nn3d.model_shape_appearance import EmbeddingDecoderShapeAppearanceWithContext
from near.datasets.coronary_tier2_dataset import CoronaryTier2Dataset
from repairing.phase1.inference import create_full_grid, map_to_global

# Class definitions
CLASS_INFO = {
    1:  {"name": "Myocardium", "dir": "myocardium", "config": "config.py"},
    2:  {"name": "LA",         "dir": "la",         "config": "config_2016526.py"},
    3:  {"name": "LV",         "dir": "lv",         "config": "config_2016526.py"},
    4:  {"name": "RA",         "dir": "ra",         "config": "config_2016526.py"},
    5:  {"name": "RV",         "dir": "rv",         "config": "config_2016526.py"},
    6:  {"name": "Aorta",      "dir": "aorta",      "config": "config.py"},
    7:  {"name": "PA",         "dir": "pa",         "config": "config_2016526.py"},
    8:  {"name": "LAA",        "dir": "laa",        "config": "config_2016526.py"},
    9:  {"name": "Coronary",   "dir": "coronary",   "config": "config.py"},
    10: {"name": "PV",         "dir": "pv",         "config": "config_2016526.py"},
}

# Checkpoint base paths
CKPT_BASES = {
    "config.py": "/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/checkpoints",
    "config_2016526.py": "/scratch/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset/phase1/checkpoints",
}

def find_checkpoint(class_info):
    """Find the best checkpoint for a class."""
    ckpt_base = CKPT_BASES[class_info["config"]]
    run_flag = f"{class_info['name']}_Tier2_"
    
    # Find latest checkpoint
    import glob
    pattern = os.path.join(ckpt_base, f"{run_flag}*/best.ckpt")
    matches = sorted(glob.glob(pattern), reverse=True)
    
    if matches:
        return matches[0]
    return None

def run_inference_for_class(class_id, case_id, data_root, device):
    """Run inference for a single class on a single case."""
    info = CLASS_INFO[class_id]
    class_name = info["name"]
    class_dir = info["dir"]
    
    # Paths
    tier2_dir = os.path.join(data_root, f"{class_dir}_tier2")
    case_dir = os.path.join(tier2_dir, str(case_id))
    output_dir = os.path.join(data_root, f"{class_dir}_global")
    
    # Check if case exists
    if not os.path.exists(case_dir):
        print(f"  [SKIP] Case {case_id} not found in {tier2_dir}")
        return None, None
    
    # Load dataset
    dataset = CoronaryTier2Dataset(
        root=tier2_dir,
        resolution=128,
        use_appearance=True,
        boundary_bias_ratio=0.0,
        augment=False
    )
    
    # Find case index
    case_idx = None
    for i, d in enumerate(dataset.case_dirs):
        if d.name == str(case_id):
            case_idx = i
            break
    
    if case_idx is None:
        print(f"  [SKIP] Case {case_id} not in dataset")
        return None, None
    
    # Find checkpoint
    ckpt_path = find_checkpoint(info)
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"  [SKIP] Checkpoint not found for {class_name}")
        return None, None
    
    print(f"  Checkpoint: {os.path.basename(os.path.dirname(ckpt_path))}")
    
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
    
    # Get data
    batch = dataset[case_idx]
    appearance = batch['appearance'].unsqueeze(0).to(device)
    context = batch['context'].unsqueeze(0).to(device)
    indices = torch.tensor([case_idx], dtype=torch.long, device=device)
    grid = create_full_grid((128, 128, 128), device)
    
    # Inference
    with torch.no_grad():
        pred_logit, _ = model(indices, grid, appearance, context)
        pred_prob = torch.sigmoid(pred_logit)
    
    mask_crop = (pred_prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    
    # Map to global
    crop_params = dataset.get_crop_params(str(case_id))
    mask_global = map_to_global(mask_crop, crop_params, (256, 256, 256))
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{case_id}_mask.npy")
    np.save(output_path, mask_global)
    
    return mask_global, mask_crop.sum()

def compute_dice(pred, gt):
    intersection = np.logical_and(pred > 0, gt > 0).sum()
    return 2 * intersection / (pred.sum() + gt.sum() + 1e-8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", type=str, required=True, help="Case ID to process")
    parser.add_argument("--data_root", type=str, default="/scratch/project_2016517/JunjieCheng/dataset")
    parser.add_argument("--gt_root", type=str, default=None, help="GT root (default: data_root/original/segmentations)")
    args = parser.parse_args()
    
    if args.gt_root is None:
        args.gt_root = os.path.join(args.data_root, "original", "segmentations")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Case ID: {args.case_id}")
    print(f"Data Root: {args.data_root}")
    print("=" * 60)
    
    # Load GT
    gt_candidates = [
        os.path.join(args.gt_root, f"{args.case_id}.nii.gz"),
        os.path.join(args.gt_root, f"{args.case_id}.nii.img.nii.gz"),
    ]
    gt_path = next((p for p in gt_candidates if os.path.exists(p)), None)
    
    if not gt_path:
        print(f"GT not found! Tried: {gt_candidates}")
        return
    
    gt_data = nib.load(gt_path).get_fdata()
    print(f"GT shape: {gt_data.shape}")
    
    results = []
    
    for class_id, info in CLASS_INFO.items():
        print(f"\n[{class_id:2d}] {info['name']}")
        
        try:
            mask_global, crop_vol = run_inference_for_class(
                class_id, args.case_id, args.data_root, device
            )
            
            if mask_global is None:
                results.append((class_id, info['name'], 0, 0, 0, "SKIP"))
                continue
            
            # Compute Dice with GT
            gt_cls = (gt_data == class_id).astype(np.uint8)
            zoom_fac = np.array([256, 256, 256]) / np.array(gt_cls.shape)
            gt_256 = zoom(gt_cls, zoom_fac, order=0)
            gt_256 = (gt_256 > 0.5).astype(np.uint8)
            
            dice = compute_dice(mask_global, gt_256)
            
            print(f"  Crop Vol: {crop_vol:,}, Global Vol: {mask_global.sum():,}, GT Vol: {gt_256.sum():,}")
            print(f"  Dice: {dice:.4f}")
            
            results.append((class_id, info['name'], mask_global.sum(), gt_256.sum(), dice, "OK"))
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append((class_id, info['name'], 0, 0, 0, f"ERROR: {e}"))
    
    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY - Case {args.case_id}")
    print("=" * 60)
    print(f"{'Class':<12} {'P1_Vol':>10} {'GT_Vol':>10} {'Dice':>8} {'Status':<10}")
    print("-" * 55)
    for class_id, name, p1_vol, gt_vol, dice, status in results:
        print(f"{name:<12} {p1_vol:>10,} {gt_vol:>10,} {dice:>8.4f} {status:<10}")
    print("=" * 60)

if __name__ == "__main__":
    main()
