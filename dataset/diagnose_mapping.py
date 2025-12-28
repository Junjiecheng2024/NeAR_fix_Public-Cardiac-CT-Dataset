#!/usr/bin/env python3
"""
诊断脚本：检查坐标映射是否正确
"""

import numpy as np
import json
import os

# 尝试导入 nibabel
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("Warning: nibabel not installed, will skip NIfTI loading")

# 尝试导入 scipy
try:
    from scipy.ndimage import zoom
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed, will skip zoom operations")

# 配置路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TIER2_ROOT = os.path.join(BASE_DIR, "coronary_tier2")
P1_GLOBAL_ROOT = os.path.join(BASE_DIR, "coronary_global")
GT_ROOT = os.path.join(BASE_DIR, "original", "segmentations")

case_id = "1"
class_id = 9  # Coronary

print("=" * 70)
print(f"坐标映射诊断 - Case: {case_id}")
print("=" * 70)

# 1. 加载 crop_params
crop_params_path = os.path.join(TIER2_ROOT, case_id, "crop_params.json")
try:
    with open(crop_params_path) as f:
        params = json.load(f)
    print("\n[1. Crop Params]")
    print(f"  origin: {params['origin']} (裁剪起点在原始空间)")
    print(f"  size: {params['size']} (原始裁剪尺寸)")
    print(f"  original_shape: {params['original_shape']} (原始 GT 尺寸)")
    print(f"  cropped_shape: {params['cropped_shape']} (Tier2 保存尺寸)")
    print(f"  resize_applied: {params['resize_applied']}")
except Exception as e:
    print(f"  ERROR loading crop_params: {e}")
    params = None

# 2. 加载 Tier2 数据 (训练用的数据)
tier2_files = {
    'ct': os.path.join(TIER2_ROOT, case_id, "ct.npy"),
    'mask_target': os.path.join(TIER2_ROOT, case_id, "mask_target.npy"),
    'mask_context': os.path.join(TIER2_ROOT, case_id, "mask_context.npy"),
    'seg_full': os.path.join(TIER2_ROOT, case_id, "seg_full.npy"),
}

print("\n[2. Tier2 数据 (训练用)]")
for name, path in tier2_files.items():
    if os.path.exists(path):
        data = np.load(path)
        print(f"  {name}: shape={data.shape}, dtype={data.dtype}", end="")
        if 'mask' in name or 'seg' in name:
            print(f", positive={(data > 0).sum()}, unique={np.unique(data)[:10]}")
        else:
            print(f", range=[{data.min():.3f}, {data.max():.3f}]")
    else:
        print(f"  {name}: NOT FOUND")

# 检查 mask_target 的内容
mask_target_path = tier2_files['mask_target']
if os.path.exists(mask_target_path):
    mask_target = np.load(mask_target_path)
    print(f"\n[2.1 mask_target 详情]")
    print(f"  Shape: {mask_target.shape}")
    print(f"  Unique values: {np.unique(mask_target)}")
    print(f"  Positive voxels: {(mask_target > 0).sum()}")
    if (mask_target > 0).sum() > 0:
        z, y, x = np.where(mask_target > 0)
        print(f"  Z range: {z.min()} - {z.max()} (span: {z.max()-z.min()+1})")
        print(f"  Y range: {y.min()} - {y.max()} (span: {y.max()-y.min()+1})")
        print(f"  X range: {x.min()} - {x.max()} (span: {x.max()-x.min()+1})")

# 3. 加载原始 GT
print("\n[3. 原始 Ground Truth]")
gt_path = os.path.join(GT_ROOT, f"{case_id}.nii.img.nii.gz")

if HAS_NIBABEL and os.path.exists(gt_path):
    gt_nii = nib.load(gt_path)
    gt_full = gt_nii.get_fdata().astype(np.uint8)
    print(f"  Path: {gt_path}")
    print(f"  Shape: {gt_full.shape}")
    print(f"  Affine:\n{gt_nii.affine}")
    print(f"  Unique labels: {np.unique(gt_full)}")
    
    # 提取 Coronary (class 9)
    gt_coronary = (gt_full == class_id).astype(np.uint8)
    print(f"\n[3.1 GT Coronary (class={class_id}) 位置 - 原始空间]")
    print(f"  Positive voxels: {gt_coronary.sum()}")
    if gt_coronary.sum() > 0:
        z, y, x = np.where(gt_coronary > 0)
        print(f"  Z range: {z.min()} - {z.max()} (span: {z.max()-z.min()+1})")
        print(f"  Y range: {y.min()} - {y.max()} (span: {y.max()-y.min()+1})")
        print(f"  X range: {x.min()} - {x.max()} (span: {x.max()-x.min()+1})")
        print(f"  Center: [{(z.min()+z.max())//2}, {(y.min()+y.max())//2}, {(x.min()+x.max())//2}]")
        
        # 对比 crop_params 中的位置
        if params:
            origin = params['origin']
            size = params['size']
            expected_end = [origin[i] + size[i] for i in range(3)]
            print(f"\n[3.2 位置对比]")
            print(f"  Crop origin: {origin}")
            print(f"  Crop end: {expected_end}")
            print(f"  GT bbox: [{z.min()}, {y.min()}, {x.min()}] -> [{z.max()}, {y.max()}, {x.max()}]")
            
            # 检查 GT 是否在裁剪区域内
            in_crop_z = (z.min() >= origin[0]) and (z.max() < expected_end[0])
            in_crop_y = (y.min() >= origin[1]) and (y.max() < expected_end[1])
            in_crop_x = (x.min() >= origin[2]) and (x.max() < expected_end[2])
            print(f"  GT 完全在裁剪区域内? Z:{in_crop_z} Y:{in_crop_y} X:{in_crop_x}")
    
    # Zoom 到 256
    if HAS_SCIPY:
        factors = np.array([256, 256, 256]) / np.array(gt_full.shape)
        print(f"\n[3.3 GT Coronary 位置 - 256³ 空间 (评估用)]")
        print(f"  Zoom factors: {factors}")
        gt_coronary_256 = zoom(gt_coronary, factors, order=0)
        print(f"  256 shape: {gt_coronary_256.shape}")
        print(f"  Positive voxels: {(gt_coronary_256 > 0).sum()}")
        if (gt_coronary_256 > 0).sum() > 0:
            z, y, x = np.where(gt_coronary_256 > 0)
            print(f"  Z range: {z.min()} - {z.max()}")
            print(f"  Y range: {y.min()} - {y.max()}")
            print(f"  X range: {x.min()} - {x.max()}")
            print(f"  Center: [{(z.min()+z.max())//2}, {(y.min()+y.max())//2}, {(x.min()+x.max())//2}]")

elif not os.path.exists(gt_path):
    print(f"  ERROR: GT file not found at {gt_path}")
else:
    print(f"  SKIPPED: nibabel not installed")

# 3.5 加载 Phase 1 Global 输出 (推理结果)
print("\n[3.5 Phase 1 Global 输出 (推理结果)]")
p1_path = os.path.join(P1_GLOBAL_ROOT, f"{case_id}_mask.npy")
if os.path.exists(p1_path):
    p1_mask = np.load(p1_path)
    print(f"  Path: {p1_path}")
    print(f"  Shape: {p1_mask.shape}")
    print(f"  Dtype: {p1_mask.dtype}")
    print(f"  Unique values: {np.unique(p1_mask)[:10]}")
    print(f"  Positive voxels: {(p1_mask > 0).sum()}")
    if (p1_mask > 0).sum() > 0:
        z, y, x = np.where(p1_mask > 0)
        print(f"  Z range: {z.min()} - {z.max()} (span: {z.max()-z.min()+1})")
        print(f"  Y range: {y.min()} - {y.max()} (span: {y.max()-y.min()+1})")
        print(f"  X range: {x.min()} - {x.max()} (span: {x.max()-x.min()+1})")
        print(f"  Center: [{(z.min()+z.max())//2}, {(y.min()+y.max())//2}, {(x.min()+x.max())//2}]")
        
        # 直接与 GT 256 对比计算 Dice
        if HAS_NIBABEL and HAS_SCIPY and 'gt_coronary_256' in dir():
            p1_bin = (p1_mask > 0.5).astype(np.uint8)
            gt_bin = (gt_coronary_256 > 0.5).astype(np.uint8)
            intersection = np.logical_and(p1_bin, gt_bin).sum()
            union = p1_bin.sum() + gt_bin.sum()
            dice = 2 * intersection / (union + 1e-8)
            print(f"\n  [P1 vs GT Dice]")
            print(f"    Dice: {dice:.6f}")
            print(f"    P1 volume: {p1_bin.sum()}")
            print(f"    GT volume: {gt_bin.sum()}")
            print(f"    Intersection: {intersection}")
            
            # 检查位置对比
            gz, gy, gx = np.where(gt_bin > 0)
            if len(gz) > 0:
                print(f"\n  [位置对比]")
                print(f"    P1 位置: Z[{z.min()}-{z.max()}] Y[{y.min()}-{y.max()}] X[{x.min()}-{x.max()}]")
                print(f"    GT 位置: Z[{gz.min()}-{gz.max()}] Y[{gy.min()}-{gy.max()}] X[{gx.min()}-{gx.max()}]")
                
                # 检查是否有重叠
                overlap_z = not (z.max() < gz.min() or z.min() > gz.max())
                overlap_y = not (y.max() < gy.min() or y.min() > gy.max())
                overlap_x = not (x.max() < gx.min() or x.min() > gx.max())
                if overlap_z and overlap_y and overlap_x:
                    print(f"    ✓ 位置有重叠")
                else:
                    print(f"    ✗ 位置无重叠! 这是 Dice 极低的原因!")
                    print(f"    Overlap check: Z:{overlap_z} Y:{overlap_y} X:{overlap_x}")
else:
    print(f"  NOT FOUND: {p1_path}")

# 4. 模拟 map_to_global 映射
if params and HAS_SCIPY:
    print("\n" + "=" * 70)
    print("[4. 模拟 map_to_global 坐标映射]")
    print("=" * 70)
    
    origin = np.array(params['origin'])  # 原始空间的裁剪起点
    original_crop_size = np.array(params['size'])  # 原始裁剪尺寸
    original_full_shape = tuple(params['original_shape'])  # 原始 GT 尺寸
    cropped_shape = np.array(params['cropped_shape'])  # Tier2 尺寸 (128³)
    
    # 假设推理分辨率是 256
    inference_resolution = 256
    pred_crop_shape = np.array([inference_resolution] * 3)
    
    print(f"\n[4.1 参数]")
    print(f"  推理分辨率: {inference_resolution}³")
    print(f"  原始裁剪尺寸: {original_crop_size}")
    print(f"  Tier2 尺寸: {cropped_shape}")
    print(f"  原始 GT 尺寸: {original_full_shape}")
    
    # Step 1: pred_crop (256³) -> original crop size (318x282x172)
    if params.get('resize_applied', False):
        zoom_factors_step1 = original_crop_size / pred_crop_shape
        restored_size = pred_crop_shape * zoom_factors_step1
    else:
        zoom_factors_step1 = np.ones(3)
        restored_size = pred_crop_shape
    
    print(f"\n[4.2 Step 1: 恢复到原始裁剪尺寸]")
    print(f"  zoom_factors = {original_crop_size} / {pred_crop_shape} = {zoom_factors_step1}")
    print(f"  恢复后尺寸: {restored_size.astype(int)}")
    
    # Step 2: 放入原始空间
    end = origin + original_crop_size
    print(f"\n[4.3 Step 2: 放入原始空间]")
    print(f"  Origin: {origin}")
    print(f"  End: {end}")
    print(f"  在原始空间 {original_full_shape} 中的位置: [{origin[0]}:{end[0]}, {origin[1]}:{end[1]}, {origin[2]}:{end[2]}]")
    
    # Step 3: 从原始空间 zoom 到 256³
    global_shape = np.array([256, 256, 256])
    zoom_factors_step3 = global_shape / np.array(original_full_shape)
    
    final_origin = origin * zoom_factors_step3
    final_end = end * zoom_factors_step3
    final_size = final_end - final_origin
    
    print(f"\n[4.4 Step 3: 从原始空间 zoom 到 256³]")
    print(f"  zoom_factors = {global_shape} / {original_full_shape} = {zoom_factors_step3}")
    print(f"  最终位置 (256³ 空间):")
    print(f"    Origin: {final_origin.astype(int)}")
    print(f"    End: {final_end.astype(int)}")
    print(f"    Size: {final_size.astype(int)}")
    
    # 与 GT 256 位置对比
    if HAS_NIBABEL and os.path.exists(gt_path) and (gt_coronary_256 > 0).sum() > 0:
        z, y, x = np.where(gt_coronary_256 > 0)
        gt_origin = np.array([z.min(), y.min(), x.min()])
        gt_end = np.array([z.max(), y.max(), x.max()])
        
        print(f"\n[4.5 与 GT 256³ 位置对比]")
        print(f"  预测位置: [{final_origin.astype(int)}] -> [{final_end.astype(int)}]")
        print(f"  GT 位置:  [{gt_origin}] -> [{gt_end}]")
        
        # 检查重叠
        overlap_start = np.maximum(final_origin, gt_origin)
        overlap_end = np.minimum(final_end, gt_end)
        overlap = np.all(overlap_end > overlap_start)
        if overlap:
            overlap_size = overlap_end - overlap_start
            print(f"  ✓ 有重叠! 重叠范围: {overlap_start.astype(int)} -> {overlap_end.astype(int)}")
        else:
            print(f"  ✗ 无重叠! 这就是 Dice 接近 0 的原因!")
            
            # 计算偏移量
            offset = gt_origin - final_origin.astype(int)
            print(f"  需要的偏移量: {offset}")

print("\n" + "=" * 70)
print("诊断完成")
print("=" * 70)
