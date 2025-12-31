#!/usr/bin/env python3
"""
Phase 3.5: CT-Based Boundary Refinement
使用形态学活动轮廓（Morphological Active Contours）优化分割边界，使其更贴合 CT 图像的真实边缘。

原理：
1. 以 Phase 1/2 的预测作为初始分割
2. 让轮廓根据 CT 图像的灰度信息自动演化
3. 边界会自动移动到灰度变化最大的位置（即真实的组织边界）

Usage:
    python refine_with_ct.py --case_id 1 --data_root /path/to/dataset
"""
import os
import sys
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, gaussian_filter
from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour
from skimage.filters import sobel
from tqdm import tqdm
import glob

CLASS_NAMES = {
    1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
    6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
}

def load_ct_256(ct_path):
    """Load CT and resize to 256³"""
    nii = nib.load(ct_path)
    ct = nii.get_fdata()
    
    if ct.shape != (256, 256, 256):
        zoom_fac = np.array([256, 256, 256]) / np.array(ct.shape)
        ct = zoom(ct, zoom_fac, order=1)
    
    # Normalize to [0, 1]
    ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-8)
    return ct

def refine_mask_with_ct(mask, ct, iterations=50, smoothing=1, band_width=5):
    """
    使用 Morphological Chan-Vese 活动轮廓优化 mask 边界
    
    Args:
        mask: 初始分割 (256, 256, 256) binary
        ct: CT 图像 (256, 256, 256) float [0, 1]
        iterations: 演化迭代次数
        smoothing: 边界平滑程度
        band_width: 只在原始边界 ±N 像素范围内优化
    
    Returns:
        refined_mask: 优化后的分割
    """
    from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
    
    # 为了速度，先降采样到 128³ 处理
    mask_128 = zoom(mask.astype(float), 0.5, order=0) > 0.5
    ct_128 = zoom(ct, 0.5, order=1)
    
    # 创建窄带区域：只在原始边界 ±band_width 像素内进行优化
    inner = binary_erosion(mask_128, iterations=band_width)
    outer = binary_dilation(mask_128, iterations=band_width)
    band = outer & ~inner  # 边界带
    
    # 应用轻微高斯模糊减少噪声
    ct_smooth = gaussian_filter(ct_128, sigma=1)
    
    # Morphological Chan-Vese (只在边界带内运行)
    refined_128 = morphological_chan_vese(
        ct_smooth, 
        num_iter=iterations,
        init_level_set=mask_128.astype(float),
        smoothing=smoothing,
        lambda1=1,
        lambda2=1
    )
    
    # 关键：限制结果只在窄带内变化
    # 窄带内用 chan-vese 结果，窄带外保持原样
    final_128 = np.where(band, refined_128, mask_128)
    
    # 上采样回 256³
    refined = zoom(final_128.astype(float), 2, order=0) > 0.5
    
    return refined.astype(np.uint8)

def refine_mask_with_geodesic(mask, ct, iterations=50):
    """
    使用 Geodesic Active Contour（基于边缘）优化
    适合边缘清晰的结构（如心腔边界）
    """
    mask_128 = zoom(mask.astype(float), 0.5, order=0) > 0.5
    ct_128 = zoom(ct, 0.5, order=1)
    
    # 计算边缘图 (g = 1 / (1 + |∇I|))
    edges = sobel(ct_128)
    gimage = 1.0 / (1.0 + edges * 10)
    
    # Geodesic Active Contour
    refined_128 = morphological_geodesic_active_contour(
        gimage,
        num_iter=iterations,
        init_level_set=mask_128.astype(float),
        smoothing=1,
        balloon=-1  # 收缩倾向（-1 收缩，1 膨胀）
    )
    
    refined = zoom(refined_128.astype(float), 2, order=0) > 0.5
    return refined.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="/scratch/project_2016517/JunjieCheng/dataset")
    parser.add_argument("--iterations", type=int, default=30, help="Active contour iterations")
    parser.add_argument("--method", type=str, default="chan_vese", choices=["chan_vese", "geodesic"])
    args = parser.parse_args()
    
    # Find CT
    ct_candidates = [
        os.path.join(args.data_root, "original", "images", f"{args.case_id}.nii.gz"),
        os.path.join(args.data_root, "original", "images", f"{args.case_id}.nii.img.nii.gz"),
    ]
    ct_path = next((p for p in ct_candidates if os.path.exists(p)), None)
    
    if not ct_path:
        print(f"CT not found for case {args.case_id}")
        return
    
    print(f"Loading CT: {ct_path}")
    ct = load_ct_256(ct_path)
    print(f"CT shape: {ct.shape}, range: [{ct.min():.2f}, {ct.max():.2f}]")
    
    # Process each class
    output_dir = os.path.join(args.data_root, "refined")
    os.makedirs(output_dir, exist_ok=True)
    
    for cls_id, cls_name in CLASS_NAMES.items():
        cls_lower = cls_name.lower()
        
        # Load Phase 1 or Phase 2 mask
        mask_path = os.path.join(args.data_root, f"{cls_lower}_morph", f"{args.case_id}_mask.npy")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(args.data_root, f"{cls_lower}_global", f"{args.case_id}_mask.npy")
        
        if not os.path.exists(mask_path):
            print(f"  [{cls_name}] Mask not found, skipping")
            continue
        
        mask = np.load(mask_path)
        mask = (mask > 0.5).astype(np.uint8)
        
        if mask.sum() == 0:
            print(f"  [{cls_name}] Empty mask, skipping")
            continue
        
        print(f"  [{cls_name}] Refining with {args.method}... ", end="")
        
        if args.method == "chan_vese":
            refined = refine_mask_with_ct(mask, ct, iterations=args.iterations)
        else:
            refined = refine_mask_with_geodesic(mask, ct, iterations=args.iterations)
        
        # 保存
        out_path = os.path.join(output_dir, f"{args.case_id}_{cls_lower}_refined.npy")
        np.save(out_path, refined)
        
        # 统计
        vol_before = mask.sum()
        vol_after = refined.sum()
        change = (vol_after - vol_before) / vol_before * 100
        print(f"Vol: {vol_before:,} -> {vol_after:,} ({change:+.1f}%)")
    
    print(f"\nRefined masks saved to {output_dir}")

if __name__ == "__main__":
    main()
