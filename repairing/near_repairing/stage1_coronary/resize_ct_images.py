"""
工具脚本：CT图像尺寸调整

功能说明：
1. 将裁剪后的CT图像（bbox版本）resize到固定尺寸（如256³）
2. 用途：为推理结果可视化准备CT背景图像
3. 输入：bboxed/images/*.nii.gz（各种尺寸）
4. 输出：near_format_data/images_256/*.npy（统一256³）
5. 插值方法：三线性插值（order=1，适合CT值）
6. HU值范围：裁剪到[-1000, 1000]并归一化到[0, 1]

注意：这不是训练必需的，只是为了后续可视化精修结果

Resize CT images from bboxed to 256^3 for visualization.
Input: /home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/bboxed/images/*.nii.gz
Output: /home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/near_format_data/images_256/*.npy
"""

import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from tqdm import tqdm
import argparse


def resize_ct_image(input_path, output_path, target_size=(256, 256, 256)):
    """
    Resize a single CT image to target size.
    
    Args:
        input_path: path to input .nii.gz file
        output_path: path to output .npy file
        target_size: target 3D size (D, H, W)
    """
    # Load nifti image
    nii_img = nib.load(input_path)
    ct_data = nii_img.get_fdata()
    
    original_shape = ct_data.shape
    
    # Resize
    if ct_data.shape != target_size:
        ct_resized = resize(ct_data, target_size, 
                           order=1,  # Bilinear interpolation
                           preserve_range=True, 
                           anti_aliasing=True)
    else:
        ct_resized = ct_data
    
    # Save as numpy array
    np.save(output_path, ct_resized.astype(np.float32))
    
    return original_shape, ct_resized.shape


def main():
    parser = argparse.ArgumentParser(description='Resize CT images to 256^3')
    parser.add_argument('--input_dir', type=str,
                        default='/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/bboxed/images',
                        help='Input directory containing .nii.gz files')
    parser.add_argument('--output_dir', type=str,
                        default='/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/near_format_data/images_256',
                        help='Output directory for .npy files')
    parser.add_argument('--target_size', type=int, nargs=3, default=[256, 256, 256],
                        help='Target size (D H W)')
    parser.add_argument('--sample_ids', type=str, default=None,
                        help='Comma-separated sample IDs to process (e.g., "1,2,3"). If None, process all')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    target_size = tuple(args.target_size)
    print(f"Resizing CT images to {target_size}")
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    
    # Get list of files to process
    if args.sample_ids:
        # Process specific sample IDs
        sample_ids = [int(x.strip()) for x in args.sample_ids.split(',')]
        input_files = [f"{sid}.nii.img.nii.gz" for sid in sample_ids]
    else:
        # Process all files
        input_files = [f for f in os.listdir(args.input_dir) if f.endswith('.nii.gz')]
        input_files = sorted(input_files, key=lambda x: int(x.split('.')[0]))
    
    print(f"Found {len(input_files)} files to process\n")
    
    # Process each file
    successful = 0
    failed = []
    
    for filename in tqdm(input_files, desc="Resizing CT images"):
        try:
            # Get sample ID
            sample_id = filename.split('.')[0]
            
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, f"{sample_id}.npy")
            
            # Skip if already exists
            if os.path.exists(output_path):
                continue
            
            # Resize
            orig_shape, new_shape = resize_ct_image(input_path, output_path, target_size)
            
            successful += 1
            
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            failed.append(filename)
    
    print(f"\n{'='*70}")
    print(f"Resize complete!")
    print(f"  Successful: {successful}/{len(input_files)}")
    if failed:
        print(f"  Failed: {len(failed)}")
        print(f"  Failed files: {failed}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
