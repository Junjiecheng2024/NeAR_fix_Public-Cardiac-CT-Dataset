import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import random
from tqdm import tqdm
from scipy.ndimage import zoom

# Class mapping
CLASS_NAMES = {
    0: "Background",
    1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
    6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
}

# Define a fixed color map for consistency
COLORS = [
    (0, 0, 0, 0),       # 0: Background (Transparent)
    (1, 0, 0, 1),       # 1: Myo (Red)
    (0, 1, 0, 1),       # 2: LA (Green)
    (0, 0, 1, 1),       # 3: LV (Blue)
    (1, 1, 0, 1),       # 4: RA (Yellow)
    (0, 1, 1, 1),       # 5: RV (Cyan)
    (1, 0, 1, 1),       # 6: Aorta (Magenta)
    (1, 0.5, 0, 1),     # 7: PA (Orange)
    (0.5, 0, 1, 1),     # 8: LAA (Purple)
    (0.5, 0, 0, 1),     # 9: Coronary (Dark Red)
    (0, 0.5, 0.5, 1)    # 10: PV (Teal)
]
CMAP = mcolors.ListedColormap(COLORS)
BOUNDS = list(range(12))
NORM = mcolors.BoundaryNorm(BOUNDS, CMAP.N)

def load_volume(path):
    if not os.path.exists(path):
        return None
    if path.endswith('.npy'):
        return np.load(path)
    elif path.endswith('.nii.gz'):
        import nibabel as nib
        return np.asanyarray(nib.load(path).dataobj)
    return None

def visualize_case(case_id, s3_path, gt_path, img_path, output_dir):
    s3_mask = load_volume(s3_path)
    gt_mask = load_volume(gt_path)
    ct_img = load_volume(img_path)
    
    if s3_mask is None or gt_mask is None:
        print(f"Error loading masks for {case_id}")
        return

    # Handle CT Image
    if ct_img is None:
        print(f"Warning: CT image not found for {case_id}, using blank")
        ct_img = np.zeros_like(s3_mask)
    else:
        # Orientation fix based on inference_and_evaluate.py
        # Image is likely (X, Y, Z), Mask is (Z, Y, X)
        if ct_img.ndim == 3 and ct_img.shape == s3_mask.shape:
             # Heuristic: Check if transpose improves correlation? 
             # Or just trust the previous script's logic: transpose(2, 1, 0)
             # Let's try to be robust.
             pass
        
        # If CT is (X, Y, Z) and Mask is (Z, Y, X), we need to transpose CT to (Z, Y, X)
        # Standard medical imaging often has Z as first dim in python if loaded from NIfTI, 
        # but .npy might be saved differently.
        # Let's assume the previous script was correct and transpose CT.
        ct_img = ct_img.transpose(2, 1, 0)

    # Find best slice (Max Area in S3 mask)
    # We focus on Axial (Z) slices as they are standard for CT
    axis = 0 # Z-axis
    area_per_slice = np.sum(s3_mask > 0, axis=(1, 2))
    slice_idx = np.argmax(area_per_slice)
    
    # Extract slices
    s3_slice = s3_mask[slice_idx, :, :]
    gt_slice = gt_mask[slice_idx, :, :]
    ct_slice = ct_img[slice_idx, :, :]
    
    # Normalize CT for display
    # CT is usually Hounsfield Units (-1000 to 3000). 
    # Clip to soft tissue window (-160 to 240) or similar for better contrast
    # Or just min-max if unknown range.
    # NeAR data is likely pre-normalized to [-1, 1] or [0, 1].
    if ct_img.min() < -100: # HU likely
        ct_slice = np.clip(ct_slice, -160, 240)
        ct_slice = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min())
    else:
        # Already normalized
        pass

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. CT Only
    axes[0].imshow(ct_slice, cmap='gray')
    axes[0].set_title(f"Original CT\nSlice {slice_idx}")
    axes[0].axis('off')
    
    # 2. GT Overlay
    axes[1].imshow(ct_slice, cmap='gray')
    axes[1].imshow(gt_slice, cmap=CMAP, norm=NORM, alpha=0.5, interpolation='nearest')
    axes[1].set_title(f"Original Overlay\nSlice {slice_idx}")
    axes[1].axis('off')
    
    # 3. S3 Overlay
    axes[2].imshow(ct_slice, cmap='gray')
    axes[2].imshow(s3_slice, cmap=CMAP, norm=NORM, alpha=0.5, interpolation='nearest')
    axes[2].set_title(f"Repaired Overlay\nSlice {slice_idx}")
    axes[2].axis('off')
    
    # Legend
    patches = [plt.Rectangle((0,0),1,1, color=COLORS[i]) for i in range(1, 11)]
    labels = [CLASS_NAMES[i] for i in range(1, 11)]
    fig.legend(patches, labels, loc='center right', title="Classes")
    
    plt.suptitle(f"Case: {case_id}", fontsize=16)
    plt.subplots_adjust(right=0.9)
    
    save_path = os.path.join(output_dir, f"{case_id}_vis_overlay.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--img_dir", required=True, help="Directory containing CT images")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(args.stage3_dir) if f.endswith('.npy')]
    case_ids = [f.split('_stage3')[0] for f in files]
    
    # Randomly select samples
    selected_cases = random.sample(case_ids, min(len(case_ids), args.num_samples))
    selected_cases.sort()
    
    print(f"Generating overlay visualizations for {len(selected_cases)} cases...")
    
    for case_id in tqdm(selected_cases):
        s3_path = os.path.join(args.stage3_dir, f"{case_id}_stage3.npy")
        gt_path = os.path.join(args.gt_dir, f"{case_id}.npy")
        img_path = os.path.join(args.img_dir, f"{case_id}.npy")
        
        visualize_case(case_id, s3_path, gt_path, img_path, args.output_dir)
        
    print(f"Done. Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
