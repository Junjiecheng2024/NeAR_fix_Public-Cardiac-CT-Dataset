"""
Script to generate visualization comparisons (CT overlay) between Original and Phase 3.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import random
from tqdm import tqdm
from scipy.ndimage import zoom
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(os.environ.get("NEAR_DATA_ROOT", REPO_ROOT / "dataset"))

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

    # Resize Phase3 mask (256³) to match GT resolution
    if s3_mask.shape != gt_mask.shape:
        zoom_factors = np.array(gt_mask.shape) / np.array(s3_mask.shape)
        s3_mask = zoom(s3_mask, zoom_factors, order=0)  # Nearest neighbor for labels
    
    # Handle CT Image
    if ct_img is None:
        print(f"Warning: CT image not found for {case_id}, using blank")
        ct_img = np.zeros_like(gt_mask)
    
    # Make sure all three have the same shape
    target_shape = gt_mask.shape
    
    if ct_img.shape != target_shape:
        # Try transpose first (common orientation issue)
        if ct_img.shape == target_shape[::-1]:
            ct_img = ct_img.transpose(2, 1, 0)
        else:
            # Resize CT
            zoom_factors = np.array(target_shape) / np.array(ct_img.shape)
            ct_img = zoom(ct_img, zoom_factors, order=1)

    # Find best slice (Max Area in Phase3 mask)
    axis = 0  # Z-axis
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
    parser.add_argument("--data_root", type=str, 
                        default=str(DEFAULT_DATA_ROOT),
                        help="Root directory of dataset")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    phase3_dir = os.path.join(args.data_root, "repaired_phase3")
    gt_dir = os.path.join(args.data_root, "original", "segmentations")
    img_dir = os.path.join(args.data_root, "original", "images")
    
    # Find Phase3 files
    files = [f for f in os.listdir(phase3_dir) if f.endswith('.npy') or f.endswith('.nii.gz')]
    case_ids = []
    for f in files:
        if '_phase3' in f:
            case_ids.append(f.split('_phase3')[0])
        else:
            case_ids.append(f.split('.')[0])
    case_ids = list(set(case_ids))
    
    # Randomly select samples
    selected_cases = random.sample(case_ids, min(len(case_ids), args.num_samples))
    selected_cases.sort()
    
    print(f"Generating overlay visualizations for {len(selected_cases)} cases...")
    print(f"Phase3 dir: {phase3_dir}")
    print(f"GT dir: {gt_dir}")
    print(f"Image dir: {img_dir}")
    
    for case_id in tqdm(selected_cases):
        # Phase 3 mask
        s3_candidates = [
            os.path.join(phase3_dir, f"{case_id}_phase3.npy"),
            os.path.join(phase3_dir, f"{case_id}_phase3.nii.gz"),
        ]
        s3_path = next((p for p in s3_candidates if os.path.exists(p)), None)
        
        # GT mask
        gt_candidates = [
            os.path.join(gt_dir, f"{case_id}.nii.img.nii.gz"),
            os.path.join(gt_dir, f"{case_id}.nii.gz"),
        ]
        gt_path = next((p for p in gt_candidates if os.path.exists(p)), None)
        
        # CT image
        img_candidates = [
            os.path.join(img_dir, f"{case_id}.nii.img.nii.gz"),
            os.path.join(img_dir, f"{case_id}.nii.gz"),
            os.path.join(img_dir, f"{case_id}.npy"),
        ]
        img_path = next((p for p in img_candidates if os.path.exists(p)), None)
        
        if s3_path and gt_path:
            visualize_case(case_id, s3_path, gt_path, img_path, args.output_dir)
        else:
            print(f"Skipping {case_id}: missing files")
        
    print(f"Done. Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
