"""
3D visualization script for generating surface comparison figures.
Uses marching cubes to extract surface meshes and renders them with matplotlib.
"""
import os
import numpy as np
import nibabel as nib
from pathlib import Path
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(os.environ.get("NEAR_DATA_ROOT", REPO_ROOT / "dataset"))

# Class definitions
CLASS_NAMES = {
    1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
    6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
}

# Color definitions (RGB, 0-1)
CLASS_COLORS = {
    1: (0.8, 0.2, 0.2),    # Myocardium - Red
    2: (0.2, 0.8, 0.2),    # LA - Green
    3: (0.2, 0.2, 0.8),    # LV - Blue
    4: (0.9, 0.9, 0.2),    # RA - Yellow
    5: (0.2, 0.9, 0.9),    # RV - Cyan
    6: (0.9, 0.2, 0.9),    # Aorta - Magenta
    7: (1.0, 0.5, 0.0),    # PA - Orange
    8: (0.6, 0.0, 0.8),    # LAA - Purple
    9: (0.8, 0.0, 0.0),    # Coronary - Dark Red
    10: (0.0, 0.6, 0.6),   # PV - Teal
}

def load_mask(path):
    """Load a mask file."""
    if path.endswith('.npy'):
        return np.load(path)
    elif path.endswith('.nii.gz') or path.endswith('.nii'):
        return np.asanyarray(nib.load(path).dataobj)
    return None

def extract_surface(mask, class_id, step_size=2):
    """
    Extract a surface mesh with marching cubes.
    step_size: subsampling stride; larger values are faster but coarser.
    """
    binary_mask = (mask == class_id).astype(np.float32)
    if binary_mask.sum() == 0:
        return None, None
    
    try:
        # Downsample to accelerate mesh extraction
        if step_size > 1:
            binary_mask = binary_mask[::step_size, ::step_size, ::step_size]
        
        verts, faces, normals, values = measure.marching_cubes(binary_mask, level=0.5)
        verts = verts * step_size  # Restore the original scale
        return verts, faces
    except:
        return None, None

def extract_surface_binary(binary_mask, step_size=2):
    """
    Extract a surface from a binary mask (used for single-class Phase1 outputs).
    """
    binary_mask = (binary_mask > 0).astype(np.float32)
    if binary_mask.sum() == 0:
        return None, None
    
    try:
        if step_size > 1:
            binary_mask = binary_mask[::step_size, ::step_size, ::step_size]
        
        verts, faces, normals, values = measure.marching_cubes(binary_mask, level=0.5)
        verts = verts * step_size
        return verts, faces
    except:
        return None, None

def plot_3d_comparison(orig_mask, case_id, output_path, 
                       classes_to_show=[8, 9, 10], alpha=0.7, phase1_masks=None):
    """
    Generate a 3D comparison figure: Original -> Phase1 (two columns).
    classes_to_show: list of class IDs to display.
    phase1_masks: dict {class_id: mask} containing Phase1 single-class masks.
    """
    # Always use two columns: Original and Phase1
    ncols = 2
    
    fig = plt.figure(figsize=(8 * ncols, 8))
    
    # Original data
    ax1 = fig.add_subplot(1, ncols, 1, projection='3d')
    ax1.set_title(f'Original (Case {case_id})', fontsize=14)
    
    # Phase1 data
    ax_p1 = fig.add_subplot(1, ncols, 2, projection='3d')
    ax_p1.set_title(f'After NeAR (Phase1)', fontsize=14)
    
    # Store bounds so both panels share the same view
    all_verts = []
    
    for class_id in classes_to_show:
        color = CLASS_COLORS.get(class_id, (0.5, 0.5, 0.5))
        name = CLASS_NAMES.get(class_id, f'Class{class_id}')
        
        # Original data
        verts1, faces1 = extract_surface(orig_mask, class_id)
        if verts1 is not None:
            mesh1 = Poly3DCollection(verts1[faces1], alpha=alpha)
            mesh1.set_facecolor(color)
            mesh1.set_edgecolor('none')
            ax1.add_collection3d(mesh1)
            all_verts.append(verts1)
        
        # Phase1 data (binary single-class mask; any nonzero voxel belongs to the class)
        if phase1_masks and class_id in phase1_masks and phase1_masks[class_id] is not None:
            p1_mask = phase1_masks[class_id]
            verts_p1, faces_p1 = extract_surface_binary(p1_mask)
            if verts_p1 is not None:
                mesh_p1 = Poly3DCollection(verts_p1[faces_p1], alpha=alpha)
                mesh_p1.set_facecolor(color)
                mesh_p1.set_edgecolor('none')
                ax_p1.add_collection3d(mesh_p1)
                all_verts.append(verts_p1)
    
    # Use a shared coordinate range
    axes_list = [ax1, ax_p1]
    if all_verts:
        all_verts = np.vstack(all_verts)
        max_range = np.array([all_verts[:, 0].max() - all_verts[:, 0].min(),
                              all_verts[:, 1].max() - all_verts[:, 1].min(),
                              all_verts[:, 2].max() - all_verts[:, 2].min()]).max() / 2.0
        mid_x = (all_verts[:, 0].max() + all_verts[:, 0].min()) * 0.5
        mid_y = (all_verts[:, 1].max() + all_verts[:, 1].min()) * 0.5
        mid_z = (all_verts[:, 2].max() + all_verts[:, 2].min()) * 0.5
        
        for ax in axes_list:
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
    
    # Add legend
    legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLORS[c]) 
                      for c in classes_to_show if c in CLASS_COLORS]
    legend_labels = [CLASS_NAMES[c] for c in classes_to_show if c in CLASS_NAMES]
    fig.legend(legend_patches, legend_labels, loc='lower center', ncol=len(classes_to_show))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_single_class_comparison(orig_mask, case_id, class_id, output_path, phase1_mask=None):
    """
    Detailed 3D comparison for a single class (two columns: Original -> Phase1).
    phase1_mask: binary Phase1 NeAR output for one class.
    """
    ncols = 2
    
    fig = plt.figure(figsize=(7 * ncols, 6))
    
    class_name = CLASS_NAMES.get(class_id, f'Class{class_id}')
    color = CLASS_COLORS.get(class_id, (0.5, 0.5, 0.5))
    edge_color = (0.8, 0.1, 0.1, 0.15)  # Red edges
    
    # Original
    ax1 = fig.add_subplot(1, ncols, 1, projection='3d')
    ax1.set_title(f'Original {class_name}', fontsize=14)
    
    # Phase1
    ax_p1 = fig.add_subplot(1, ncols, 2, projection='3d')
    ax_p1.set_title(f'After NeAR (Phase1)', fontsize=14)

    all_verts = []
    
    # Original
    verts1, faces1 = extract_surface(orig_mask, class_id, step_size=1)
    if verts1 is not None:
        mesh1 = Poly3DCollection(verts1[faces1], alpha=0.8)
        mesh1.set_facecolor(color)
        mesh1.set_edgecolor(edge_color)
        ax1.add_collection3d(mesh1)
        all_verts.append(verts1)
    else:
        ax1.text(0.5, 0.5, 0.5, 'Empty', ha='center', transform=ax1.transAxes)
    
    # Phase1 (binary single-class mask)
    if phase1_mask is not None:
        verts_p1, faces_p1 = extract_surface_binary(phase1_mask, step_size=1)
        if verts_p1 is not None:
            mesh_p1 = Poly3DCollection(verts_p1[faces_p1], alpha=0.8)
            mesh_p1.set_facecolor(color)
            mesh_p1.set_edgecolor(edge_color)
            ax_p1.add_collection3d(mesh_p1)
            all_verts.append(verts_p1)
        else:
            ax_p1.text(0.5, 0.5, 0.5, 'Empty', ha='center', transform=ax_p1.transAxes)
    else:
        ax_p1.text(0.5, 0.5, 0.5, 'Missing', ha='center', transform=ax_p1.transAxes)
    
    # Shared coordinate range
    axes_list = [ax1, ax_p1]
    if all_verts:
        all_verts = np.vstack(all_verts)
        max_range = np.array([all_verts[:, 0].max() - all_verts[:, 0].min(),
                              all_verts[:, 1].max() - all_verts[:, 1].min(),
                              all_verts[:, 2].max() - all_verts[:, 2].min()]).max() / 2.0
        mid_x = (all_verts[:, 0].max() + all_verts[:, 0].min()) * 0.5
        mid_y = (all_verts[:, 1].max() + all_verts[:, 1].min()) * 0.5
        mid_z = (all_verts[:, 2].max() + all_verts[:, 2].min()) * 0.5
        
        for ax in axes_list:
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.suptitle(f'Case {case_id} - {class_name} Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='3D Visualization of Repair Results')
    parser.add_argument('--case_ids', type=str, default='1,10,100',
                        help='Comma-separated case IDs to visualize, e.g., "1,10,100"')
    parser.add_argument('--data_root', type=str, 
                        default=str(DEFAULT_DATA_ROOT),
                        help='Root directory of dataset')
    parser.add_argument('--output_dir', type=str, default='./vis_3d',
                        help='Output directory for visualizations')
    parser.add_argument('--classes', type=str, default='8,9,10',
                        help='Classes for overview comparison (default: LAA, Coronary, PV)')
    parser.add_argument('--single_classes', type=str, default='1,2,3,4,5,6,7,8,9,10',
                        help='Classes for single-class comparison (default: all 10 classes)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    case_ids = [x.strip() for x in args.case_ids.split(',')]
    classes = [int(x.strip()) for x in args.classes.split(',')]
    single_classes = [int(x.strip()) for x in args.single_classes.split(',')]
    
    data_root = Path(args.data_root)
    gt_dir = data_root / "original" / "segmentations"
    
    print(f"Generating 3D visualizations for cases: {case_ids}")
    print(f"Classes to show: {[CLASS_NAMES.get(c, c) for c in classes]}")
    print(f"Data root: {data_root}")
    
    for case_id in case_ids:
        print(f"\nProcessing Case {case_id}...")
        
        # Load Ground Truth (Original segmentation)
        gt_paths = [
            gt_dir / f"{case_id}.nii.img.nii.gz",
            gt_dir / f"{case_id}.nii.gz",
        ]
        
        orig_mask = None
        for p in gt_paths:
            if p.exists():
                orig_mask = load_mask(str(p))
                print(f"  Loaded GT: {p}")
                break
        
        if orig_mask is None:
            print(f"  Warning: GT not found for case {case_id}")
            continue
        
        # Load Phase1 masks for each class
        phase1_masks = {}
        all_class_ids = list(set(classes + single_classes))
        
        for class_id in all_class_ids:
            class_name = CLASS_NAMES.get(class_id, f'Class{class_id}').lower()
            
            p1_paths = [
                data_root / f"{class_name}_global" / f"{case_id}_mask.npy",
                data_root / f"{class_name}_morph" / f"{case_id}_mask.npy",
            ]
            
            for p in p1_paths:
                if p.exists():
                    phase1_masks[class_id] = load_mask(str(p))
                    print(f"  Loaded Phase1 {class_name}: {p.name}")
                    break
        
        # Generate comparison figure (GT -> Phase1)
        output_path = os.path.join(args.output_dir, f"case_{case_id}_comparison.png")
        plot_3d_comparison(orig_mask, case_id, output_path, classes, 
                          phase1_masks=phase1_masks)
        
        # Generate single-class comparison for each class
        for class_id in single_classes:
            class_name = CLASS_NAMES.get(class_id, f'Class{class_id}')
            output_path = os.path.join(args.output_dir, 
                                       f"case_{case_id}_{class_name}_3d.png")
            p1_mask = phase1_masks.get(class_id, None)
            plot_single_class_comparison(orig_mask, case_id, class_id, 
                                        output_path, phase1_mask=p1_mask)
    
    print(f"\nDone! Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()

