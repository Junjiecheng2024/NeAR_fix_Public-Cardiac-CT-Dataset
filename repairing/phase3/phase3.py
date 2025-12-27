"""
Main script for Phase 3: Multi-class Fusion and Anatomical Correction.
Fuses single-class masks from Phase 2 and applies anatomical rules (PV-LA, LAA-LA, Coronary-Myo).
"""
import os
import numpy as np
import nibabel as nib
import cc3d
from scipy.ndimage import binary_dilation, distance_transform_edt
import argparse
from tqdm import tqdm
import glob
import multiprocessing
import time

# Priority (High to Low)
PRIORITY_ORDER = [
    9,  # Coronary
    10, # PV
    8,  # LAA
    3, 5, 2, 4, # LV, RV, LA, RA (Chambers)
    1,  # Myocardium
    6,  # Aorta
    7   # PA
]

CLASS_NAMES = {
    1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
    6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
}

def load_masks(case_id, data_root):
    """Load all 10 class masks for a given case_id."""
    masks = {}
    for class_id in range(1, 11):
        class_name = CLASS_NAMES[class_id].lower() # unify to lowercase, e.g. 'la'
        
        # 1. Try Phase 2 Output (_morph)
        path1 = os.path.join(data_root, f"{class_name}_morph", f"{case_id}_mask.npy")
        # 2. Try Phase 1 Output (_global)
        path2 = os.path.join(data_root, f"{class_name}_global", f"{case_id}_mask.npy")
        
        # Legacy compatibility (optional)
        path3 = os.path.join(data_root, f"{class_name}_global", f"{case_id}.npy")

        if os.path.exists(path1):
            masks[class_id] = np.load(path1)
        elif os.path.exists(path2):
            masks[class_id] = np.load(path2)
        elif os.path.exists(path3):
            masks[class_id] = np.load(path3)
        else:
            masks[class_id] = None 
            
    return masks

def resolve_priority_overlap(masks, shape=(256, 256, 256)):
    """Step 1: Apply priority fusion."""
    final_mask = np.zeros(shape, dtype=np.uint8)
    reversed_priority = PRIORITY_ORDER[::-1]
    
    for class_id in reversed_priority:
        mask = masks.get(class_id)
        if mask is not None:
            final_mask[mask > 0] = class_id
            
    return final_mask

def get_class_mask(final_mask, class_id):
    return (final_mask == class_id).astype(np.uint8)

def connect_structure_to_target(final_mask, source_cls, target_cls, max_dist=5, name=""):
    source_mask = get_class_mask(final_mask, source_cls)
    target_mask = get_class_mask(final_mask, target_cls)
    
    if source_mask.sum() == 0:
        return final_mask
    if target_mask.sum() == 0:
        final_mask[final_mask == source_cls] = 0
        return final_mask

    labels_out, N = cc3d.connected_components(source_mask, connectivity=26, return_N=True)
    dist_map = distance_transform_edt(1 - target_mask)
    
    new_source_mask = np.zeros_like(source_mask)
    
    for i in range(1, N + 1):
        component = (labels_out == i)
        min_d = dist_map[component].min()
        
        if min_d > max_dist:
            pass # Delete
        elif min_d > 0:
            # Connect
            dilated = binary_dilation(component, iterations=int(min_d) + 1)
            connection = np.logical_and(dilated, final_mask == 0)
            final_mask[connection] = source_cls
            new_source_mask[component] = 1
        else:
            new_source_mask[component] = 1
            
    final_mask[final_mask == source_cls] = 0
    final_mask[new_source_mask > 0] = source_cls
    
    return final_mask

def filter_floating_structures(final_mask, source_cls, target_mask, max_dist=6, name=""):
    source_mask = get_class_mask(final_mask, source_cls)
    
    if source_mask.sum() == 0:
        return final_mask
    if target_mask.sum() == 0:
        final_mask[final_mask == source_cls] = 0
        return final_mask
        
    labels_out, N = cc3d.connected_components(source_mask, connectivity=26, return_N=True)
    dist_map = distance_transform_edt(1 - target_mask)
    
    keep_mask = np.zeros_like(source_mask)
    
    for i in range(1, N + 1):
        component = (labels_out == i)
        min_d = dist_map[component].min()
        
        if min_d <= max_dist:
            keep_mask[component] = 1
            
    final_mask[final_mask == source_cls] = 0
    final_mask[keep_mask > 0] = source_cls
    
    return final_mask

def clean_chamber_fragments(final_mask, cls):
    mask = get_class_mask(final_mask, cls)
    if mask.sum() == 0:
        return final_mask
        
    labels_out, N = cc3d.connected_components(mask, connectivity=26, return_N=True)
    if N <= 1:
        return final_mask
        
    stats = cc3d.statistics(labels_out)
    counts = stats['voxel_counts'][1:]
    sorted_indices = np.argsort(counts)[::-1]
    
    largest_idx = sorted_indices[0] + 1
    largest_comp = (labels_out == largest_idx)
    
    keep_mask = np.zeros_like(mask)
    keep_mask[largest_comp] = 1
    
    dist_map = distance_transform_edt(1 - largest_comp)
    
    for i in range(1, len(sorted_indices)):
        idx = sorted_indices[i] + 1
        comp = (labels_out == idx)
        if dist_map[comp].min() < 3:
            keep_mask[comp] = 1
            
    final_mask[final_mask == cls] = 0
    final_mask[keep_mask > 0] = cls
    
    return final_mask

def enforce_anatomical_constraints(final_mask):
    # Rule 3 & 4: PV/LAA -> LA
    final_mask = connect_structure_to_target(final_mask, source_cls=10, target_cls=2, max_dist=5, name="PV-LA")
    final_mask = connect_structure_to_target(final_mask, source_cls=8, target_cls=2, max_dist=3, name="LAA-LA")
    
    # Rule 5: Coronary -> Myo/Ao
    myo_mask = get_class_mask(final_mask, 1)
    ao_mask = get_class_mask(final_mask, 6)
    target_mask = np.logical_or(myo_mask, ao_mask).astype(np.uint8)
    final_mask = filter_floating_structures(final_mask, source_cls=9, target_mask=target_mask, max_dist=6, name="Coronary-Myo/Ao")
    
    # Rule 1: Chamber Enclosure (Ensure Chambers are inside Myocardium + slight margin)
    # This prevents chambers from leaking into background where Myocardium doesn't exist
    # Only applies if Myocardium exists
    if myo_mask.sum() > 0:
        # Dilate Myocardium to form an envelope
        myo_envelope = binary_dilation(myo_mask, iterations=2)
        # Also include Aorta and PA in envelope to allow connection? 
        # Ideally Chambers connect to Ao/PA.
        # Let's add Ao and PA to the allowed region to be safe
        pa_mask = get_class_mask(final_mask, 7)
        allowed_region = np.logical_or(myo_envelope, ao_mask)
        allowed_region = np.logical_or(allowed_region, pa_mask)
        
        # FIX: Rule 1 should ONLY apply to Left Ventricle (LV).
        # Class 1 Myocardium is typically LV Myocardium. 
        # LA (2) and RA (4) are ABOVE it. RV (5) is BESIDE it.
        # Enforcing them to be inside Myo would delete them.
        for chamber_id in [3]: # LV only! 
            chamber_mask = get_class_mask(final_mask, chamber_id)
            # Find chamber parts outside allowed region
            leak = np.logical_and(chamber_mask, ~allowed_region)
            if leak.sum() > 0:
                final_mask[leak] = 0

    
    # Rule 2: Chamber Denoising
    for cls in [2, 3, 4, 5]:
        final_mask = clean_chamber_fragments(final_mask, cls)
        
    return final_mask

def process_case_worker(args):
    """Worker function for multiprocessing."""
    case_id, phase2_dir, output_dir = args
    
    try:
        # 1. Load
        masks = load_masks(case_id, phase2_dir)
        
        if all(m is None for m in masks.values()):
            return f"Skipped {case_id} (No masks)"
            
        for k in masks:
            if masks[k] is None:
                masks[k] = np.zeros((256, 256, 256), dtype=np.uint8)
                
        # 2. Priority Fusion
        fused_mask = resolve_priority_overlap(masks)
        
        # 3. Anatomy Rules
        final_mask = enforce_anatomical_constraints(fused_mask)
        
        # 4. Save
        np.save(os.path.join(output_dir, f"{case_id}_phase3.npy"), final_mask)
        
        affine = np.eye(4)
        nii = nib.Nifti1Image(final_mask, affine)
        nib.save(nii, os.path.join(output_dir, f"{case_id}_phase3.nii.gz"))
        
        return None # Success
    except Exception as e:
        return f"Error {case_id}: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Root dataset dir containing {class}_morph or {class}_global")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find cases from one of the class directories, e.g. LV
    lv_dir = os.path.join(args.data_root, "lv_morph")
    if not os.path.exists(lv_dir):
         lv_dir = os.path.join(args.data_root, "lv_global")
    
    if not os.path.exists(lv_dir):
        print(f"Error: Could not find LV directory in {args.data_root}")
        return

    files = glob.glob(os.path.join(lv_dir, "*_mask.npy"))
    # Fallback to .npy
    if not files:
        files = glob.glob(os.path.join(lv_dir, "*.npy"))
        
    case_ids = [os.path.basename(f).split('_mask')[0] for f in files if "_mask" in f]
    if not case_ids:
         case_ids = [os.path.basename(f).split('.')[0] for f in files]
    
    # Remove duplicates
    case_ids = sorted(list(set(case_ids)))
    
    print(f"Found {len(case_ids)} cases to process from {lv_dir}")
    
    # Prepare args for worker
    tasks = [(cid, args.data_root, args.output_dir) for cid in case_ids]
    
    # Multiprocessing
    num_workers = min(16, multiprocessing.cpu_count())
    print(f"Starting pool with {num_workers} workers...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_case_worker, tasks), total=len(tasks)):
            if res:
                print(res)
        
    print("Phase 3 Complete.")

if __name__ == "__main__":
    main()
