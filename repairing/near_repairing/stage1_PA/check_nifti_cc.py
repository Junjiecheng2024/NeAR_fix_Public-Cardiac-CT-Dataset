
import nibabel as nib
import numpy as np
from scipy.ndimage import label

def check_nifti(sample_id):
    path = f'/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset_backup/dataset/original/segmentations/{sample_id}.nii.img.nii.gz'
    print(f"Loading {path}...")
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    print(f"Shape: {data.shape}")
    
    # Extract PA (Class 7)
    # Note: data might be float, round it
    data = np.rint(data).astype(np.uint8)
    pa_mask = (data == 7).astype(np.uint8)
    
    print(f"PA Voxels: {pa_mask.sum()}")
    
    labeled, n_components = label(pa_mask)
    print(f"Number of Components: {n_components}")
    
    if n_components > 0:
        sizes = [np.sum(labeled == i) for i in range(1, n_components + 1)]
        sizes.sort(reverse=True)
        print(f"Sizes: {sizes}")

if __name__ == "__main__":
    check_nifti('106')
