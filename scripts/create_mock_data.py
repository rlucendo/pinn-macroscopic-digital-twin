import os
import numpy as np
import nibabel as nib

def create_sphere_mask(shape, center, radius):
    """Creates a 3D boolean mask of a sphere."""
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    return distance <= radius

def generate_mock_nifti():
    """
    Generates dummy 3D NIfTI files to test the GlioSim pipeline without 
    needing to download the massive BraTS or UPenn-GBM datasets.
    """
    data_dir = "data/"
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating synthetic medical data for Smoke Testing...")
    
    # Define a small spatial resolution to run fast on CPU/Local GPU
    # Standard BraTS is 240x240x155, but we use 32x32x32 for instant testing
    spatial_shape = (32, 32, 32)
    center = (16, 16, 16)
    
    # 1. Generate T0 MRI (4 modalities: FLAIR, T1w, T1gd, T2w)
    # Shape for NIfTI should be (X, Y, Z, Channels)
    mri_data = np.zeros(spatial_shape + (4,), dtype=np.float32)
    brain_mask = create_sphere_mask(spatial_shape, center, radius=12)
    
    for c in range(4):
        # Fill the "brain" area with random intensities, leave background as 0
        noise = np.random.rand(*spatial_shape) * 100
        mri_data[..., c] = np.where(brain_mask, noise, 0.0)
        
    # Standard identity affine matrix (1mm isotropic spacing)
    affine = np.eye(4)
    
    # 2. Generate Tn Ground Truth Mask (The Tumor after 30 days)
    # Shape: (X, Y, Z) - A smaller sphere inside the brain
    tumor_mask = create_sphere_mask(spatial_shape, center, radius=5)
    mask_data = tumor_mask.astype(np.float32)
    
    # 3. Save the mocked files (we simulate 10 patients pointing to the same file)
    for i in range(10):
        mri_path = os.path.join(data_dir, f"patient_{i}_T0.nii.gz")
        mask_path = os.path.join(data_dir, f"patient_{i}_Tn.nii.gz")
        
        nib.save(nib.Nifti1Image(mri_data, affine), mri_path)
        nib.save(nib.Nifti1Image(mask_data, affine), mask_path)
        
    print(f"Generated 10 mock patients in '{data_dir}'")
    print(f" - MRI Shape: {mri_data.shape}")
    print(f" - Mask Shape: {mask_data.shape}")

if __name__ == "__main__":
    generate_mock_nifti()