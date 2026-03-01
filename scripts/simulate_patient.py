import argparse
import logging
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from monai.transforms import Resize

# Domain imports
from src.models.differentiable_solver import DifferentiableEulerSolver
from src.physics.fisher_kolmogorov import FisherKolmogorovPDE
from src.models.unet_baseline import BaselineStateExtractor
from src.models.pinn_simulator import MacroscopicDigitalTwin
# Assuming the training module is in train_simulator.py
from scripts.train_simulator import HardPhysicsGlioSimSystem

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DigitalTwin_Inference")

def _discretize_to_clinical_classes(u_pred: np.ndarray) -> np.ndarray:
    """
    Maps continuous tumor density to Slicer3D compatible discrete segments.
    Values are theoretical placeholders and should be calibrated clinically.
    - Class 0: Background
    - Class 1: Edema / Infiltration (0.05 < u <= 0.20)
    - Class 2: Enhancing Core (0.20 < u <= 0.80)
    - Class 3: Necrotic Core (u > 0.80)
    """
    segmentation = np.zeros_like(u_pred, dtype=np.uint8)
    
    segmentation[(u_pred > 0.05) & (u_pred <= 0.20)] = 1
    segmentation[(u_pred > 0.20) & (u_pred <= 0.80)] = 2
    segmentation[u_pred > 0.80] = 3
    
    return segmentation

def run_simulation(
    checkpoint_path: str,
    image_t0_path: str,
    mask_t0_path: str,
    output_dir: str,
    target_days: float
):
    """Executes the forward simulation for a single patient."""
    logger.info("Initializing inference pipeline...", extra={"checkpoint": checkpoint_path})
    
    # 1. Load Model from Checkpoint
    try:
        model = HardPhysicsGlioSimSystem.load_from_checkpoint(checkpoint_path)
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
        raise

    # 2. Load and Preprocess Data (Mirroring DataModule logic)
    img_nifti = nib.load(image_t0_path)
    mask_nifti = nib.load(mask_t0_path)
    
    original_affine = img_nifti.affine
    original_shape = img_nifti.shape
    
    img_data = img_nifti.get_fdata()
    mask_data = mask_nifti.get_fdata()
    
    # Z-Score Normalization
    mask_bg = img_data > 0
    if mask_bg.sum() > 0:
        img_data = (img_data - img_data[mask_bg].mean()) / (img_data[mask_bg].std() + 1e-8)
        img_data[~mask_bg] = 0.0
        
    img_tensor = torch.from_numpy(img_data).float().unsqueeze(0).unsqueeze(0) # [1, 1, D, H, W]
    mask_tensor = torch.from_numpy(mask_data).float().unsqueeze(0).unsqueeze(0)
    
    # Spatial alignment to UNet topology
    resizer = Resize(spatial_size=(96, 96, 96), mode="trilinear")
    mask_resizer = Resize(spatial_size=(96, 96, 96), mode="nearest")
    
    img_resized = resizer(img_tensor).to(model.device)
    mask_resized = mask_resizer(mask_tensor).to(model.device)
    delta_t = torch.tensor([target_days], dtype=torch.float32).to(model.device)

    logger.info("Starting temporal integration.", extra={"target_days": target_days})

    # 3. Execute Hard Physics Simulation
    with torch.no_grad():
        u_pred_tn, parametric_maps = model(img_resized, mask_resized, delta_t)
        
    # 4. Post-process and Restore original geometry
    # We must upsample the prediction back to the patient's native resolution
    restore_resizer = Resize(spatial_size=original_shape, mode="trilinear")
    u_pred_restored = restore_resizer(u_pred_tn.cpu()).squeeze().numpy()
    
    D_map_restored = restore_resizer(parametric_maps["diffusion_D"].cpu()).squeeze().numpy()
    rho_map_restored = restore_resizer(parametric_maps["proliferation_rho"].cpu()).squeeze().numpy()

    # Apply clinical discretization
    clinical_segmentation = _discretize_to_clinical_classes(u_pred_restored)

    # 5. Export NIfTI files for 3D Slicer
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    patient_id = Path(image_t0_path).name.split("_")[0]
    
    nib.save(nib.Nifti1Image(clinical_segmentation, original_affine), out_path / f"{patient_id}_simulated_day_{int(target_days)}_classes.nii.gz")
    nib.save(nib.Nifti1Image(u_pred_restored, original_affine), out_path / f"{patient_id}_simulated_day_{int(target_days)}_density.nii.gz")
    nib.save(nib.Nifti1Image(D_map_restored, original_affine), out_path / f"{patient_id}_param_D.nii.gz")
    nib.save(nib.Nifti1Image(rho_map_restored, original_affine), out_path / f"{patient_id}_param_rho.nii.gz")

    logger.info("Simulation completed and exported successfully.", extra={"output_dir": str(out_path)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Glioblastoma Growth")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .ckpt file")
    parser.add_argument("--image_t0", type=str, required=True, help="Path to T0 FLAIR MRI")
    parser.add_argument("--mask_t0", type=str, required=True, help="Path to T0 Segmentation Mask")
    parser.add_argument("--output_dir", type=str, default="simulations/output", help="Directory to save NIfTI results")
    parser.add_argument("--target_days", type=float, default=60.0, help="Days to simulate into the future")
    
    args = parser.parse_args()
    run_simulation(args.checkpoint, args.image_t0, args.mask_t0, args.output_dir, args.target_days)