import argparse
import logging
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from monai.transforms import Resize

# Core architecture imports
from scripts.train_simulator import SelfSupervisedGlioSim

# Configure professional structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DigitalTwin_Inference")

def _discretize_to_clinical_classes(u_pred: np.ndarray) -> np.ndarray:
    """
    Translates continuous tumor density into BraTS/UPENN discrete segments.
    1: Necrosis, 2: Edema/Infiltrating, 4: Enhancing Core.
    """
    segmentation = np.zeros_like(u_pred, dtype=np.uint8)
    
    # Edema (Infiltrating region, low to moderate density)
    segmentation[(u_pred > 0.05) & (u_pred <= 0.20)] = 2
    
    # Enhancing Core (High density)
    segmentation[(u_pred > 0.20) & (u_pred <= 0.80)] = 4
    
    # Necrotic Core (Hypoxic inner region, max density)
    segmentation[u_pred > 0.80] = 1
    
    return segmentation

def run_simulation(
    checkpoint_path: str,
    data_dir: str,
    patient_id: str,
    output_dir: str,
    target_days: float
):
    """Executes the forward simulation using the 4-channel Amortized model."""
    logger.info("Initializing inference pipeline...", extra={"checkpoint": checkpoint_path})
    
    # --- PYTORCH 2.6+ SECURITY PATCH ---
    import monai
    try:
        torch.serialization.add_safe_globals([monai.utils.enums.TraceKeys])
    except AttributeError:
        pass # Fallback for older PyTorch versions
    # -----------------------------------
    
    try:
        # Pytorch Lightning will now allow the MONAI TraceKeys to pass
        model = SelfSupervisedGlioSim.load_from_checkpoint(checkpoint_path)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
        raise

    # 1. Locate and Load Multimodal Data
    images_dir = Path(data_dir) / "images"
    masks_dir = Path(data_dir) / "masks"
    
    modality_suffixes = ["FLAIR", "T1", "T1GD", "T2"]
    modalities = []
    original_affine = None
    original_shape = None
    
    logger.info(f"Loading 4-channel structural MRI for patient {patient_id}...")
    
    for mod in modality_suffixes:
        # Glob search to handle slight naming variations
        matches = list(images_dir.glob(f"*{patient_id}*{mod}*"))
        if not matches:
            logger.error(f"Missing {mod} modality for patient {patient_id}")
            raise FileNotFoundError(f"Missing modality: {mod}")
            
        nifti = nib.load(matches[0])
        
        if original_affine is None:
            original_affine = nifti.affine
            original_shape = nifti.shape[:3]
            
        img = nifti.get_fdata()
        
        # Z-Score Normalization
        mask_bg = img > 0
        if mask_bg.sum() > 0:
            img = (img - img[mask_bg].mean()) / (img[mask_bg].std() + 1e-8)
            img[~mask_bg] = 0.0
            
        modalities.append(torch.from_numpy(img).float().unsqueeze(0))

    # Shape: [4, D, H, W] -> Eliminamos el .unsqueeze(0) extra aquí
    multi_channel_img = torch.cat(modalities, dim=0)
    
    # 2. Locate and Load T0 Mask (The starting point for the future simulation)
    mask_matches = list(masks_dir.glob(f"*{patient_id}*"))
    if not mask_matches:
        raise FileNotFoundError(f"Missing T0 mask for patient {patient_id}")
        
    mask_nifti = nib.load(mask_matches[0])
    raw_mask = mask_nifti.get_fdata()
    
    # Translate to density
    density_map = np.zeros_like(raw_mask, dtype=np.float32)
    density_map[(raw_mask == 1) | (raw_mask == 4) | (raw_mask == 3)] = 1.0
    density_map[raw_mask == 2] = 0.5
    
    # Shape: [1, D, H, W] -> Eliminamos un .unsqueeze(0) para que MONAI lo entienda
    mask_tensor = torch.from_numpy(density_map).unsqueeze(0)

    # 3. Spatial Alignment to UNet Topology
    target_shape = (96, 96, 96)
    resizer = Resize(spatial_size=target_shape, mode="trilinear")
    mask_resizer = Resize(spatial_size=target_shape, mode="nearest")
    
    # Redimensionamos primero y LUEGO añadimos el Batch dimension (.unsqueeze(0))
    img_resized = resizer(multi_channel_img).unsqueeze(0).to(device)
    mask_resized = mask_resizer(mask_tensor).unsqueeze(0).to(device)
    delta_t = torch.tensor([target_days], dtype=torch.float32).to(device)

    logger.info("Executing Hard Physics Integration.", extra={"target_days": target_days})
    
    # 4. Predict D, rho and integrate PDE over time
    with torch.no_grad():
        u_pred_tn, parametric_maps = model.digital_twin(img_resized, mask_resized, delta_t)

    # 5. Restore Original Geometry and Discretize
    logger.info("Restoring original patient geometry and generating Slicer3D segments.")
    restore_resizer = Resize(spatial_size=original_shape, mode="trilinear")
    
    u_pred_restored = restore_resizer(u_pred_tn.cpu()).squeeze().numpy()
    D_map_restored = restore_resizer(parametric_maps["diffusion_D"].cpu()).squeeze().numpy()
    rho_map_restored = restore_resizer(parametric_maps["proliferation_rho"].cpu()).squeeze().numpy()

    clinical_segmentation = _discretize_to_clinical_classes(u_pred_restored)

    # 6. Export Results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    nib.save(nib.Nifti1Image(clinical_segmentation, original_affine), out_path / f"{patient_id}_simulated_day_{int(target_days)}_classes.nii.gz")
    nib.save(nib.Nifti1Image(u_pred_restored, original_affine), out_path / f"{patient_id}_simulated_day_{int(target_days)}_density.nii.gz")
    nib.save(nib.Nifti1Image(D_map_restored, original_affine), out_path / f"{patient_id}_param_D.nii.gz")
    nib.save(nib.Nifti1Image(rho_map_restored, original_affine), out_path / f"{patient_id}_param_rho.nii.gz")

    logger.info("Simulation completed successfully.", extra={"output_dir": str(out_path)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Glioblastoma Growth")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .ckpt file")
    parser.add_argument("--data_dir", type=str, default="data", help="Root directory containing 'images' and 'masks'")
    parser.add_argument("--patient_id", type=str, required=True, help="Base ID of the patient (e.g., UPENN-GBM-00036_11)")
    parser.add_argument("--output_dir", type=str, default="simulations/output", help="Directory to save NIfTI results")
    parser.add_argument("--target_days", type=float, default=60.0, help="Days to simulate into the future")
    
    args = parser.parse_args()
    run_simulation(args.checkpoint, args.data_dir, args.patient_id, args.output_dir, args.target_days)