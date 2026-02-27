import os
import argparse
import logging
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# Assume the system class is importable from the training script
from scripts.train_simulator import TheoreticalGlioSimSystem
from src.data.longitudinal_dm import LongitudinalDataModule

# Structured logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("DigitalTwinEvaluator")

class InferencePipeline:
    """
    Handles post-training evaluation, biomarker extraction, and NIfTI export
    for the Theoretical Glioblastoma Digital Twin.
    """
    def __init__(self, checkpoint_path: str, data_dir: str, output_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing inference pipeline on device: {self.device}")
        
        try:
            # Pass weights_only=False to allow MONAI metadata to load
            self.model = TheoreticalGlioSimSystem.load_from_checkpoint(
                checkpoint_path,
                weights_only=False
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model checkpoint loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
            raise

        self.datamodule = LongitudinalDataModule(data_dir=data_dir, batch_size=1)
        self.datamodule.setup(stage="test")

    def _export_nifti(self, tensor: torch.Tensor, filename: str, reference_affine: np.ndarray = np.eye(4)) -> None:
        """
        Converts a PyTorch tensor to a NIfTI file and writes it to disk.
        """
        try:
            # Squeeze batch and channel dimensions, detach from graph, move to CPU
            numpy_array = tensor.detach().cpu().squeeze().numpy()
            nifti_img = nib.Nifti1Image(numpy_array, reference_affine)
            
            output_path = self.output_dir / filename
            nib.save(nifti_img, str(output_path))
            logger.debug(f"Exported volume to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export NIfTI file {filename}: {str(e)}")

    def run_evaluation(self) -> None:
        """
        Executes the forward pass on the validation dataset, computes metrics,
        and extracts physical parameter maps.
        """
        val_loader = self.datamodule.val_dataloader()
        logger.info(f"Commencing evaluation on {len(val_loader)} subjects.")

        total_dice = 0.0
        processed_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    image_t0 = batch["image_t0"].to(self.device)
                    mask_tn = batch["mask_tn"].to(self.device)
                    delta_t = batch["time_delta"].to(self.device)
                    
                    # El DataLoader guarda la matriz espacial si usamos MONAI
                    # Extraemos el Affine original si está disponible (fallback a eye(4))
                    if "image_t0_meta_dict" in batch and "affine" in batch["image_t0_meta_dict"]:
                        original_affine = batch["image_t0_meta_dict"]["affine"][0].cpu().numpy()
                    else:
                        original_affine = np.eye(4)
                    
                    # 1. Forward simulation to time T_n
                    u_pred_tn, raw_D, raw_rho = self.model(image_t0, delta_t)

                    # 2. Binarize prediction (Crucial to remove the "Gray Ghost")
                    u_pred_binary = (u_pred_tn > 0.5).float()
                    
                    # 3. Dice Computation
                    intersection = torch.sum(u_pred_binary * mask_tn)
                    union = torch.sum(u_pred_binary) + torch.sum(mask_tn)
                    dice_score = (2.0 * intersection + 1e-8) / (union + 1e-8)
                    
                    total_dice += dice_score.item()
                    processed_count += 1

                    # 4. Map raw outputs to biological scales
                    d_map = 0.001 + torch.sigmoid(raw_D) * (0.020 - 0.001)
                    rho_map = 0.012 + torch.sigmoid(raw_rho) * (0.034 - 0.012)

                    # 5. Export NIfTIs using the BINARY mask and the REAL Affine matrix
                    subject_prefix = f"subject_{batch_idx:03d}"
                    self._export_nifti(u_pred_binary, f"{subject_prefix}_pred_density_tn.nii.gz", original_affine)
                    self._export_nifti(d_map, f"{subject_prefix}_diffusion_map.nii.gz", original_affine)
                    self._export_nifti(rho_map, f"{subject_prefix}_proliferation_map.nii.gz", original_affine)

                    logger.info(f"Processed {subject_prefix} | Dice Score: {dice_score.item():.4f}")

                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    continue

        if processed_count > 0:
            mean_dice = total_dice / processed_count
            logger.info(f"Evaluation complete. Mean Dice Score across cohort: {mean_dice:.4f}")
        else:
            logger.warning("Evaluation finished, but no subjects were successfully processed.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate the PINN Digital Twin.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the best .ckpt file.")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to the dataset directory.")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Directory to save NIfTI outputs.")
    
    args = parser.parse_args()

    pipeline = InferencePipeline(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    pipeline.run_evaluation()

if __name__ == "__main__":
    main()