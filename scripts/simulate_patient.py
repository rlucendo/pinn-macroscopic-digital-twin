import os
import argparse
import logging
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Optional

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    NormalizeIntensityd, Resized, Orientationd
)

# Domain imports
from src.models.train_simulator import TheoreticalGlioSimSystem

# Structured professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("PatientSimulator")


class PatientDigitalTwin:
    """
    Production-grade inference engine for individual patient simulation.
    Encapsulates input standardization, neural parameter extraction, and PDE integration.
    """
    def __init__(self, checkpoint_path: str, target_shape: tuple = (96, 96, 96)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_shape = target_shape
        
        logger.info(f"Initializing Patient Digital Twin on {self.device}...")
        
        try:
            self.model = TheoreticalGlioSimSystem.load_from_checkpoint(
                checkpoint_path, 
                map_location=self.device,
                weights_only=False
            )
            self.model.eval()
            logger.info("Neural PDE Solver loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint at {checkpoint_path}: {str(e)}")
            raise

        # Strict inference pipeline to match training physical domain space
        self.preprocessor = Compose([
            LoadImaged(keys=["image_t0", "mask_t0"]),
            EnsureChannelFirstd(keys=["image_t0", "mask_t0"]),
            Orientationd(keys=["image_t0", "mask_t0"], axcodes="RAS"),
            NormalizeIntensityd(keys=["image_t0"]),
            Resized(keys=["image_t0", "mask_t0"], spatial_size=self.target_shape, mode=("trilinear", "nearest"))
        ])

    @torch.no_grad()
    def simulate(self, image_path: str, mask_path: str, days_to_simulate: float) -> Dict[str, torch.Tensor]:
        """
        Executes the forward simulation for a specific timeframe.
        
        Args:
            image_path: Path to the T0 MRI NIfTI file.
            mask_path: Path to the T0 tumor segmentation mask.
            days_to_simulate: Temporal delta (t) for the PDE Euler integration.
            
        Returns:
            Dictionary containing the physical parameter maps and the future density prediction.
        """
        logger.info(f"Preparing simulation for {days_to_simulate} days based on {image_path}")
        
        try:
            # 1. Standardize physical domain
            input_data = {"image_t0": image_path, "mask_t0": mask_path}
            processed_data = self.preprocessor(input_data)
            
            img_t0 = processed_data["image_t0"].unsqueeze(0).to(self.device)
            u_t0 = processed_data["mask_t0"].unsqueeze(0).to(self.device)
            delta_t = torch.tensor([[days_to_simulate]], dtype=torch.float32).to(self.device)

            # 2. Extract patient-specific mechanical properties
            raw_D, raw_rho = self.model(img_t0)
            
            # Biological scaling constraints
            D_map = 0.001 + torch.sigmoid(raw_D) * (0.020 - 0.001)
            rho_map = 0.012 + torch.sigmoid(raw_rho) * (0.034 - 0.012)

            # 3. Explicit PDE mathematical simulation
            u_pred_tn = self.model.simulator(
                u_t0=u_t0, D_map=D_map, rho_map=rho_map, delta_t=delta_t
            )

            logger.info("Simulation completed successfully.")
            
            return {
                "input_image": img_t0.cpu().squeeze(),
                "initial_mask": u_t0.cpu().squeeze(),
                "predicted_density": u_pred_tn.cpu().squeeze(),
                "diffusion_map": D_map.cpu().squeeze(),
                "proliferation_map": rho_map.cpu().squeeze(),
                "affine": processed_data["image_t0"].affine
            }
            
        except Exception as e:
            logger.error(f"Simulation failed during forward pass: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Simulate Glioblastoma growth for a single patient.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .ckpt")
    parser.add_argument("--image", type=str, required=True, help="Path to patient T0 MRI (.nii.gz)")
    parser.add_argument("--mask", type=str, required=True, help="Path to patient T0 tumor mask (.nii.gz)")
    parser.add_argument("--days", type=float, required=True, help="Number of days to simulate")
    parser.add_argument("--output_dir", type=str, default="simulation_outputs", help="Directory to save NIfTI results")
    
    args = parser.parse_args()

    simulator = PatientDigitalTwin(checkpoint_path=args.checkpoint)
    results = simulator.simulate(
        image_path=args.image,
        mask_path=args.mask,
        days_to_simulate=args.days
    )

    # CLI Output Handling
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Export binarized prediction to NIfTI
    pred_binary = (results["predicted_density"] > 0.5).float().numpy()
    affine = results["affine"].numpy()
    
    nib.save(nib.Nifti1Image(pred_binary, affine), str(out_dir / "simulated_tumor.nii.gz"))
    logger.info(f"Artifacts exported to {out_dir}/")

if __name__ == "__main__":
    main()