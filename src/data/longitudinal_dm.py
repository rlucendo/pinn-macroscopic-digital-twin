import os
import logging
import pandas as pd
import torch
import lightning.pytorch as pl
from typing import Optional, List, Dict, Any
from torch.utils.data import DataLoader

from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    EnsureTyped,
    DivisiblePadd
)

# Configure module-level logger
logger = logging.getLogger(__name__)

class LongitudinalDataModule(pl.LightningDataModule):
    """
    DataModule for physics-informed tumor growth simulation.
    Handles loading of baseline anatomy (T0), baseline tumor mask (T0),
    follow-up pseudo-label mask (Tn), and the temporal delta.
    """
    def __init__(
        self, 
        data_dir: str = "data", 
        batch_size: int = 1, 
        num_workers: int = 4
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None

    def get_transforms(self) -> Compose:
        """
        Defines the preprocessing pipeline for the physics simulator.
        Ensures strict spatial alignment between anatomy and multi-temporal masks.
        """
        keys = ["image_t0", "mask_t0", "mask_tn"]
        return Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            # Resample all volumes to 1x1x1 mm isotropic resolution.
            Spacingd(
                keys=keys, 
                pixdim=(1.0, 1.0, 1.0), 
                mode=("bilinear", "nearest", "nearest")
            ),
            # Force all spatial dimensions to be multiples of 16 to prevent 
            # U-Net skip connection dimension mismatches during down/upsampling.
            DivisiblePadd(keys=keys, k=16, mode="constant", constant_values=0),
            
            NormalizeIntensityd(keys=["image_t0"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=keys + ["time_delta"], data_type="tensor")
        ])

    def _parse_longitudinal_registry(self) -> List[Dict[str, Any]]:
        """
        Parses the unified longitudinal registry containing real T0 anatomy 
        and pseudo-labeled Tn targets.
        """
        csv_path = os.path.join(self.hparams.data_dir, "dataset_registry.csv")
        
        if not os.path.exists(csv_path):
            logger.critical(f"Registry not found: {csv_path}")
            raise FileNotFoundError(f"Missing registry: {csv_path}")
            
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Failed to read CSV registry: {e}")
            raise

        registry: List[Dict[str, Any]] = []
        required_columns = ["image_t0_path", "mask_tn_path", "days_between", "patient_id"]
        
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise KeyError(f"Invalid CSV structure. Missing: {missing}")

        for index, row in df.iterrows():
            try:
                patient_id = row["patient_id"]
                t0_visit_id = f"{patient_id}_11" 
                
                # Extract root path dynamically from the structural image path
                base_dir_parts = row["image_t0_path"].split("images_structural")
                if len(base_dir_parts) == 2:
                    nifti_root = base_dir_parts[0]
                    
                    t0_mask_manual = os.path.join(nifti_root, "images_segm", f"{t0_visit_id}_segm.nii.gz")
                    t0_mask_auto = os.path.join(nifti_root, "automated_segm", f"{t0_visit_id}_automated_approx_segm.nii.gz")
                    
                    # Resolve baseline mask path
                    t0_mask_path = t0_mask_manual if os.path.exists(t0_mask_manual) else t0_mask_auto
                    
                    if not os.path.exists(t0_mask_path):
                        logger.warning(f"No baseline mask found for {patient_id}. Skipping.")
                        continue
                else:
                    logger.warning(f"Could not resolve NIfTI root for {patient_id}. Skipping.")
                    continue

                registry.append({
                    "image_t0": str(row["image_t0_path"]),
                    "mask_t0": str(t0_mask_path),
                    "mask_tn": str(row["mask_tn_path"]),
                    "time_delta": torch.tensor([float(row["days_between"])], dtype=torch.float32)
                })
            except Exception as e:
                logger.warning(f"Row {index} skipped due to parsing error: {e}")
                continue

        if not registry:
            logger.error("No valid samples could be compiled from the registry.")
            raise ValueError("Empty dataset registry.")

        logger.info(f"DataModule initialized with {len(registry)} valid longitudinal pairs.")
        return registry

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Instantiates the MONAI datasets. Implements an 80/20 train-validation split.
        """
        data_dicts = self._parse_longitudinal_registry()
        
        split_idx = int(len(data_dicts) * 0.8)
        train_files = data_dicts[:split_idx]
        val_files = data_dicts[split_idx:]
        
        logger.info(f"Dataset split: {len(train_files)} Train / {len(val_files)} Validation")
        
        self.train_ds = Dataset(data=train_files, transform=self.get_transforms())
        self.val_ds = Dataset(data=val_files, transform=self.get_transforms())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )