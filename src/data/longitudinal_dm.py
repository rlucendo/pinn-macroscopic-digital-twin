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
        Parses the registry with strict path resolution and detailed logging.
        """
        from pathlib import Path
        
        csv_path = Path(self.hparams.data_dir) / "dataset_registry.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Registry not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        registry: List[Dict[str, Any]] = []

        for index, row in df.iterrows():
            try:
                # 1. Clean IDs and resolve baseline visit
                p_id = str(row["patient_id"]).strip()
                t0_visit_id = f"{p_id}_11"
                
                # 2. Absolute Path Construction
                data_root = Path(self.hparams.data_dir)
                img_t0 = data_root / str(row["image_t0_path"]).strip()
                mask_tn = data_root / str(row["mask_tn_path"]).strip()
                
                # 3. Baseline Mask Logic (Check both manual and automated folders)
                m_path = data_root / "images_segm" / f"{t0_visit_id}_segm.nii.gz"
                a_path = data_root / "automated_segm" / f"{t0_visit_id}_automated_approx_segm.nii.gz"
                
                mask_t0 = m_path if m_path.exists() else a_path

                # 4. Strict Validation with Explicit Logging
                if not img_t0.exists():
                    logger.warning(f"MISSING IMAGE T0: {img_t0}")
                    continue
                if not mask_t0.exists():
                    # This will tell us EXACTLY where it is looking
                    logger.warning(f"MISSING MASK T0 for {p_id}. Tried: {m_path} and {a_path}")
                    continue
                if not mask_tn.exists():
                    logger.warning(f"MISSING PSEUDO-LABEL TN: {mask_tn}")
                    continue

                registry.append({
                    "image_t0": str(img_t0),
                    "mask_t0": str(mask_t0),
                    "mask_tn": str(mask_tn),
                    "time_delta": torch.tensor([float(row["days_between"])], dtype=torch.float32)
                })
            except Exception as e:
                logger.error(f"Error parsing row {index}: {e}")
                continue

        if not registry:
            logger.error(f"Registry compilation failed. Checked data_dir: {self.hparams.data_dir}")
            raise ValueError("Empty dataset registry. Check file paths and existence.")

        logger.info(f"DataModule successfully loaded {len(registry)} patient pairs.")
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