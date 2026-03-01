import logging
import torch
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

# High-performance MONAI transforms
from monai.transforms import Resize

logger = logging.getLogger(__name__)

class CachedGlioDataset(Dataset):
    """
    Optimized RAM-cached Dataset ensuring strict spatial resizing
    and physical scale preservation for PDE integration.
    """
    def __init__(self, data_list: List[Dict[str, Any]], target_shape=(96, 96, 96)):
        self.data_list = data_list
        self.cached_samples = []
        
        # Dual Resizing Strategy
        self.img_resizer = Resize(spatial_size=target_shape, mode="trilinear")
        self.mask_resizer = Resize(spatial_size=target_shape, mode="nearest")
        
        logger.info("RAM Caching & Resizing subjects...", extra={"target_shape": target_shape})
        
        for idx, item in enumerate(data_list):
            try:
                # 1. Load Data and Metadata
                img_path = item["image_t0"]
                img_nifti = nib.load(img_path)
                
                # Extract original voxel spacing from the affine header
                original_spacing = img_nifti.header.get_zooms()[:3]
                original_shape = img_nifti.shape[:3]
                
                # Calculate physical dimensions (mm)
                physical_size = [original_shape[i] * original_spacing[i] for i in range(3)]
                
                # Calculate NEW isotropic voxel spacing after resizing to target_shape
                # We assume the FOV (Field of View) remains strictly identical
                new_spacing = [physical_size[i] / target_shape[i] for i in range(3)]
                # Average to enforce isotropic assumption for the PDE solver
                isotropic_spacing = sum(new_spacing) / 3.0

                # 2. Extract Data
                img = torch.from_numpy(img_nifti.get_fdata()).float().unsqueeze(0)
                m0 = self._load_tensor(item["mask_t0"])
                mn = self._load_tensor(item["mask_tn"])
                
                # 3. Intensity Normalization (Zero-Mean, Unit-Variance) excluding background
                mask_bg = img > 0
                if mask_bg.sum() > 0:
                    mean = img[mask_bg].mean()
                    std = img[mask_bg].std()
                    img = (img - mean) / (std + 1e-8)
                    img[~mask_bg] = 0.0

                # 4. Strict Alignment via Resizing
                img_final = self.img_resizer(img)
                m0_final = self.mask_resizer(m0)
                mn_final = self.mask_resizer(mn)
                
                self.cached_samples.append({
                    "image_t0": img_final.float(),
                    "mask_t0": m0_final.float(),
                    "mask_tn": mn_final.float(),
                    "time_delta": item["time_delta"],
                    "voxel_spacing": torch.tensor(isotropic_spacing, dtype=torch.float32)
                })
            except Exception as e:
                logger.error(f"Failed to process subject {idx}", exc_info=True)
                continue

    def _load_tensor(self, path: str) -> torch.Tensor:
        data = nib.load(path).get_fdata()
        return torch.from_numpy(data).float().unsqueeze(0)

    def __len__(self) -> int:
        return len(self.cached_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.cached_samples[idx]


class LongitudinalDataModule(pl.LightningDataModule):
    """
    Orchestrates high-speed data delivery for Hard-Physics Neural Networks.
    Ensures strict longitudinal data validity.
    """
    def __init__(self, data_dir: str = "data", batch_size: int = 4):
        super().__init__()
        self.save_hyperparameters()
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        csv_path = Path(self.hparams.data_dir) / "dataset_registry.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing registry: {csv_path}")

        df = pd.read_csv(csv_path)
        raw_data = []
        data_root = Path(self.hparams.data_dir)

        for _, row in df.iterrows():
            p_id = str(row["patient_id"]).strip()
            t0_id, tn_id = f"{p_id}_11", f"{p_id}_21"
            
            img_t0 = data_root / "images_structural" / t0_id / f"{t0_id}_FLAIR.nii.gz"
            mask_t0 = data_root / "masks_t0" / f"{t0_id}_FLAIR_pseudo_segm.nii.gz"
            mask_tn = data_root / "masks_tn" / f"{tn_id}_FLAIR_pseudo_segm.nii.gz"

            # Strict validation: Only accept subjects with actual longitudinal growth data
            if img_t0.exists() and mask_t0.exists() and mask_tn.exists():
                delta = float(row["days_between"])
                if delta <= 0:
                    continue # Skip illogical time steps
                    
                raw_data.append({
                    "image_t0": str(img_t0),
                    "mask_t0": str(mask_t0),
                    "mask_tn": str(mask_tn),
                    "time_delta": torch.tensor([delta], dtype=torch.float32)
                })

        if not raw_data:
            raise RuntimeError("No valid longitudinal pairs found after filtering.")

        # Train/Val Split (80/20) - Deterministic for reproducibility
        np.random.seed(42)
        np.random.shuffle(raw_data)
        split_idx = int(len(raw_data) * 0.8)
        
        # Target shape must be multiples of 16 for UNet optimal performance
        target_shape = (96, 96, 96)
        self.train_ds = CachedGlioDataset(raw_data[:split_idx], target_shape=target_shape)
        self.val_ds = CachedGlioDataset(raw_data[split_idx:], target_shape=target_shape)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )