import logging
import torch
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

# High-performance MONAI transforms
from monai.transforms import Resize, Compose

logger = logging.getLogger("HighPerformance_DataModule")

class CachedGlioDataset(Dataset):
    """
    Optimized RAM-cached Dataset with strict Spatial Resizing.
    Forces all volumes to a multiple of 16 (96x96x96) to ensure 
    perfect UNet skip-connection alignment and A100 Tensor Core efficiency.
    """
    def __init__(self, data_list: List[Dict[str, Any]], target_shape=(96, 96, 96)):
        self.data_list = data_list
        self.cached_samples = []
        
        # Dual Resizing Strategy:
        # Trilinear for continuous MRI data, Nearest for discrete Segmentation masks
        self.img_resizer = Resize(spatial_size=target_shape, mode="trilinear")
        self.mask_resizer = Resize(spatial_size=target_shape, mode="nearest")
        
        logger.info(f"RAM Caching & Resizing {len(data_list)} subjects to {target_shape}...")
        
        for idx, item in enumerate(data_list):
            try:
                # 1. Load from Disk
                img = self._load_tensor(item["image_t0"])
                m0 = self._load_tensor(item["mask_t0"])
                mn = self._load_tensor(item["mask_tn"])
                
                # 2. Force strict alignment to target_shape
                img_final = self.img_resizer(img)
                m0_final = self.mask_resizer(m0)
                mn_final = self.mask_resizer(mn)
                
                self.cached_samples.append({
                    "image_t0": img_final.float(),
                    "mask_t0": m0_final.float(),
                    "mask_tn": mn_final.float(),
                    "time_delta": item["time_delta"]
                })
            except Exception as e:
                logger.error(f"Failed to process subject {idx}: {e}")
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
    Orchestrates high-speed data delivery for Physics-Informed Neural Networks.
    Handles caching, splitting, and GPU memory pinning.
    """
    def __init__(self, data_dir: str = "data", batch_size: int = 4):
        super().__init__()
        self.save_hyperparameters()
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Prepares datasets by reading the registry and loading volumes into RAM."""
        csv_path = Path(self.hparams.data_dir) / "dataset_registry.csv"
        if not csv_path.exists():
            logger.critical(f"Data registry not found at {csv_path}")
            raise FileNotFoundError(f"Missing registry: {csv_path}")

        df = pd.read_csv(csv_path)
        raw_data = []

        # Parse logic with strict folder synchronization
        data_root = Path(self.hparams.data_dir)
        for _, row in df.iterrows():
            p_id = str(row["patient_id"]).strip()
            t0_id, tn_id = f"{p_id}_11", f"{p_id}_21"
            
            # Resolve paths using the structured folders (images_structural, masks_t0, masks_tn)
            img_t0 = data_root / "images_structural" / t0_id / f"{t0_id}_FLAIR.nii.gz"
            mask_t0 = data_root / "masks_t0" / f"{t0_id}_FLAIR_pseudo_segm.nii.gz"
            mask_tn = data_root / "masks_tn" / f"{tn_id}_FLAIR_pseudo_segm.nii.gz"

            if img_t0.exists() and mask_t0.exists():
                raw_data.append({
                    "image_t0": str(img_t0),
                    "mask_t0": str(mask_t0),
                    "mask_tn": str(mask_tn) if mask_tn.exists() else str(mask_t0),
                    "time_delta": torch.tensor([float(row["days_between"])], dtype=torch.float32)
                })

        # Train/Val Split (80/20)
        split_idx = int(len(raw_data) * 0.8)
        train_list = raw_data[:split_idx]
        val_list = raw_data[split_idx:]

        # Initialize with explicit 96x96x96 padding
        self.train_ds = CachedGlioDataset(train_list, target_shape=(96, 96, 96))
        self.val_ds = CachedGlioDataset(val_list, target_shape=(96, 96, 96))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=0,      # num_workers=0 is faster when using RAM Caching
            pin_memory=True     # Critical for A100 GPU transfer speed
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=0, pin_memory=True)