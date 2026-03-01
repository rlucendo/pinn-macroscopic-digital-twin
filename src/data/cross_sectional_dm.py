import logging
import torch
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from monai.transforms import Resize

logger = logging.getLogger(__name__)

class CrossSectionalDataset(Dataset):
    """
    Multimodal Dataset loading FLAIR, T1, T1GD, and T2.
    Translates multi-class clinical segmentations into biological density maps (u).
    """
    def __init__(self, data_list: List[Dict[str, Any]], target_shape=(96, 96, 96)):
        self.data_list = data_list
        self.cached_samples = []
        
        self.img_resizer = Resize(spatial_size=target_shape, mode="trilinear")
        self.mask_resizer = Resize(spatial_size=target_shape, mode="nearest")
        
        logger.info("Caching Multimodal Subjects...", extra={"total_subjects": len(data_list)})
        
        for idx, item in enumerate(data_list):
            try:
                # 1. Load the 4 modalities
                modalities = []
                for mod_path in item["image_paths"]:
                    nifti = nib.load(mod_path)
                    img = nifti.get_fdata()
                    
                    # Independent Z-Score Normalization per modality excluding background
                    mask_bg = img > 0
                    if mask_bg.sum() > 0:
                        mean = img[mask_bg].mean()
                        std = img[mask_bg].std()
                        img = (img - mean) / (std + 1e-8)
                        img[~mask_bg] = 0.0
                    
                    # Shape: [1, D, H, W]
                    mod_tensor = torch.from_numpy(img).float().unsqueeze(0)
                    modalities.append(mod_tensor)

                # Concatenate along channel dimension. Shape: [4, D, H, W]
                multi_channel_img = torch.cat(modalities, dim=0)

                # Extract spatial metadata from the first modality
                ref_nifti = nib.load(item["image_paths"][0])
                original_spacing = ref_nifti.header.get_zooms()[:3]
                original_shape = ref_nifti.shape[:3]
                physical_size = [original_shape[i] * original_spacing[i] for i in range(3)]
                new_spacing = [physical_size[i] / target_shape[i] for i in range(3)]
                isotropic_spacing = sum(new_spacing) / 3.0

                # 2. Load and Translate the Mask
                mask_nifti = nib.load(item["mask_path"])
                raw_mask = mask_nifti.get_fdata()
                
                # Biomechanical Mapping: Classes to Cell Density (u)
                # Note: BraTS/UPENN standard: 1=Necrosis, 2=Edema, 4=Enhancing Core
                # Adjust these numbers if UPENN-GBM uses a different standard (e.g., 1, 2, 3)
                density_map = np.zeros_like(raw_mask, dtype=np.float32)
                
                # Dense tumor tissue (Core & Necrosis) mapped to max density 1.0
                density_map[(raw_mask == 1) | (raw_mask == 4) | (raw_mask == 3)] = 1.0
                
                # Infiltrating tissue (Edema) mapped to intermediate density 0.5
                density_map[raw_mask == 2] = 0.5
                
                density_tensor = torch.from_numpy(density_map).unsqueeze(0)

                # 3. Spatial Alignment
                img_final = self.img_resizer(multi_channel_img)
                mask_final = self.mask_resizer(density_tensor)
                
                self.cached_samples.append({
                    "image_t0": img_final.float(),
                    "mask_t0": mask_final.float(),
                    "voxel_spacing": torch.tensor(isotropic_spacing, dtype=torch.float32)
                })
            except Exception as e:
                logger.error(f"Failed to process subject {item['patient_id']}", exc_info=True)
                continue

    def __len__(self) -> int:
        return len(self.cached_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.cached_samples[idx]


class CrossSectionalDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 4):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        csv_path = Path(self.hparams.data_dir) / "dataset_registry.csv"
        # Since we use all available, we can also just scan the directory if CSV is outdated,
        # but using the CSV is safer for metadata linking.
        df = pd.read_csv(csv_path)
        raw_data = []
        data_root = Path(self.hparams.data_dir)

        modality_suffixes = ["FLAIR", "T1", "T1GD", "T2"]

        for _, row in df.iterrows():
            p_id = str(row["patient_id"]).strip()
            t0_id = f"{p_id}_11"
            
            # Construct paths for all 4 modalities
            img_dir = data_root / "images_structural" / t0_id
            mod_paths = [img_dir / f"{t0_id}_{mod}.nii.gz" for mod in modality_suffixes]
            
            # Use the automatic segmentation provided by the dataset
            # Adjust the suffix '_segm.nii.gz' if the dataset uses a different naming convention
            mask_path = data_root / "masks_t0" / f"{t0_id}_automatic_segm.nii.gz"

            # Check if all 4 images and the mask exist
            all_images_exist = all(p.exists() for p in mod_paths)
            
            if all_images_exist and mask_path.exists():
                raw_data.append({
                    "patient_id": p_id,
                    "image_paths": [str(p) for p in mod_paths],
                    "mask_path": str(mask_path)
                })

        if not raw_data:
            raise RuntimeError("No valid multimodal pairs found.")

        np.random.seed(42)
        np.random.shuffle(raw_data)
        split_idx = int(len(raw_data) * 0.8)
        
        target_shape = (96, 96, 96)
        self.train_ds = CrossSectionalDataset(raw_data[:split_idx], target_shape=target_shape)
        self.val_ds = CrossSectionalDataset(raw_data[split_idx:], target_shape=target_shape)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0, pin_memory=True)