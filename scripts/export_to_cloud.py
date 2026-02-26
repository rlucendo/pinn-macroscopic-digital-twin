import os
import shutil
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("CloudExporter")

def build_cloud_payload(csv_path: str, export_dir: str):
    """
    Isolates and packages only the required NIfTI volumes for the target cohort,
    minimizing cloud upload payload.
    """
    target_root = Path(export_dir)
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True)
    
    df = pd.read_csv(csv_path)
    
    # Copy the registry itself
    shutil.copy2(csv_path, target_root / "dataset_registry.csv")
    
    transferred_count = 0
    missing_files = []

    for _, row in df.iterrows():
        patient_id = row['patient_id']
        t0_visit_id = f"{patient_id}_11"
        
        # 1. Resolve source paths
        img_t0_src = Path(row['image_t0_path'])
        mask_tn_src = Path(row['mask_tn_path'])
        
        # Resolve mask T0 (replicating DataModule logic)
        base_dir_parts = str(img_t0_src).split("images_structural")
        nifti_root = Path(base_dir_parts[0])
        mask_t0_manual = nifti_root / "images_segm" / f"{t0_visit_id}_segm.nii.gz"
        mask_t0_auto = nifti_root / "automated_segm" / f"{t0_visit_id}_automated_approx_segm.nii.gz"
        mask_t0_src = mask_t0_manual if mask_t0_manual.exists() else mask_t0_auto

        # Validate existence before copy
        if not img_t0_src.exists() or not mask_tn_src.exists() or not mask_t0_src.exists():
            missing_files.append(patient_id)
            continue

        # 2. Define target relative paths
        # We flatten the structure slightly for Colab efficiency while keeping logical separation
        img_t0_dst = target_root / "images_structural" / t0_visit_id / img_t0_src.name
        mask_tn_dst = target_root / "pseudo_labels" / mask_tn_src.name
        
        # Maintain original parent folder name for T0 mask logic (images_segm or automated_segm)
        mask_t0_parent_name = mask_t0_src.parent.name
        mask_t0_dst = target_root / mask_t0_parent_name / mask_t0_src.name

        # 3. Create directories and copy
        img_t0_dst.parent.mkdir(parents=True, exist_ok=True)
        mask_tn_dst.parent.mkdir(parents=True, exist_ok=True)
        mask_t0_dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img_t0_src, img_t0_dst)
        shutil.copy2(mask_tn_src, mask_tn_dst)
        shutil.copy2(mask_t0_src, mask_t0_dst)
        
        # 4. Update CSV paths to be relative to the new export root
        df.loc[df['patient_id'] == patient_id, 'image_t0_path'] = str(img_t0_dst.relative_to(target_root))
        df.loc[df['patient_id'] == patient_id, 'mask_tn_path'] = str(mask_tn_dst.relative_to(target_root))
        
        transferred_count += 1

    # Overwrite the CSV in the export folder with relative paths
    df.to_csv(target_root / "dataset_registry.csv", index=False)
    
    logger.info(f"Payload extraction complete. Transferred {transferred_count} patients.")
    if missing_files:
        logger.warning(f"Missing source files for: {missing_files}")

if __name__ == "__main__":
    build_cloud_payload("data/dataset_registry.csv", "colab_export")