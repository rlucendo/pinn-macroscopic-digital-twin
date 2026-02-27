import os
import shutil
import logging
import argparse
import pandas as pd
from pathlib import Path
from typing import List

# Configure structured professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("CloudPayloadPackager")


def build_cloud_payload(csv_path: str, export_dir: str) -> None:
    """
    Isolates and packages the required structural NIfTI volumes and harmonized
    segmentation masks for the target cohort, minimizing the cloud upload payload.
    """
    source_csv = Path(csv_path)
    target_root = Path(export_dir)
    
    if not source_csv.exists():
        logger.critical(f"Source registry not found at: {source_csv}")
        raise FileNotFoundError(f"Missing registry: {source_csv}")

    # Reset export directory to ensure a clean state
    if target_root.exists():
        logger.info(f"Cleaning existing export directory: {target_root}")
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True)
    
    try:
        df = pd.read_csv(source_csv)
    except Exception as e:
        logger.critical(f"Failed to read dataset registry: {e}")
        raise

    transferred_count: int = 0
    missing_files: List[str] = []

    for index, row in df.iterrows():
        patient_id = row['patient_id']
        
        # 1. Resolve source paths directly from the coherent registry
        img_t0_src = Path(row['image_t0_path'])
        mask_t0_src = Path(row['mask_t0_path'])
        mask_tn_src = Path(row['mask_tn_path'])

        # 2. Validate strict existence before attempting I/O operations
        if not img_t0_src.exists() or not mask_t0_src.exists() or not mask_tn_src.exists():
            logger.warning(f"Incomplete source files for patient {patient_id}. Skipping.")
            missing_files.append(patient_id)
            continue

        # 3. Define target relative paths maintaining logical separation
        # Flattening the directory tree slightly to optimize Colab filesystem operations
        t0_visit_id = f"{patient_id}_11"
        
        img_t0_dst = target_root / "images_structural" / t0_visit_id / img_t0_src.name
        mask_t0_dst = target_root / "pseudo_labels" / mask_t0_src.name
        mask_tn_dst = target_root / "pseudo_labels" / mask_tn_src.name

        # 4. Create topological structure and execute file transfer
        for dst_path in [img_t0_dst, mask_t0_dst, mask_tn_dst]:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
        shutil.copy2(img_t0_src, img_t0_dst)
        shutil.copy2(mask_t0_src, mask_t0_dst)
        shutil.copy2(mask_tn_src, mask_tn_dst)
        
        # 5. Update registry paths to be strictly relative to the target_root
        # This guarantees environment-agnostic execution (Local vs Cloud)
        df.at[index, 'image_t0_path'] = str(img_t0_dst.relative_to(target_root))
        df.at[index, 'mask_t0_path'] = str(mask_t0_dst.relative_to(target_root))
        df.at[index, 'mask_tn_path'] = str(mask_tn_dst.relative_to(target_root))
        
        transferred_count += 1

    # Serialize the environment-agnostic registry
    export_csv_path = target_root / "dataset_registry.csv"
    try:
        df.to_csv(export_csv_path, index=False)
        logger.info(f"Payload packaging complete. Transferred {transferred_count} complete patient records.")
        logger.info(f"Environment-agnostic registry generated at: {export_csv_path}")
    except Exception as e:
        logger.critical(f"Failed to serialize cloud registry: {e}")
        raise
        
    if missing_files:
        logger.warning(f"Total dropped patients due to missing physical files: {len(missing_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package minimal dataset payload for Cloud/A100 training")
    parser.add_argument("--csv_path", type=str, default="data/dataset_registry_coherent.csv", help="Path to the source registry")
    parser.add_argument("--export_dir", type=str, default="colab_export", help="Target packaging directory")
    
    args = parser.parse_args()
    build_cloud_payload(args.csv_path, args.export_dir)