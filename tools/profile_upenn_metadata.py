import os
import logging
import argparse
import pandas as pd
from typing import Optional

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Metadata_Profiler")


def profile_clinical_metadata(
    availability_path: str, 
    clinical_path: str
) -> None:
    """
    Profiles the UPenn-GBM clinical and availability metadata CSVs to map
    the exact column schemas required for longitudinal data extraction.
    """
    if not os.path.exists(availability_path):
        logger.critical(f"Availability metadata missing at target path: {availability_path}")
        raise FileNotFoundError(f"Missing file: {availability_path}")
        
    if not os.path.exists(clinical_path):
        logger.critical(f"Clinical metadata missing at target path: {clinical_path}")
        raise FileNotFoundError(f"Missing file: {clinical_path}")

    try:
        df_avail = pd.read_csv(availability_path)
        df_clin = pd.read_csv(clinical_path)
        
        logger.info("=== Data Availability Schema ===")
        logger.info(f"Shape: {df_avail.shape}")
        logger.info(f"Columns: {list(df_avail.columns)}")
        # Transpose head(1) for cleaner vertical logging in the terminal
        logger.info(f"Sample Record:\n{df_avail.head(1).T}")
        
        logger.info("=== Clinical Data Schema ===")
        logger.info(f"Shape: {df_clin.shape}")
        logger.info(f"Columns: {list(df_clin.columns)}")
        logger.info(f"Sample Record:\n{df_clin.head(1).T}")
        
    except pd.errors.EmptyDataError:
        logger.error("One or both CSV files are empty.")
    except Exception as e:
        logger.critical(f"Unhandled exception during data profiling: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile UPenn-GBM Metadata")
    parser.add_argument(
        "--avail_csv", 
        type=str, 
        required=True, 
        help="Path to the 'Data availability per subject' CSV"
    )
    parser.add_argument(
        "--clinical_csv", 
        type=str, 
        required=True, 
        help="Path to the 'Clinical Data' CSV"
    )
    
    args = parser.parse_args()
    
    try:
        profile_clinical_metadata(
            availability_path=args.avail_csv, 
            clinical_path=args.clinical_csv
        )
    except Exception as e:
        logger.critical(f"Execution halted: {e}")
        exit(1)