import logging
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Production logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Cohort_Selector")


def extract_base_identifier(full_id: str) -> str:
    """Extracts the base patient string from the UPenn visit format."""
    return full_id.split('_')[0]


def compile_inference_cohort(
    avail_csv: str, 
    clinical_csv: str, 
    output_csv: str, 
    cohort_size: int = 30
) -> None:
    """
    Identifies the optimal patient cohort based on maximum temporal divergence.
    Outputs an intermediary registry for the batch pseudo-labeling pipeline.
    """
    logger.info("Initializing longitudinal cohort selection.")
    
    try:
        df_avail = pd.read_csv(avail_csv)
        df_clin = pd.read_csv(clinical_csv)
    except Exception as e:
        logger.critical(f"Metadata ingestion failure: {e}")
        raise

    df_merged = pd.merge(df_avail, df_clin, on='ID', how='inner')
    
    df_valid = df_merged[
        (df_merged['Structural imaging'].str.lower() == 'available') &
        (df_merged['Time_since_baseline_preop'].notna())
    ].copy()

    df_valid['Base_ID'] = df_valid['ID'].apply(extract_base_identifier)
    grouped = df_valid.groupby('Base_ID')
    
    candidate_cohort: List[Dict[str, any]] = []

    for base_id, group in grouped:
        if len(group) < 2:
            continue
            
        group_sorted = group.sort_values(by='Time_since_baseline_preop')
        baseline = group_sorted.iloc[0]
        followup = group_sorted.iloc[-1]
        
        time_delta = float(followup['Time_since_baseline_preop']) - float(baseline['Time_since_baseline_preop'])
        
        if time_delta > 0:
            candidate_cohort.append({
                "patient_id": base_id,
                "baseline_visit_id": baseline['ID'],
                "followup_visit_id": followup['ID'],
                "time_delta_days": time_delta
            })

    # Sort to prioritize patients with the longest follow-up times for physical simulation
    candidate_cohort.sort(key=lambda x: x['time_delta_days'], reverse=True)
    selected_cohort = candidate_cohort[:cohort_size]
    
    if not selected_cohort:
        logger.error("Failed to identify any valid longitudinal patients.")
        return

    try:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(selected_cohort).to_csv(output_path, index=False)
        
        logger.info(f"Cohort compilation successful. N={len(selected_cohort)}")
        logger.info(f"Max time delta: {selected_cohort[0]['time_delta_days']} days.")
        logger.info(f"Target list persisted to: {output_path}")
    except Exception as e:
        logger.critical(f"Failed to serialize cohort registry: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile target cohort for pseudo-labeling")
    parser.add_argument("--avail_csv", type=str, required=True, help="Availability metadata CSV")
    parser.add_argument("--clinical_csv", type=str, required=True, help="Clinical metadata CSV")
    parser.add_argument("--out_csv", type=str, default="data/inference_cohort.csv", help="Output execution list")
    parser.add_argument("--n", type=int, default=30, help="Number of patients to select")
    
    args = parser.parse_args()
    compile_inference_cohort(args.avail_csv, args.clinical_csv, args.out_csv, args.n)