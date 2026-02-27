import os
import logging
import argparse
import pandas as pd
import torch
from pathlib import Path
from omegaconf import OmegaConf

# PyTorch checkpoint security bypass
_original_load = torch.load
def _trusted_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _trusted_load

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, NormalizeIntensityd, SaveImage, AsDiscrete, ConcatItemsd
)
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader

# Import custom LightningModule from your previous project
from src.models.seg_module import BraTSSegmentationModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Batch_Harmonization_Engine")


def get_inference_transforms() -> Compose:
    """
    Defines the strict preprocessing pipeline for BraTS model inference.
    Loads 4 MRI modalities independently and concatenates them into a single 4-channel tensor.
    """
    keys_to_load = ["flair", "t1", "t1gd", "t2"]
    return Compose([
        LoadImaged(keys=keys_to_load),
        EnsureChannelFirstd(keys=keys_to_load),
        ConcatItemsd(keys=keys_to_load, name="image", dim=0),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])


def process_single_visit(
    visit_id: str, 
    root_path: Path, 
    model: torch.nn.Module, 
    model_cfg: OmegaConf, 
    device: torch.device, 
    saver: SaveImage, 
    pseudo_labels_dir: Path
) -> str:
    """
    Executes the 4-modality inference pipeline for a single clinical visit.
    Returns the absolute path to the generated segmentation mask.
    """
    modalities = ["FLAIR", "T1", "T1GD", "T2"]
    
    visit_paths = {
        mod.lower(): str(root_path / "images_structural" / visit_id / f"{visit_id}_{mod}.nii.gz") 
        for mod in modalities
    }
    
    if not all(Path(p).exists() for p in visit_paths.values()):
        logger.warning(f"Incomplete MRI sequences for visit {visit_id}. 4 modalities required.")
        return None

    logger.info(f"Processing 4-channel volume for visit: {visit_id}")
    
    test_ds = Dataset(data=[visit_paths], transform=get_inference_transforms())
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)
    post_pred = AsDiscrete(argmax=True)
    
    for batch_data in test_loader:
        inputs = batch_data["image"].to(device)
        
        outputs = sliding_window_inference(
            inputs=inputs, 
            roi_size=model_cfg.roi_size, 
            sw_batch_size=4, 
            predictor=model.forward,
            overlap=0.5
        )
        
        discrete_outputs = post_pred(outputs[0]).unsqueeze(0)
        batch_data["pred"] = discrete_outputs.cpu()
        
        # Execute saver transform using metadata from the FLAIR sequence
        saver(batch_data["pred"][0], meta_data=batch_data["flair"][0].meta)

    # MONAI SaveImage uses the input filename to construct the output path
    expected_mask_name = f"{visit_id}_FLAIR_pseudo_segm.nii.gz"
    generated_mask_path = pseudo_labels_dir / expected_mask_name
    
    return str(generated_mask_path.absolute())


def execute_batch_harmonization(
    cohort_csv: str,
    nifti_root: str,
    config_dir: str,
    ckpt_path: str,
    output_dir: str
) -> None:
    target_dir = Path(output_dir)
    pseudo_labels_dir = target_dir / "pseudo_labels"
    pseudo_labels_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        cohort_df = pd.read_csv(cohort_csv)
    except Exception as e:
        logger.critical(f"Failed to read cohort execution list: {e}")
        raise

    logger.info(f"Loading configuration from: {config_dir} and weights from: {ckpt_path}")
    try:
        model_cfg = OmegaConf.load(os.path.join(config_dir, "model_config.yaml"))
        model = BraTSSegmentationModule.load_from_checkpoint(
            checkpoint_path=ckpt_path, 
            cfg=model_cfg,
            hparams_file=None,
            strict=False
        )
        model.cfg = model_cfg
        model.net.in_channels = model_cfg.in_channels
    except Exception as e:
        logger.critical(f"Model initialization failed. Error: {e}")
        raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"Neural Engine loaded and allocated to {device}")

    saver = SaveImage(
        output_dir=str(pseudo_labels_dir), 
        output_postfix="pseudo_segm", 
        output_ext=".nii.gz",
        resample=False,
        separate_folder=False
    )

    pinn_registry = []
    root_path = Path(nifti_root)

    with torch.no_grad():
        for idx, row in cohort_df.iterrows():
            patient_id = row['patient_id']
            t0_id = f"{patient_id}_11"
            tn_id = f"{patient_id}_21"
            delta = row['days_between']
            
            logger.info(f"--- Processing Patient [{idx+1}/{len(cohort_df)}]: {patient_id} ---")

            # Process Baseline (T0)
            t0_mask_path = process_single_visit(t0_id, root_path, model, model_cfg, device, saver, pseudo_labels_dir)
            
            # Process Follow-up (Tn)
            tn_mask_path = process_single_visit(tn_id, root_path, model, model_cfg, device, saver, pseudo_labels_dir)
            
            if not t0_mask_path or not tn_mask_path:
                logger.warning(f"Patient {patient_id} dropped due to incomplete modality inference.")
                continue

            t0_flair_path = root_path / "images_structural" / t0_id / f"{t0_id}_FLAIR.nii.gz"

            pinn_registry.append({
                "patient_id": patient_id,
                "image_t0_path": str(t0_flair_path.absolute()),
                "mask_t0_path": t0_mask_path,  # Now dynamically generated by the same model
                "mask_tn_path": tn_mask_path,  # Dynamically generated by the same model
                "days_between": delta
            })

    try:
        final_csv_path = target_dir / "dataset_registry_coherent.csv"
        pd.DataFrame(pinn_registry).to_csv(final_csv_path, index=False)
        logger.info(f"Batch harmonization complete. Coherent PINN registry established at {final_csv_path}")
    except Exception as e:
        logger.critical(f"Failed to compile final dataset registry: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute batch inference for coherent longitudinal pseudo-labels")
    parser.add_argument("--cohort_csv", type=str, required=True, help="Path to input cohort list")
    parser.add_argument("--nifti_root", type=str, required=True, help="Path to UPenn-GBM structural images")
    parser.add_argument("--config_dir", type=str, default="configs", help="Path to model config YAML")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to trained segmentation checkpoint")
    parser.add_argument("--out_dir", type=str, default="data", help="Target directory for outputs")
    
    args = parser.parse_args()
    execute_batch_harmonization(
        args.cohort_csv, 
        args.nifti_root, 
        args.config_dir, 
        args.ckpt_path, 
        args.out_dir
    )