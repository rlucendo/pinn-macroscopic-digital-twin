import os
import argparse
import torch
from pathlib import Path

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    EnsureTyped,
    Invertd,
    SaveImaged,
    AsDiscreted,
    DivisiblePadd,
)
from monai.data import Dataset, DataLoader, decollate_batch

# Import our trained system
from scripts.train_simulator import GlioSimSystem


def run_digital_twin_simulation(
    input_mri_path: str, 
    ckpt_path: str, 
    days_forward: float, 
    output_dir: str
):
    """
    Executes the Macroscopic Digital Twin forward simulation.
    Takes a baseline MRI, advances time by 'days_forward', and outputs a clinical NIfTI mask.
    """
    print(f"Initializing GlioSim Digital Twin...")
    print(f"Patient Baseline: {input_mri_path}")
    print(f"Simulating Forward: {days_forward} days")
    
    # 1. Hardware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load the trained model from the checkpoint
    # strict=False allows us to load even if some hyperparameters changed slightly
    model = GlioSimSystem.load_from_checkpoint(ckpt_path, strict=False, weights_only=False)
    model.eval()
    model.to(device)
    
    # 3. Define the Pre-processing Transforms (Must match training exactly)
    # We use a dictionary-based approach to keep track of the metadata for inversion
    preprocess_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        DivisiblePadd(keys=["image"], k=16),
        EnsureTyped(keys=["image"]),
    ])
    
    # 4. Prepare the Data
    # We package the input into a list of dictionaries for MONAI
    data_dict = [{"image": input_mri_path}]
    dataset = Dataset(data=data_dict, transform=preprocess_transforms)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    # 5. Define Post-processing & Inversion Transforms
    # This is the Senior MLOps touch: returning the prediction to the original physical space
    postprocess_transforms = Compose([
        EnsureTyped(keys=["pred"]),
        # Binarize the probability map at a 0.5 threshold
        AsDiscreted(keys=["pred"], threshold=0.5),
        # Invert the Spacing and Orientation applied during preprocessing
        Invertd(
            keys=["pred"],
            transform=preprocess_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        ),
        # Save to disk as a clinically compliant NIfTI file
        SaveImaged(
            keys=["pred"], 
            meta_keys="pred_meta_dict", 
            output_dir=output_dir, 
            output_postfix=f"simulated_day_{int(days_forward)}", 
            resample=False,
            separate_folder=False
        )
    ])

    # 6. Execute the Simulation Loop
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_data in dataloader:
            # Move tensor to GPU
            inputs = batch_data["image"].to(device)
            
            # Format the time parameter as a tensor
            t_tensor = torch.tensor([days_forward], dtype=torch.float32, device=device)
            
            # Forward pass through the Digital Twin architecture
            pred_u, _, _, _ = model(inputs, t_tensor)
            
            # Store the prediction back into the dictionary
            batch_data["pred"] = pred_u.cpu()
            
            # 7. Apply Post-processing (Inversion & Saving)
            # CRITICAL: We must decollate the batch back into individual items 
            # so the Inverse transforms can read the metadata dictionaries properly.
            for item in decollate_batch(batch_data):
                item = postprocess_transforms(item)
            
    print(f"Simulation Complete. Results saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GlioSim: Digital Twin Inference")
    parser.add_argument("--input_mri", type=str, required=True, help="Path to Day 0 NIfTI MRI")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained PyTorch Lightning checkpoint")
    parser.add_argument("--days", type=float, default=30.0, help="Days into the future to simulate")
    parser.add_argument("--out_dir", type=str, default="simulations/", help="Output directory")
    
    args = parser.parse_args()
    
    run_digital_twin_simulation(
        input_mri_path=args.input_mri,
        ckpt_path=args.ckpt,
        days_forward=args.days,
        output_dir=args.out_dir
    )