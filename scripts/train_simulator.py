import os
import argparse
import logging
import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from src.models.differentiable_solver import DifferentiablePDESolver
from src.models.unet_baseline import BaselineStateExtractor

# Domain imports (These must be present for the script to recognize our classes)
from src.data.longitudinal_dm import LongitudinalDataModule
from src.models.unet_baseline import BaselineStateExtractor
from src.models.pinn_simulator import PINNSimulator

from src.models.differentiable_solver import DifferentiablePDESolver

# Configure structured professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DigitalTwin_TheoreticalTrainer")


class TheoreticalGlioSimSystem(pl.LightningModule):
    """
    Hard-Physics Digital Twin Orchestrator.
    Binds the static feature extractor (UNet) with the dynamic PDE Solver (Euler Integration).
    """
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Neural Network: Extracts D and rho from static T0 MRI
        self.extractor = BaselineStateExtractor(in_channels=1, out_channels=2)
        
        # 2. Hard Physics Engine: Explicitly solves the PDE over time
        self.simulator = DifferentiablePDESolver(dt=1.0)

    def forward(self, x: torch.Tensor):
        """Standard forward pass purely for parameter extraction."""
        return self.extractor(x)

    def training_step(self, batch, batch_idx):
        image_t0 = batch["image_t0"]
        mask_t0 = batch["mask_t0"]
        mask_tn = batch["mask_tn"]
        delta_t = batch["time_delta"]

        # 1. Neural Network extraction
        raw_D, raw_rho = self(image_t0) 

        # 2. Scale to Swanson's biological ranges
        D_map = 0.001 + torch.sigmoid(raw_D) * (0.020 - 0.001)
        rho_map = 0.012 + torch.sigmoid(raw_rho) * (0.034 - 0.012)

        # 3. Hard Physics Simulation (Forward Euler)
        # Guarantees mathematical compliance while simulating future growth
        u_pred_tn = self.simulator(u_t0=mask_t0, D_map=D_map, rho_map=rho_map, delta_t=delta_t)

        # 4. Pure Data Assimilation Loss (Target Anchor)
        loss_total = F.mse_loss(u_pred_tn, mask_tn)

        self.log("train/loss_total", loss_total, prog_bar=True, on_step=False, on_epoch=True)
        return loss_total

    def validation_step(self, batch, batch_idx):
        image_t0 = batch["image_t0"]
        mask_t0 = batch["mask_t0"]
        mask_tn = batch["mask_tn"]
        delta_t = batch["time_delta"]

        # 1. Extract & Scale
        raw_D, raw_rho = self(image_t0)
        D_map = 0.001 + torch.sigmoid(raw_D) * (0.020 - 0.001)
        rho_map = 0.012 + torch.sigmoid(raw_rho) * (0.034 - 0.012)

        # 2. Simulate future
        u_pred_tn = self.simulator(u_t0=mask_t0, D_map=D_map, rho_map=rho_map, delta_t=delta_t)

        # 3. Validation Metrics
        val_loss = F.mse_loss(u_pred_tn, mask_tn)

        # Binary Dice Score Calculation
        u_pred_binary = (u_pred_tn > 0.5).float()
        intersection = torch.sum(u_pred_binary * mask_tn)
        union = torch.sum(u_pred_binary) + torch.sum(mask_tn)
        dice_score = (2.0 * intersection + 1e-8) / (union + 1e-8)

        self.log("val/loss_total", val_loss, sync_dist=True, prog_bar=True)
        self.log("val/dice_tn", dice_score, sync_dist=True, prog_bar=True)

        return {"val_loss": val_loss, "val_dice": dice_score}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main():
    parser = argparse.ArgumentParser(description="Train the Theoretical Digital Twin PINN")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset_registry.csv")
    parser.add_argument("--epochs", type=int, default=150, help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (1 is recommended for 3D volumes on 40GB VRAM)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    logger.info("Initializing Theoretical GlioSim Training Pipeline...")

    # Initialize Modules
    datamodule = LongitudinalDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    model = TheoreticalGlioSimSystem(lr=args.lr)

    # Callbacks for MLOps tracking
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",  # Bypass directo: string hardcodeado
        filename="gliosim-theory-{epoch:02d}-{val_dice_tn:.3f}",
        monitor="val/dice_tn",
        mode="max",
        save_top_k=3,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/dice_tn",
        patience=30,
        mode="max",
        verbose=True
    )

    # Integration with Weights & Biases
    wandb_logger = WandbLogger(project="GlioSim-Theoretical-Twin", log_model="all")

    # 1. Force TF32 for matrix multiplications (A100 specific)
    torch.set_float32_matmul_precision('high')

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        # Use bf16-mixed: It has the dynamic range of FP32 but the speed of FP16. 
        # Only available on A100/H100.
        precision="bf16-mixed", 
        gradient_clip_val=0.5,
        benchmark=True, # JIT kernel auto-tuner
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    logger.info("Theoretical engine ignition. Utilizing A100 TF32 Tensor Cores.")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the Differentiable PDE Solver Digital Twin.")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory.")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum number of epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    
    args = parser.parse_args()
    main()