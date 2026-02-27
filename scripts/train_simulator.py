import os
import argparse
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# Domain imports
from src.data.longitudinal_dm import LongitudinalDataModule
from src.models.unet_baseline import BaselineStateExtractor
from src.models.pinn_simulator import PINNSimulator

# Configure structured professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DigitalTwin_TheoreticalTrainer")


class TheoreticalGlioSimSystem(pl.LightningModule):
    """
    LightningModule orchestrating the Physics-Informed Neural Network.
    Optimizes the neural weights to solve the Fisher-Kolmogorov PDE 
    without relying on future ground-truth data.
    """
    def __init__(self, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Neural architecture for feature extraction
        self.extractor = BaselineStateExtractor(in_channels=1, spatial_dims=3)
        
        # Theoretical PDE Solver
        self.simulator = PINNSimulator()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Executes the neural forward pass.
        Returns the continuous tumor density (u), and the raw biological tensors (D, rho).
        """
        # The extractor must handle broadcasting or concatenating the scalar 't'
        # into the spatial volume to compute the state at time 't'.
        u_pred, raw_D, raw_rho = self.extractor(x, t)
        return u_pred, raw_D, raw_rho

    def training_step(self, batch, batch_idx):
        """
        Executes the physics-constrained training loop.
        """
        image_t0 = batch["image_t0"]
        mask_t0 = batch["mask_t0"]
        batch_size = image_t0.shape[0]

        # 1. Evaluate Initial Condition (t = 0)
        # Time tensor must match the batch dimension, shaped [B, 1]
        t0 = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)
        u_pred_t0, raw_D, raw_rho = self(image_t0, t0)

        # 2. Evaluate Physics at random continuous time
        # Generate t ~ U(1.0, 300.0) days
        t_rand = torch.empty((batch_size, 1), device=self.device, dtype=torch.float32).uniform_(1.0, 300.0)
        t_rand.requires_grad = True
        
        u_pred_t, _, _ = self(image_t0, t_rand)

        # 3. Compute Theoretical Loss via PINN Simulator
        total_loss, loss_ic, loss_physics = self.simulator.calculate_loss(
            u_pred_t=u_pred_t,
            u_pred_t0=u_pred_t0,
            target_t0=mask_t0,
            t=t_rand,
            raw_D=raw_D,
            raw_rho=raw_rho
        )

        # 4. Telemetry and Logging
        self.log("train/loss_total", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_ic", loss_ic, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_physics", loss_physics, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        """
        Configures the AdamW optimizer with weight decay for regularization.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=150, 
            eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


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
    model = TheoreticalGlioSimSystem(learning_rate=args.lr)

    # Callbacks for MLOps tracking
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="gliosim-theoretical-{epoch:03d}-{train/loss_total:.4f}",
        monitor="train/loss_total",
        save_top_k=3,
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="train/loss_physics",
        patience=30,
        mode="min",
        verbose=True
    )

    # Integration with Weights & Biases
    wandb_logger = WandbLogger(project="GlioSim-Theoretical-Twin", log_model="all")

    # Trainer configuration for NVIDIA A100
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=5,
        enable_progress_bar=True
    )

    logger.info("Commencing theoretical PDE optimization loop.")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    # Force PyTorch to use Tensor Cores for matrix multiplication
    torch.set_float32_matmul_precision('high')
    main()