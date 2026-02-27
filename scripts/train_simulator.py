import os
import argparse
import logging
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

# Domain imports (These must be present for the script to recognize our classes)
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
        self.extractor = BaselineStateExtractor(in_channels=1)
        
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
        image_t0 = batch["image_t0"]
        mask_t0 = batch["mask_t0"]
        batch_size = image_t0.shape[0]

        # 1. Initial Condition (t=0)
        t0 = torch.zeros((batch_size, 1), device=self.device)
        u_pred_t0, raw_D, raw_rho = self(image_t0, t0)

        # 2. Physics Condition (t=random)
        # We sample a random time for each batch to enforce the PDE across the timeline
        t_rand = torch.empty((batch_size, 1), device=self.device).uniform_(1.0, 100.0)
        t_rand.requires_grad = True 
        u_pred_t, _, _ = self(image_t0, t_rand)

        # 3. Comprehensive Loss Calculation
        total_loss, loss_ic, loss_physics = self.simulator.calculate_loss(
            u_pred_t=u_pred_t,
            u_pred_t0=u_pred_t0,
            target_t0=mask_t0,
            t=t_rand,
            raw_D=raw_D,
            raw_rho=raw_rho
        )

        # 4. LOGGING (This satisfies EarlyStopping and ModelCheckpoint)
        # sync_dist=True is good practice for multi-GPU, though here we are on 1x A100
        self.log("train/loss_total", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/loss_ic", loss_ic, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/loss_physics", loss_physics, prog_bar=True, on_step=False, on_epoch=True)

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
    # Force PyTorch to use Tensor Cores for matrix multiplication
    torch.set_float32_matmul_precision('high')
    main()