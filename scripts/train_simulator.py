import argparse
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# Import our custom domain modules
from src.data.longitudinal_dm import LongitudinalDataModule
from src.models.unet_baseline import BaselineStateExtractor
from src.models.pinn_simulator import PINNSimulator


class GlioSimSystem(L.LightningModule):
    """
    The orchestrator module that composes the Baseline Extractor and the PINN Simulator
    into a single end-to-end trainable system.
    """
    def __init__(self, in_channels: int = 1, lambda_pde: float = 0.1, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Instantiate the modules
        # The U-Net extracts 16 latent channels from the 4 MRI modalities
        self.extractor = BaselineStateExtractor(in_channels=in_channels, out_features=16)
        
        # The PINN Simulator takes the 16 channels + 1 time channel
        self.simulator = PINNSimulator(
            in_channels=16, 
            lambda_pde=lambda_pde, 
            learning_rate=lr
        )

    def forward(self, x_t0: torch.Tensor, t: torch.Tensor):
        """
        End-to-end forward pass: Image T0 + time -> Future State Tn
        """
        # Extract baseline topological state
        spatial_features = self.extractor(x_t0)
        
        # Predict future state and physical parameters
        u, dudt, D, rho = self.simulator(spatial_features, t)
        return u, dudt, D, rho

    def training_step(self, batch, batch_idx):
        """
        Defines the end-to-end training loop using our PDE Loss.
        """
        # Batch unpacking based on our LongitudinalDataModule transforms
        # MONAI dictionaries output the keys we defined in the ETL pipeline
        x_t0 = batch["image_t0"]
        target_tn = batch["mask_tn"]
        t = batch["time_delta"]
        
        # Forward pass through the composed system
        pred_u, pred_dudt, pred_D, pred_rho = self(x_t0, t)
        
        # Calculate losses (delegating to the simulator's loss functions)
        loss_data = self.simulator.data_loss_fn(pred_u, target_tn)
        loss_physics = self.simulator.physics_loss_fn(pred_u, pred_dudt, pred_D, pred_rho)
        
        # Total Loss formulation: L_Total = L_Data + lambda * L_PDE
        loss_total = loss_data + (self.hparams.lambda_pde * loss_physics)
        
        # Logging for Weights & Biases
        self.log("train/loss_data", loss_data, on_step=True, on_epoch=True)
        self.log("train/loss_physics", loss_physics, on_step=True, on_epoch=True)
        self.log("train/loss_total", loss_total, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss_total

    def configure_optimizers(self):
        # We optimize both the extractor and the simulator simultaneously (End-to-End)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss_total"
            }
        }


def main():
    # 1. Parse arguments (mocked for simplicity, ideally loaded from config_pinn.yaml)
    parser = argparse.ArgumentParser(description="Train GlioSim PINN")
    parser.add_argument("--data_dir", type=str, default="data/", help="Path to NIfTI data")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (1 for 3D)")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    args = parser.parse_args()

    # 2. Initialize DataModule
    datamodule = LongitudinalDataModule(
        data_dir=args.data_dir, 
        batch_size=args.batch_size
    )

    # 3. Initialize the Composed System
    model = GlioSimSystem(in_channels=1, lambda_pde=0.1, lr=1e-4)

    # 4. Setup MLOps Callbacks & Logger
    wandb_logger = WandbLogger(project="GlioSim-Digital-Twin", name="PINN-End-to-End")
    
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/",
            filename="gliosim-{epoch:02d}-{train/loss_total:.4f}",
            monitor="train/loss_total",
            mode="min",
            save_top_k=3,
        ),
        EarlyStopping(monitor="train/loss_total", patience=20, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # 5. Initialize Lightning Trainer
    # Hardware resilience: mixed precision for memory efficiency
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=5,
    )

    # 6. Start Training!
    print("Starting GlioSim Digital Twin Training Pipeline...")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()