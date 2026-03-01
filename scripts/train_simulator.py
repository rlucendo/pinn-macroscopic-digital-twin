import argparse
import logging
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

# Domain imports
from src.data.cross_sectional_dm import CrossSectionalDataModule
from src.models.unet_baseline import BaselineStateExtractor
from src.models.differentiable_solver import DifferentiableEulerSolver
from src.physics.fisher_kolmogorov import FisherKolmogorovPDE
from src.models.pinn_simulator import MacroscopicDigitalTwin

# Configure structured professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SelfSupervised_Trainer")

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, u_pred: torch.Tensor, u_target: torch.Tensor) -> torch.Tensor:
        u_pred_f = u_pred.view(-1)
        u_target_f = u_target.view(-1)
        intersection = torch.sum(u_pred_f * u_target_f)
        cardinality = torch.sum(u_pred_f) + torch.sum(u_target_f)
        return 1.0 - ((2.0 * intersection + self.smooth) / (cardinality + self.smooth))

class SelfSupervisedGlioSim(pl.LightningModule):
    """
    Amortized Inference System for Glioblastoma.
    Learns to deduce D and rho by growing a virtual seed until it matches the patient's T0 MRI.
    """
    def __init__(self, lr: float = 1e-4, simulation_horizon: float = 100.0):
        super().__init__()
        self.save_hyperparameters()
        self.simulation_horizon = simulation_horizon
        
        pde_module = FisherKolmogorovPDE(voxel_spacing=1.0)
        # The UNet now receives a 4-channel tensor [FLAIR, T1, T1GD, T2]
        extractor = BaselineStateExtractor(in_channels=4)
        
        # Substeps are critical here. Growing from a seed requires stable gradients.
        solver = DifferentiableEulerSolver(
            pde_module=pde_module, 
            time_step_days=1.0, 
            numerical_substeps=15
        )
        
        self.digital_twin = MacroscopicDigitalTwin(extractor_module=extractor, physics_solver=solver)
        self.mse_loss = torch.nn.MSELoss()
        self.dice_loss = DiceLoss()

    def _generate_virtual_seed(self, mask_t0: torch.Tensor, seed_radius: float = 2.0) -> torch.Tensor:
        """
        Calculates the center of mass of the target mask and plants a spherical seed.
        This represents the origin point of the tumor in the brain.
        """
        B, C, D, H, W = mask_t0.shape
        seed = torch.zeros_like(mask_t0)
        
        for b in range(B):
            # Find foreground voxels
            indices = torch.nonzero(mask_t0[b, 0])
            
            if len(indices) == 0:
                # Fallback to absolute center if mask is completely empty
                center = torch.tensor([D//2, H//2, W//2], device=mask_t0.device)
            else:
                # Center of mass
                center = indices.float().mean(dim=0).long()
                
            # Create a coordinate grid
            z, y, x = torch.meshgrid(
                torch.arange(D, device=mask_t0.device),
                torch.arange(H, device=mask_t0.device),
                torch.arange(W, device=mask_t0.device),
                indexing='ij'
            )
            
            # Calculate squared distance from center
            dist_sq = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2
            
            # Plant the seed
            seed[b, 0, dist_sq <= seed_radius**2] = 1.0
            
        return seed

    def _compute_losses(self, u_pred: torch.Tensor, mask_target: torch.Tensor) -> dict:
        loss_dice = self.dice_loss(u_pred, mask_target)
        loss_mse = self.mse_loss(u_pred, mask_target)
        loss_total = (0.7 * loss_dice) + (0.3 * loss_mse)
        return {"total": loss_total, "dice": loss_dice, "mse": loss_mse}

    def training_step(self, batch, batch_idx):
        image_t0 = batch["image_t0"]
        mask_t0 = batch["mask_t0"] # This is our ground-truth target now
        
        # 1. Generate the origin seed
        virtual_seed = self._generate_virtual_seed(mask_t0)
        
        # 2. Fixed temporal horizon for the entire batch
        B = image_t0.shape[0]
        delta_t = torch.full((B,), self.simulation_horizon, dtype=torch.float32, device=self.device)

        # 3. Forward Pass: Extract D/rho from T0, and simulate from SEED to T0_pred
        # Notice we pass `virtual_seed` to the solver, not `mask_t0`.
        u_pred_t0, _ = self.digital_twin(image_t0, virtual_seed, delta_t)

        # 4. Compare the grown seed to the actual patient mask
        losses = self._compute_losses(u_pred_t0, mask_t0)

        self.log("train/loss_total", losses["total"], prog_bar=True)
        self.log("train/loss_dice", losses["dice"])
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        image_t0 = batch["image_t0"]
        mask_t0 = batch["mask_t0"]
        
        virtual_seed = self._generate_virtual_seed(mask_t0)
        B = image_t0.shape[0]
        delta_t = torch.full((B,), self.simulation_horizon, dtype=torch.float32, device=self.device)

        u_pred_t0, _ = self.digital_twin(image_t0, virtual_seed, delta_t)

        losses = self._compute_losses(u_pred_t0, mask_t0)

        # Validation Metrics
        pred_binary = (u_pred_t0 > 0.2).float()
        target_binary = (mask_t0 > 0.2).float()
        
        intersection = torch.sum(pred_binary * target_binary)
        union = torch.sum(pred_binary) + torch.sum(target_binary)
        val_dice = (2.0 * intersection + 1e-8) / (union + 1e-8)

        self.log("val/loss_total", losses["total"], sync_dist=True, prog_bar=True)
        self.log("val/dice_global", val_dice, sync_dist=True, prog_bar=True)

        return {"val_loss": losses["total"], "val_dice": val_dice}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)

def main(args: argparse.Namespace) -> None:
    logger.info("Initializing Amortized Inference Pipeline...")

    # If in fast_dev_run, we don't want WandB to log a fake 1-step experiment
    logger_type = WandbLogger(project="GlioSim-Amortized-Twin", log_model="all") if not args.fast_dev_run else False

    datamodule = CrossSectionalDataModule(
        data_dir=args.data_dir, 
        batch_size=args.batch_size
    )
    
    model = SelfSupervisedGlioSim(
        lr=args.lr, 
        simulation_horizon=args.simulation_horizon
    )

    local_checkpoint_dir = "checkpoints"
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=local_checkpoint_dir,
        filename="amortized-twin-{epoch:02d}-{val_dice_global:.3f}",
        monitor="val/dice_global",
        mode="max",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/dice_global",
        patience=20,
        mode="max",
        verbose=True
    )

    torch.set_float32_matmul_precision('high')

    # Add the fast_dev_run flag to the Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed", 
        gradient_clip_val=1.0,
        benchmark=True,
        logger=logger_type,
        callbacks=[checkpoint_callback, early_stop_callback],
        fast_dev_run=args.fast_dev_run
    )

    mode = "DRY RUN (Smoke Test)" if args.fast_dev_run else "FULL TRAINING"
    logger.info(f"Igniting self-supervised simulation engine. Mode: {mode}")
    
    try:
        trainer.fit(model, datamodule=datamodule)
    except Exception as e:
        logger.error("Training pipeline encountered a critical failure.", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Amortized Digital Twin")
    
    # Hiperparámetros de control total
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset")
    parser.add_argument("--epochs", type=int, default=150, help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--simulation_horizon", type=float, default=50.0, help="Fixed virtual days")
    
    # Interruptor para el Dry-Run (Prueba rápida)
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 1 batch of train and val to find bugs quickly")
    
    parsed_args = parser.parse_args()
    main(parsed_args)