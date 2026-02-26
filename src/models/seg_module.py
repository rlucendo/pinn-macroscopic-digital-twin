import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.handlers.utils import from_engine

class BraTSSegmentationModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # 1. Network Architecture: Standard 3D U-Net
        self.net = UNet(
            spatial_dims=3, # Fundamental requirement for volumetric data
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            channels=cfg.channels,
            strides=cfg.strides,
            num_res_units=2,
            norm="batch",
        )

        # 2. Hybrid Loss Function (Dice + Cross Entropy)
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

        # 3. Clinical Evaluation Metric
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        # 4. Post-processing transforms for metric calculation
        self.post_pred = AsDiscrete(argmax=True, to_onehot=cfg.out_channels)
        self.post_label = AsDiscrete(to_onehot=cfg.out_channels)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        
        loss = self.loss_function(outputs, labels)
        
        # Log training loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step receives the full anatomical volume. 
        Implements sliding_window_inference to extract, predict, and reassemble 
        patches seamlessly without exceeding GPU VRAM limits.
        """
        images, labels = batch["image"], batch["label"]
        
        # Execute sliding window inference
        roi_size = self.cfg.roi_size
        sw_batch_size = self.cfg.sw_batch_size
        outputs = sliding_window_inference(
            inputs=images, 
            roi_size=roi_size, 
            sw_batch_size=sw_batch_size, 
            predictor=self.forward
        )
        
        loss = self.loss_function(outputs, labels)
        
        # Post-process tensors for Dice score calculation (One-hot encoding)
        outputs_list = [self.post_pred(i) for i in outputs]
        labels_list = [self.post_label(i) for i in labels]
        
        # Update metric state
        self.dice_metric(y_pred=outputs_list, y=labels_list)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # Compute mean Dice score for the epoch and reset metric state
        mean_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        
        self.log("val/dice_score", mean_dice, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # AdamW optimization algorithm
        optimizer = torch.optim.AdamW(
            self.net.parameters(), 
            lr=self.cfg.learning_rate, 
            weight_decay=self.cfg.weight_decay
        )
        
        # Cosine Annealing learning rate scheduler to aid stable convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            }
        }