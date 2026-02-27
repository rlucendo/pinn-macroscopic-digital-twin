import torch
import torch.nn as nn
from monai.networks.nets import UNet
import logging

# Professional structured logging
logger = logging.getLogger("BaselineStateExtractor")

class BaselineStateExtractor(nn.Module):
    """
    Static anatomical feature extractor for Hard-Physics PINN.
    Analyzes the T0 MRI to predict spatial maps for Diffusion (D) and Proliferation (rho).
    Temporal dynamics are explicitly decoupled from this module.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 2):
        super().__init__()
        logger.info(f"Initializing Static Extractor: in_channels={in_channels}, out_channels={out_channels}")
        
        # Standard 3D UNet architecture optimized for spatial feature extraction
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels, # Strictly 2 channels: D_map and rho_map
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch"
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass extracting physical parameters from static anatomy.
        
        Args:
            x: Input MRI tensor [Batch, Channels, Depth, Height, Width]
            
        Returns:
            raw_D: Unscaled diffusion logits [B, 1, D, H, W]
            raw_rho: Unscaled proliferation logits [B, 1, D, H, W]
        """
        # Execute UNet without time concatenation
        output = self.unet(x)
        
        # Split the 2 output channels into their respective physical variables
        raw_D = output[:, 0:1, ...]
        raw_rho = output[:, 1:2, ...]
        
        return raw_D, raw_rho