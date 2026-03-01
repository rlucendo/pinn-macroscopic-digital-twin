import logging
import torch
import torch.nn as nn
from monai.networks.nets import BasicUNet

logger = logging.getLogger(__name__)

class BaselineStateExtractor(nn.Module):
    """
    Standard U-Net to extract dense parametric maps (D and rho) from multimodal MRI.
    """
    def __init__(self, in_channels: int = 4):
        super().__init__()
        
        # We enforce out_channels=2 to output both Diffusion (D) and Proliferation (rho)
        self.unet = BasicUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=2, # <--- VERIFICAR ESTE PARÁMETRO
            features=(16, 32, 64, 128, 256, 16),
            dropout=0.1
        )
        logger.info("Initializing BaselineStateExtractor.")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass outputting [Batch, 2, Depth, Height, Width]
        """
        return self.unet(x)