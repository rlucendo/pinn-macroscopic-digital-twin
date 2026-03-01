import logging
import torch
import torch.nn as nn
from monai.networks.nets import BasicUNet

# Configure structured logging
logger = logging.getLogger(__name__)

class BaselineStateExtractor(nn.Module):
    """
    Anatomical feature extractor for Hard-Physics PINN.
    
    Analyzes static multi-modal MRI inputs (e.g., T0, White Matter masks) 
    to predict smooth, biologically plausible spatial maps for Diffusion (D) 
    and Proliferation (rho).
    """

    def __init__(self, in_channels: int = 1):
        """
        Initializes the state extractor with strict upsampling rules.
        
        Args:
            in_channels (int): Number of input modalities. Defaults to 1 for just T0, 
                               but should be increased if concatenating anatomical priors (WM/GM).
        """
        super().__init__()
        
        # We output exactly 2 channels: [D_map, rho_map]
        out_channels = 2 
        
        logger.info(
            "Initializing BaselineStateExtractor.", 
            extra={"in_channels": in_channels, "out_channels": out_channels}
        )
        
        # Using BasicUNet with upsample="nontrainable" forces Trilinear Interpolation
        # instead of ConvTranspose3d, effectively eliminating checkerboard artifacts.
        self.unet = BasicUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            features=(16, 32, 64, 128, 256, 16), # Final feature map before output
            upsample="nontrainable" # CRITICAL: Trilinear interpolation + Conv3d
        )
        
        # Physics-constrained activations
        # Softplus ensures Diffusion is strictly positive and avoids dead gradients from ReLU
        self.diffusion_activation = nn.Softplus()
        
        # Sigmoid ensures Proliferation rate stays within [0, 1] bounds
        self.proliferation_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts bounded physical parameters from structural MRI.
        
        Args:
            x (torch.Tensor): Input tensor. Shape: [B, C, Depth, Height, Width]
            
        Returns:
            tuple:
                - D_map (torch.Tensor): Bounded diffusion map (> 0). Shape: [B, 1, D, H, W]
                - rho_map (torch.Tensor): Bounded proliferation map (0 to 1). Shape: [B, 1, D, H, W]
        """
        # 1. Forward pass through the artifact-free UNet
        output = self.unet(x)
        
        if output.shape[1] != 2:
            logger.error(f"Expected 2 output channels, got {output.shape[1]}")
            raise RuntimeError("UNet output channel mismatch.")
            
        # 2. Split feature maps
        raw_D = output[:, 0:1, ...]
        raw_rho = output[:, 1:2, ...]
        
        # 3. Enforce physical bounds via activations
        # D > 0 (prevents backward time diffusion)
        D_map = self.diffusion_activation(raw_D)
        
        # 0 <= rho <= 1 (normalized biological growth rate)
        rho_map = self.proliferation_activation(raw_rho)
        
        return D_map, rho_map