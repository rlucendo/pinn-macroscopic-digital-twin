import torch
import torch.nn as nn
from monai.networks.nets import UNet


class BaselineStateExtractor(nn.Module):
    """
    3D U-Net based spatial feature extractor for the Day 0 (T0) MRI scan.
    
    Unlike standard segmentation models that output class logits, this network 
    outputs a high-dimensional spatial embedding (e.g., 16 channels) that 
    preserves the original D x H x W resolution. This embedding serves as the 
    initial state for the PINN simulator.
    """

    def __init__(
        self, 
        in_channels: int = 1, 
        out_features: int = 16, 
        dropout: float = 0.1
    ):
        """
        Args:
            in_channels (int): Number of input MRI modalities (e.g., 4 for BraTS).
            out_features (int): Number of latent feature channels to output.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        
        # We leverage MONAI's heavily optimized 3D U-Net implementation.
        # Channels: We define the feature map sizes at each depth level.
        # Strides: Defines how the spatial dimensions are downsampled (2x2x2).
        self.feature_extractor = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_features,  # Outputting embeddings, not classes!
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,            # Residual connections inside each block
            norm="batch",
            dropout=dropout,
        )
        
        # Optional: An extra 1x1x1 Conv3d to smooth the features before passing 
        # them to the PINN simulator. Acts as a final projection layer.
        self.feature_projection = nn.Sequential(
            nn.Conv3d(out_features, out_features, kernel_size=1),
            nn.PReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract baseline spatial embeddings.
        
        Args:
            x (torch.Tensor): Raw T0 MRI volume [B, 4, D, H, W]
            
        Returns:
            torch.Tensor: Latent spatial features [B, 16, D, H, W]
        """
        # 1. Pass through the MONAI U-Net
        features = self.feature_extractor(x)
        
        # 2. Final non-linear projection
        embeddings = self.feature_projection(features)
        
        return embeddings