import torch
import torch.nn as nn
from monai.networks.nets import UNet

class BaselineStateExtractor(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        
        # Aumentamos in_channels a +1 para recibir el "canal del tiempo"
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels + 1, # 1 (MRI) + 1 (Time)
            out_channels=3, # Salidas: [u_pred, raw_D, raw_rho]
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        High-performance forward pass optimized for A100 memory alignment.
        """
        # Optimized broadcasting
        batch_size, _, d, h, w = x.shape
        
        # Pre-allocate time volume on the same device
        t_volume = t.view(batch_size, 1, 1, 1, 1).expand(-1, -1, d, h, w)
        
        # Single concatenation operation
        x_input = torch.cat([x, t_volume], dim=1)
        
        # Execute UNet
        output = self.unet(x_input)
        
        # Slice and activate
        u_pred = torch.sigmoid(output[:, 0:1, ...])
        raw_D = output[:, 1:2, ...]
        raw_rho = output[:, 2:3, ...]
        
        return u_pred, raw_D, raw_rho