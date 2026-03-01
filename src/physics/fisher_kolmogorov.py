import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure structured logging
logger = logging.getLogger(__name__)

class FisherKolmogorovPDE(nn.Module):
    """
    Computes the instantaneous rate of change (du/dt) for Glioblastoma growth
    based on the Fisher-Kolmogorov reaction-diffusion equation.
    
    Equation: du/dt = div(D * grad(u)) + rho * u * (1 - u)
    """

    def __init__(self, voxel_spacing: float = 1.0):
        super().__init__()
        
        if voxel_spacing <= 0.0:
            logger.error("Invalid voxel_spacing initialized.", extra={"voxel_spacing": voxel_spacing})
            raise ValueError(f"voxel_spacing must be strictly positive, got {voxel_spacing}")
            
        self.voxel_spacing = voxel_spacing
        self.dx = voxel_spacing
        
        # Central difference kernels for computing gradients along axes
        # Shape: [out_channels, in_channels, D, H, W] -> [1, 1, 3, 3, 3]
        
        # X-axis gradient
        grad_x = torch.zeros((3, 3, 3), dtype=torch.float32)
        grad_x[1, 1, 0] = -0.5
        grad_x[1, 1, 2] = 0.5
                               
        # Y-axis gradient
        grad_y = torch.zeros((3, 3, 3), dtype=torch.float32)
        grad_y[1, 0, 1] = -0.5
        grad_y[1, 2, 1] = 0.5
                               
        # Z-axis gradient
        grad_z = torch.zeros((3, 3, 3), dtype=torch.float32)
        grad_z[0, 1, 1] = -0.5
        grad_z[2, 1, 1] = 0.5

        self.register_buffer("grad_x", grad_x.view(1, 1, 3, 3, 3))
        self.register_buffer("grad_y", grad_y.view(1, 1, 3, 3, 3))
        self.register_buffer("grad_z", grad_z.view(1, 1, 3, 3, 3))
        
        logger.info("FisherKolmogorovPDE initialized successfully.", 
                    extra={"voxel_spacing": self.voxel_spacing})

    def _compute_gradient(self, tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        Computes the spatial gradient using central differences with Neumann boundary conditions.
        """
        # Replicate padding enforces zero-flux Neumann boundary conditions
        padded_tensor = F.pad(tensor, (1, 1, 1, 1, 1, 1), mode='replicate')
        return F.conv3d(padded_tensor, kernel) / self.dx

    def compute_spatial_diffusion(self, u: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """
        Computes div(D * grad(u)) correctly accounting for heterogeneous D.
        """
        grad_u_x = self._compute_gradient(u, self.grad_x)
        grad_u_y = self._compute_gradient(u, self.grad_y)
        grad_u_z = self._compute_gradient(u, self.grad_z)
        
        flux_x = D * grad_u_x
        flux_y = D * grad_u_y
        flux_z = D * grad_u_z
        
        div_flux_x = self._compute_gradient(flux_x, self.grad_x)
        div_flux_y = self._compute_gradient(flux_y, self.grad_y)
        div_flux_z = self._compute_gradient(flux_z, self.grad_z)
        
        return div_flux_x + div_flux_y + div_flux_z

    def forward(self, u: torch.Tensor, D: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        if not (u.shape == D.shape == rho.shape):
            logger.error("Tensor shape mismatch during forward pass.")
            raise ValueError(f"Shape mismatch: u{u.shape}, D{D.shape}, rho{rho.shape}")

        diffusion_term = self.compute_spatial_diffusion(u, D)
        reaction_term = rho * u * (1.0 - u)
        
        return diffusion_term + reaction_term