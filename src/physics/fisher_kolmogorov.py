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
    
    Designed for Hard Physics differentiable solvers (Euler integration).
    """

    def __init__(self, voxel_spacing: float = 1.0):
        """
        Initializes the PDE solver module.
        
        Args:
            voxel_spacing (float): Isotropic voxel spacing in mm.
        """
        super().__init__()
        
        if voxel_spacing <= 0.0:
            logger.error("Invalid voxel_spacing initialized.", extra={"voxel_spacing": voxel_spacing})
            raise ValueError(f"voxel_spacing must be strictly positive, got {voxel_spacing}")
            
        self.voxel_spacing = voxel_spacing
        self.dx = voxel_spacing
        
        # Central difference kernels for computing gradients along axes
        # Shape: [out_channels, in_channels, D, H, W]
        grad_x = torch.tensor([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [-0.5, 0., 0.5], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]], dtype=torch.float32)
                               
        grad_y = torch.tensor([[[0., 0., 0.], [0., -0.5, 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0.5, 0.], [0., 0., 0.]]], dtype=torch.float32)
                               
        grad_z = torch.tensor([[[-0.5, 0., 0.5]], [[0., 0., 0.]], [[0., 0., 0.]]], dtype=torch.float32).view(3, 3, 3)
        # Fix z-axis kernel orientation
        grad_z = torch.tensor([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]], dtype=torch.float32)
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
        # 1. Compute gradients of u
        grad_u_x = self._compute_gradient(u, self.grad_x)
        grad_u_y = self._compute_gradient(u, self.grad_y)
        grad_u_z = self._compute_gradient(u, self.grad_z)
        
        # 2. Multiply by D to get the flux
        flux_x = D * grad_u_x
        flux_y = D * grad_u_y
        flux_z = D * grad_u_z
        
        # 3. Compute the divergence of the flux: div(F) = dx(Fx) + dy(Fy) + dz(Fz)
        div_flux_x = self._compute_gradient(flux_x, self.grad_x)
        div_flux_y = self._compute_gradient(flux_y, self.grad_y)
        div_flux_z = self._compute_gradient(flux_z, self.grad_z)
        
        return div_flux_x + div_flux_y + div_flux_z

    def forward(self, u: torch.Tensor, D: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward temporal derivative du/dt.
        
        Args:
            u (torch.Tensor): Current tumor concentration. Shape: [B, 1, Depth, Height, Width]
            D (torch.Tensor): Diffusion coefficient map. Shape: [B, 1, Depth, Height, Width]
            rho (torch.Tensor): Proliferation rate map. Shape: [B, 1, Depth, Height, Width]
            
        Returns:
            torch.Tensor: The temporal derivative du/dt. Shape: [B, 1, Depth, Height, Width]
        """
        if not (u.shape == D.shape == rho.shape):
            logger.error("Tensor shape mismatch during forward pass.", 
                         extra={"u_shape": list(u.shape), "D_shape": list(D.shape), "rho_shape": list(rho.shape)})
            raise ValueError(f"Shape mismatch: u{u.shape}, D{D.shape}, rho{rho.shape}")

        # 1. Spatial Diffusion Term
        diffusion_term = self.compute_spatial_diffusion(u, D)
        
        # 2. Reaction/Proliferation Term: rho * u * (1 - u)
        reaction_term = rho * u * (1.0 - u)
        
        # 3. Compute du/dt
        du_dt = diffusion_term + reaction_term
        
        return du_dt