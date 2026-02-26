import torch
import torch.nn as nn
import torch.nn.functional as F


class FisherKolmogorovLoss(nn.Module):
    """
    Physics-Informed Neural Network (PINN) loss function for Glioblastoma growth.
    
    Computes the residual of the Fisher-Kolmogorov reaction-diffusion PDE:
    du/dt = div(D * grad(u)) + rho * u * (1 - u)
    
    The network is penalized if this residual deviates from zero, enforcing
    biological and physical plausibility in the spatiotemporal predictions.
    """

    def __init__(self, voxel_spacing: float = 1.0):
        """
        Args:
            voxel_spacing (float): The physical distance between voxels in mm.
                                   Assumes isotropic spacing (1x1x1 mm) after ETL.
        """
        super().__init__()
        self.voxel_spacing = voxel_spacing
        
        # 3D Laplacian kernel using finite central differences (7-point stencil)
        # Shape: [out_channels, in_channels, depth, height, width]
        laplacian_kernel = torch.tensor([
            [[0.0,  0.0, 0.0],
             [0.0,  1.0, 0.0],
             [0.0,  0.0, 0.0]],

            [[0.0,  1.0, 0.0],
             [1.0, -6.0, 1.0],
             [0.0,  1.0, 0.0]],

            [[0.0,  0.0, 0.0],
             [0.0,  1.0, 0.0],
             [0.0,  0.0, 0.0]]
        ], dtype=torch.float32)
        
        # Reshape for F.conv3d (1, 1, 3, 3, 3)
        self.register_buffer(
            "laplacian_kernel", 
            laplacian_kernel.view(1, 1, 3, 3, 3)
        )

    def compute_spatial_diffusion(self, u: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """
        Computes div(D * grad(u)) using a 3D Laplacian convolution.
        
        Args:
            u (torch.Tensor): Tumor concentration [B, 1, D, H, W]
            D (torch.Tensor): Diffusion coefficient map [B, 1, D, H, W]
            
        Returns:
            torch.Tensor: The spatial diffusion term.
        """
        # Apply the Laplacian filter to calculate grad^2(u)
        # Padding=1 ensures the output tensor shape matches the input
        laplacian_u = F.conv3d(u, self.laplacian_kernel, padding=1)
        
        # Adjust for physical voxel spacing: dx^2
        laplacian_u = laplacian_u / (self.voxel_spacing ** 2)
        
        # div(D * grad(u)) ≈ D * Laplacian(u) assuming D varies slowly locally
        diffusion_term = D * laplacian_u
        
        return diffusion_term

    def forward(
        self, 
        u: torch.Tensor, 
        du_dt: torch.Tensor, 
        D: torch.Tensor, 
        rho: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the Mean Squared Error (MSE) of the PDE residual.
        
        Args:
            u (torch.Tensor): Predicted tumor concentration (0 to 1). Shape: [B, 1, D, H, W]
            du_dt (torch.Tensor): Predicted temporal derivative. Shape: [B, 1, D, H, W]
            D (torch.Tensor): Predicted/Learned diffusion map. Shape: [B, 1, D, H, W]
            rho (torch.Tensor): Predicted/Learned proliferation map. Shape: [B, 1, D, H, W]
            
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Validate tensor shapes to prevent silent broadcasting bugs
        assert u.shape == du_dt.shape == D.shape == rho.shape, \
            f"Shape mismatch: u{u.shape}, du_dt{du_dt.shape}, D{D.shape}, rho{rho.shape}"
            
        # 2. Calculate the Spatial Diffusion Term
        diffusion_term = self.compute_spatial_diffusion(u, D)
        
        # 3. Calculate the Reaction/Proliferation Term: rho * u * (1 - u)
        reaction_term = rho * u * (1.0 - u)
        
        # 4. Compute the PDE Residual: du/dt - (Diffusion + Reaction) = 0
        pde_residual = du_dt - (diffusion_term + reaction_term)
        
        # 5. The loss is the Mean Squared Error of the residual (we want it to be 0)
        loss_pde = torch.mean(pde_residual ** 2)
        
        return loss_pde