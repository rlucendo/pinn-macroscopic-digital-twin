import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Configure structured professional logging
logger = logging.getLogger("Theoretical_PINN_Simulator")


class SwansonConstants:
    """
    Biological constants for Glioblastoma Multiforme (GBM) growth
    based on the Swanson mathematical macroscopic model.
    Values represent daily progression rates.
    """
    RHO_MIN = 0.012
    RHO_MAX = 0.034
    D_MIN = 0.001
    D_MAX = 0.020


class PINNSimulator(nn.Module):
    """
    Physics-Informed Neural Network Simulator for theoretical tumor progression.
    Solves the Fisher-Kolmogorov PDE using a hybrid approach:
    - Autograd for the temporal derivative (dudt)
    - 3D Finite Differences for the spatial Laplacian (nabla^2 u)
    """

    def __init__(self, device: torch.device = None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._register_laplacian_kernel()
        logger.info("Theoretical Fisher-Kolmogorov PDE Solver initialized.")

    def _register_laplacian_kernel(self) -> None:
        """
        Initializes a fixed 3D Laplacian kernel for spatial finite differences.
        This approximates nabla^2 u without the extreme memory overhead of 3D spatial autograd.
        """
        # Standard 7-point 3D stencil for the discrete Laplacian operator
        kernel = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32)
        kernel[0, 0, 1, 1, 1] = -6.0
        
        # Z-axis
        kernel[0, 0, 0, 1, 1] = 1.0
        kernel[0, 0, 2, 1, 1] = 1.0
        # Y-axis
        kernel[0, 0, 1, 0, 1] = 1.0
        kernel[0, 0, 1, 2, 1] = 1.0
        # X-axis
        kernel[0, 0, 1, 1, 0] = 1.0
        kernel[0, 0, 1, 1, 2] = 1.0

        # Register as a buffer so it moves to the correct device automatically 
        # but is not updated by the optimizer
        self.register_buffer("laplacian_kernel", kernel)

    def constrain_biological_tensors(self, raw_D: torch.Tensor, raw_rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Maps raw neural network outputs to strict biological ranges.
        Prevents the PDE from exploding due to non-physical coefficients.
        """
        # Scale D to [0.001, 0.020]
        constrained_D = SwansonConstants.D_MIN + torch.sigmoid(raw_D) * (SwansonConstants.D_MAX - SwansonConstants.D_MIN)
        
        # Scale rho to [0.012, 0.034]
        constrained_rho = SwansonConstants.RHO_MIN + torch.sigmoid(raw_rho) * (SwansonConstants.RHO_MAX - SwansonConstants.RHO_MIN)
        
        return constrained_D, constrained_rho

    def compute_spatial_laplacian(self, u: torch.Tensor) -> torch.Tensor:
        """
        Computes the spatial second derivative using the 3D convolution kernel.
        """
        # Ensure padding avoids border artifacts
        return F.conv3d(u, self.laplacian_kernel, padding=1)

    def compute_pde_residual(
        self, 
        u_pred: torch.Tensor, 
        t: torch.Tensor, 
        D_map: torch.Tensor, 
        rho_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluates the Fisher-Kolmogorov partial differential equation.
        Returns the residual field. A residual of 0 means perfect adherence to physics.
        
        Equation: du/dt = D * nabla^2 u + rho * u * (1 - u)
        """
        if not t.requires_grad:
            logger.error("Time tensor 't' must have requires_grad=True to compute temporal derivatives.")
            raise RuntimeError("Missing gradients on time tensor.")

        # 1. Exact Temporal Derivative via Automatic Differentiation
        # grad_outputs forces the derivative to be computed for each voxel
        dudt = torch.autograd.grad(
            outputs=u_pred,
            inputs=t,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
            retain_graph=True
        )[0]

        # 2. Spatial Derivatives
        laplacian_u = self.compute_spatial_laplacian(u_pred)

        # 3. Fisher-Kolmogorov Formulation
        diffusion_term = D_map * laplacian_u
        proliferation_term = rho_map * u_pred * (1.0 - u_pred)

        # Residual calculation (Rearranging to: dudt - D*lap_u - rho*u*(1-u) = 0)
        residual = dudt - diffusion_term - proliferation_term

        return residual

    def calculate_loss(
        self, 
        u_pred_t: torch.Tensor, 
        u_pred_t0: torch.Tensor, 
        target_t0: torch.Tensor, 
        t: torch.Tensor, 
        raw_D: torch.Tensor, 
        raw_rho: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the composite loss function for the Forward Simulation PINN.
        """
        # Enforce biological constraints on the raw network outputs
        D_map, rho_map = self.constrain_biological_tensors(raw_D, raw_rho)

        # LOSS 1: Initial Condition (Dirichlet Boundary)
        # The network must perfectly reconstruct the known baseline mask at t=0
        loss_ic = F.mse_loss(u_pred_t0, target_t0)

        # LOSS 2: Physics Infringement (PDE Residual)
        # The network must obey the theoretical growth dynamics for any time t > 0
        residual_field = self.compute_pde_residual(u_pred_t, t, D_map, rho_map)
        
        # Calculate Mean Squared Error of the residual (Aiming for 0)
        loss_physics = torch.mean(residual_field ** 2)

        # Total weighted loss
        # Weighting the IC higher usually stabilizes the PDE solver early in training
        lambda_ic = 10.0
        lambda_phys = 1.0
        
        total_loss = (lambda_ic * loss_ic) + (lambda_phys * loss_physics)

        return total_loss, loss_ic, loss_physics