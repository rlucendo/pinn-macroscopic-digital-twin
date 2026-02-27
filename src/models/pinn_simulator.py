import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Senior-level structured logging
logger = logging.getLogger("PhysicsEngine_FullVolume")

class PINNSimulator(nn.Module):
    """
    High-fidelity Physics-Informed Neural Network Simulator.
    Computes the Fisher-Kolmogorov PDE residual across the entire 3D volume
    to ensure global physical consistency.
    """
    def __init__(self):
        super().__init__()
        self._register_laplacian_kernel()
        logger.info("Full-Volume PDE Solver initialized. Accuracy prioritized over throughput.")

    def _register_laplacian_kernel(self):
        """Discrete 3D Laplacian operator (7-point stencil)."""
        kernel = torch.zeros((1, 1, 3, 3, 3))
        kernel[0, 0, 1, 1, 1] = -6.0
        # Axial neighbors (6-connectivity)
        kernel[0, 0, 0, 1, 1] = 1.0; kernel[0, 0, 2, 1, 1] = 1.0
        kernel[0, 0, 1, 0, 1] = 1.0; kernel[0, 0, 1, 2, 1] = 1.0
        kernel[0, 0, 1, 1, 0] = 1.0; kernel[0, 0, 1, 1, 2] = 1.0
        self.register_buffer("laplacian", kernel)

    def compute_pde_residual(self, u_pred: torch.Tensor, t: torch.Tensor, d_map: torch.Tensor, rho_map: torch.Tensor):
        """
        Calculates the physics residual for every single voxel in the [96, 96, 96] grid.
        Includes dimensional broadcasting for proper full-tensor subtraction.
        """
        batch_size, _, d, h, w = u_pred.shape
        
        # 1. Spatial Domain: Full-volume Laplacian
        laplacian_u = F.conv3d(u_pred, self.laplacian, padding=1)

        # 2. Temporal Domain: Full-volume Autograd
        # Autograd returns a tensor matching the shape of inputs (t), which is [B, 1].
        dudt_scalar = torch.autograd.grad(
            outputs=u_pred,
            inputs=t,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # 3. Broadcasting: Expand the [B, 1] temporal derivative back to [B, 1, D, H, W]
        # This ensures every voxel experiences the same dt effect at that specific snapshot
        dudt_volume = dudt_scalar.view(batch_size, 1, 1, 1, 1).expand(-1, -1, d, h, w)

        # 4. Fisher-Kolmogorov PDE
        diffusion_term = d_map * laplacian_u
        proliferation_term = rho_map * u_pred * (1.0 - u_pred)
        
        # Now all tensors are strictly [B, 1, 96, 96, 96]
        residual = dudt_volume - diffusion_term - proliferation_term
        return residual

        # 3. Fisher-Kolmogorov PDE: du/dt - D*nabla^2(u) - rho*u*(1-u)
        # All operations are now element-wise across the full 884,736 voxels.
        diffusion_term = d_map * laplacian_u
        proliferation_term = rho_map * u_pred * (1.0 - u_pred)
        
        residual = dudt - diffusion_term - proliferation_term
        return residual

    def calculate_loss(self, u_pred_t, u_pred_t0, target_t0, t, raw_D, raw_rho):
        """
        Composite loss function for full-volume training.
        """
        # Map raw extractor outputs to biological scales
        # Diffusion (D) in mm^2/day, Proliferation (rho) in 1/day
        D_map = 0.001 + torch.sigmoid(raw_D) * (0.020 - 0.001)
        rho_map = 0.012 + torch.sigmoid(raw_rho) * (0.034 - 0.012)

        # Term 1: Initial Condition Loss (IC)
        # Measures how well the model reconstructs the T0 segmentation
        loss_ic = F.mse_loss(u_pred_t0, target_t0)
        
        # Term 2: Global Physics Loss (GPL)
        # Evaluates the PDE residual across the entire anatomical volume
        full_residual = self.compute_pde_residual(u_pred_t, t, D_map, rho_map)
        loss_physics = torch.mean(full_residual**2)
        
        # Weighting: IC is usually prioritized initially for stability
        total_loss = (10.0 * loss_ic) + (1.0 * loss_physics)
        
        return total_loss, loss_ic, loss_physics