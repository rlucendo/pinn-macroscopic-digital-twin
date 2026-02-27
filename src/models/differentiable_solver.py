import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging

logger = logging.getLogger("DifferentiableEulerSolver")

class DifferentiablePDESolver(nn.Module):
    """
    Hard-Physics Neural Simulator with BPTT Memory Optimization.
    Uses Gradient Checkpointing to prevent VRAM explosion over long time horizons.
    """
    def __init__(self, dt: float = 1.0):
        super().__init__()
        self.dt = dt
        self._register_laplacian_kernel()
        logger.info(f"Differentiable Euler Solver initialized with dt={dt} days.")

    def _register_laplacian_kernel(self):
        kernel = torch.zeros((1, 1, 3, 3, 3))
        kernel[0, 0, 1, 1, 1] = -6.0
        kernel[0, 0, 0, 1, 1] = 1.0; kernel[0, 0, 2, 1, 1] = 1.0
        kernel[0, 0, 1, 0, 1] = 1.0; kernel[0, 0, 1, 2, 1] = 1.0
        kernel[0, 0, 1, 1, 0] = 1.0; kernel[0, 0, 1, 1, 2] = 1.0
        self.register_buffer("laplacian", kernel)

    def _euler_step(self, u, D_map, rho_map, active_mask):
        """Isolated single time-step for gradient checkpointing."""
        laplacian_u = F.conv3d(u, self.laplacian, padding=1)
        diffusion = D_map * laplacian_u
        proliferation = rho_map * u * (1.0 - u)
        
        du = diffusion + proliferation
        u_next = u + (active_mask * self.dt * du)
        
        return torch.clamp(u_next, min=0.0, max=1.0)

    def forward(self, u_t0: torch.Tensor, D_map: torch.Tensor, rho_map: torch.Tensor, delta_t: torch.Tensor):
        # 1. Strip MONAI MetaTensors to prevent metadata memory leaks in the loop
        u = u_t0.as_tensor() if hasattr(u_t0, 'as_tensor') else u_t0
        D = D_map.as_tensor() if hasattr(D_map, 'as_tensor') else D_map
        rho = rho_map.as_tensor() if hasattr(rho_map, 'as_tensor') else rho_map

        max_steps = int(torch.max(delta_t).item())

        # 2. Forward Euler with Checkpointing
        for step in range(max_steps):
            active_mask = (step < delta_t).view(-1, 1, 1, 1, 1).float()
            
            # Checkpoint the step: drastically reduces VRAM usage by not saving intermediate graphs
            # use_reentrant=False is the modern PyTorch standard for checkpointing
            u = checkpoint(self._euler_step, u, D, rho, active_mask, use_reentrant=False)

        return u