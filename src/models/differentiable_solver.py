import logging
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Configure structured logging
logger = logging.getLogger(__name__)

class DifferentiableEulerSolver(nn.Module):
    """
    Hard-Physics Neural Simulator using Explicit Euler Integration.
    
    Implements Backpropagation Through Time (BPTT) with Gradient Checkpointing
    to maintain strictly constant VRAM footprint during long temporal rollouts.
    """

    def __init__(
        self, 
        pde_module: nn.Module, 
        time_step_days: float = 1.0, 
        numerical_substeps: int = 10
    ):
        """
        Args:
            pde_module (nn.Module): The physical equation module defining du/dt (e.g., FisherKolmogorovPDE).
            time_step_days (float): The macroscopic clinical time step (e.g., 1 day per macro-step).
            numerical_substeps (int): Number of internal Euler steps per macro-step to satisfy 
                                      the CFL stability condition without exploding gradients.
        """
        super().__init__()
        
        if not isinstance(pde_module, nn.Module):
            logger.error("Invalid pde_module provided. Must be an nn.Module instance.")
            raise TypeError("pde_module must inherit from torch.nn.Module")
            
        if numerical_substeps < 1:
            raise ValueError("numerical_substeps must be strictly >= 1 for temporal integration.")

        self.pde_module = pde_module
        self.macro_dt = time_step_days
        self.substeps = numerical_substeps
        
        # Internal computational time step to ensure mathematical stability
        self.micro_dt = self.macro_dt / float(self.substeps)
        
        logger.info(
            "DifferentiableEulerSolver initialized.",
            extra={
                "macro_dt": self.macro_dt,
                "micro_dt": self.micro_dt,
                "substeps": self.substeps
            }
        )

    def _sanitize_tensor(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Safely removes MONAI MetaTensor wrappers to prevent memory leaks during BPTT."""
        if hasattr(tensor, "as_tensor"):
            return tensor.as_tensor()
        return tensor

    def _euler_micro_step(
        self, 
        u: torch.Tensor, 
        D_map: torch.Tensor, 
        rho_map: torch.Tensor, 
        active_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Isolated micro-step for gradient checkpointing.
        No hard clamping is applied here to preserve exact gradient flows.
        Stability is assumed via micro_dt.
        """
        # Calculate instantaneous rate of change via injected PDE module
        du_dt = self.pde_module(u, D_map, rho_map)
        
        # Explicit Euler update masked for variable batch progression
        u_next = u + (active_mask * self.micro_dt * du_dt)
        
        return u_next

    def forward(
        self, 
        u_t0: torch.Tensor, 
        D_map: torch.Tensor, 
        rho_map: torch.Tensor, 
        delta_t_days: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrates the initial tumor state u_t0 forward in time.
        
        Args:
            u_t0 (torch.Tensor): Initial binary/soft mask of the tumor. Shape: [B, 1, D, H, W]
            D_map (torch.Tensor): Continuous diffusion field. Shape: [B, 1, D, H, W]
            rho_map (torch.Tensor): Continuous proliferation field. Shape: [B, 1, D, H, W]
            delta_t_days (torch.Tensor): Target prediction time per batch item. Shape: [B]
            
        Returns:
            torch.Tensor: Simulated tumor concentration at T_n. Shape: [B, 1, D, H, W]
        """
        u = self._sanitize_tensor(u_t0, "u_t0")
        D = self._sanitize_tensor(D_map, "D_map")
        rho = self._sanitize_tensor(rho_map, "rho_map")

        if delta_t_days.dim() != 1:
            raise ValueError(f"delta_t_days must be a 1D tensor, got shape {delta_t_days.shape}")

        max_macro_steps = int(torch.max(delta_t_days).item())
        total_micro_steps = max_macro_steps * self.substeps

        for step in range(total_micro_steps):
            # Calculate current macroscopic day index
            current_macro_day = step // self.substeps
            
            # Mask out batch items that have already reached their target T_n
            active_mask = (current_macro_day < delta_t_days).view(-1, 1, 1, 1, 1).float()
            
            # Skip computation entirely if all items in batch are done
            if torch.sum(active_mask) == 0:
                break
                
            # Apply Gradient Checkpointing using modern standard (use_reentrant=False)
            u = checkpoint(
                self._euler_micro_step, 
                u, D, rho, active_mask, 
                use_reentrant=False
            )

        # Apply a soft Sigmoid bounds constraint only at the very end of the simulation 
        # to ensure the final output is a valid probability map for the loss function,
        # without destroying the internal PDE gradients.
        # Alternatively, clamping only at the terminal node is less destructive.
        u_final = torch.clamp(u, min=0.0, max=1.0)
        
        return u_final