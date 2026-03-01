import logging
import torch
import torch.nn as nn
from typing import Tuple, Dict

# Configure structured logging
logger = logging.getLogger(__name__)

class MacroscopicDigitalTwin(nn.Module):
    """
    Couples the U-Net feature extractor with the hard-physics Euler solver.
    Implements strict biomechanical scaling to prevent Vanishing/Exploding Gradients
    during massive Backpropagation Through Time (BPTT) unrolling.
    """
    def __init__(self, extractor_module: nn.Module, physics_solver: nn.Module):
        super().__init__()
        self.extractor = extractor_module
        self.physics_solver = physics_solver
        
        # Biological constraints (Swanson's Glioblastoma boundaries)
        # These act as gradient amplifiers to ensure the tumor physically grows
        # preventing the "cold start" vanishing gradient problem.
        self.biological_D_max = 0.20    # Max diffusion (mm^2 / day)
        self.biological_rho_max = 0.10  # Max proliferation (1 / day)
        
        logger.info("MacroscopicDigitalTwin initialized with Biomechanical Scaling.",
                    extra={
                        "D_max": self.biological_D_max, 
                        "rho_max": self.biological_rho_max
                    })

    def forward(
        self, 
        image_t0: torch.Tensor, 
        seed_mask: torch.Tensor, 
        target_days: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Executes the AI deduction and physics integration pipeline.
        """
        if image_t0.shape[0] != seed_mask.shape[0] or image_t0.shape[0] != target_days.shape[0]:
            logger.error("Batch dimension mismatch in Digital Twin forward pass.")
            raise ValueError("Inconsistent batch sizes among inputs.")

        # 1. AI Deduction: Extract raw logits from the U-Net
        raw_parameters = self.extractor(image_t0)
        
        # MONAI BasicUNet might return a tuple or a tensor depending on version/mode
        if isinstance(raw_parameters, tuple):
            raw_parameters = raw_parameters[0]
                
        # 2. Biomechanical Scaling (The Anti-Vanishing Gradient fix)
        # Apply Sigmoid to bound outputs to (0, 1), then scale to physical realities.
        # Adding a tiny epsilon (1e-4) ensures D and rho are NEVER exactly zero.
        normalized_params = torch.sigmoid(raw_parameters)
        
        D_map = (normalized_params[:, 0:1, ...] * self.biological_D_max) + 1e-4
        rho_map = (normalized_params[:, 1:2, ...] * self.biological_rho_max) + 1e-4

        # Initialize the current state with the virtual origin seed
        u_current = seed_mask.clone()
        
        # 3. Hard-Physics Integration
        B = image_t0.shape[0]
        u_simulated_batch = []
        
        for b in range(B):
            # Slicing retains the batch dimension [1, C, D, H, W]
            u_b = u_current[b:b+1]
            D_b = D_map[b:b+1]
            rho_b = rho_map[b:b+1]
            
            # Slicing retains the 1D Tensor format [1] required by the physics solver
            days_b = target_days[b:b+1] 
            
            # The solver handles the substeps internally
            u_simulated = self.physics_solver(u_b, D_b, rho_b, days_b)
            
            # Strictly enforce biological density limits [0.0, 1.0] to prevent Exploding Gradients
            u_simulated = torch.clamp(u_simulated, min=0.0, max=1.0)
            u_simulated_batch.append(u_simulated)
            
        u_final = torch.cat(u_simulated_batch, dim=0)
        
        # Return the final density and the deduced parameters for loss computation and logging
        parametric_maps = {
            "diffusion_D": D_map,
            "proliferation_rho": rho_map
        }
        
        return u_final, parametric_maps