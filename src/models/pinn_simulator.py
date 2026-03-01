import logging
import torch
import torch.nn as nn

# Configure structured logging
logger = logging.getLogger(__name__)

class MacroscopicDigitalTwin(nn.Module):
    """
    End-to-End Hard-Physics Digital Twin for Glioblastoma Growth.
    
    Orchestrates the anatomical state extraction, enforces biological constraints,
    and integrates the physical state forward in time using a differentiable PDE solver.
    """

    def __init__(
        self, 
        extractor_module: nn.Module, 
        physics_solver: nn.Module
    ):
        """
        Initializes the Digital Twin via Dependency Injection.
        
        Args:
            extractor_module (nn.Module): The U-Net to extract D and rho maps (BaselineStateExtractor).
            physics_solver (nn.Module): The explicit time integrator (DifferentiableEulerSolver).
        """
        super().__init__()
        
        if not isinstance(extractor_module, nn.Module) or not isinstance(physics_solver, nn.Module):
            logger.error("Invalid modules provided for MacroscopicDigitalTwin initialization.")
            raise TypeError("Both extractor_module and physics_solver must be nn.Module instances.")
            
        self.extractor = extractor_module
        self.solver = physics_solver
        
        # Swanson's Biological Constraints (mm^2/day for D, 1/day for rho)
        self.D_MIN = 0.001
        self.D_MAX = 0.020
        self.RHO_MIN = 0.012
        self.RHO_MAX = 0.034
        
        logger.info("MacroscopicDigitalTwin initialized with Hard-Physics architecture.")

    def _apply_biological_constraints(self, raw_D: torch.Tensor, raw_rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Maps the bounded neural network outputs into strict, literature-backed 
        physical units for glioblastoma dynamics.
        """
        # Assuming raw_D and raw_rho already passed through Softplus/Sigmoid in the Extractor
        # We use a normalized scaling approach assuming the input is roughly [0, 1] bounded.
        # If raw_D is from Softplus, we can apply a clamping or a learned scaling.
        # For strict bounds, we apply a Sigmoid-like transformation here to guarantee limits.
        
        # Transform unbounded/softplus D into strictly bounded physical D
        scaled_D = self.D_MIN + torch.sigmoid(raw_D) * (self.D_MAX - self.D_MIN)
        
        # Transform raw_rho (assuming Sigmoid from extractor) into physical rho
        scaled_rho = self.RHO_MIN + raw_rho * (self.RHO_MAX - self.RHO_MIN)
        
        return scaled_D, scaled_rho

    def forward(
        self, 
        mri_t0: torch.Tensor, 
        tumor_mask_t0: torch.Tensor, 
        target_time_days: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Executes the complete macroscopic simulation pipeline.
        
        Args:
            mri_t0 (torch.Tensor): Structural MRI at Day 0. Shape: [B, C, D, H, W]
            tumor_mask_t0 (torch.Tensor): Binary/Soft tumor segmentation at Day 0.
            target_time_days (torch.Tensor): Temporal prediction horizon T_n per batch item.
            
        Returns:
            tuple:
                - u_pred_tn (torch.Tensor): Predicted tumor density at T_n.
                - parametric_maps (dict): Dictionary containing the physical maps for logging/Slicer3D.
        """
        # 1. Extract raw physical parameters from static anatomy
        raw_D, raw_rho = self.extractor(mri_t0)
        
        # 2. Map to biological units (Swanson's Constraints)
        D_map, rho_map = self._apply_biological_constraints(raw_D, raw_rho)
        
        # 3. Integrate forward in time using the Differentiable PDE Solver
        u_pred_tn = self.solver(
            u_t0=tumor_mask_t0, 
            D_map=D_map, 
            rho_map=rho_map, 
            delta_t_days=target_time_days
        )
        
        parametric_maps = {
            "diffusion_D": D_map,
            "proliferation_rho": rho_map
        }
        
        return u_pred_tn, parametric_maps