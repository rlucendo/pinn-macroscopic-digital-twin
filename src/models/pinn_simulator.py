import torch
import torch.nn as nn
import lightning as L

from src.physics.fisher_kolmogorov import FisherKolmogorovLoss


class PINNSimulator(L.LightningModule):
    """
    Spatiotemporal Physics-Informed Neural Network for Glioblastoma progression.
    
    Takes a baseline spatial latent representation (from a T0 MRI) and a time parameter 't',
    and predicts the tumor state 'u' at time 't', along with the PDE parameters.
    """

    def __init__(
        self, 
        in_channels: int = 16, 
        lambda_pde: float = 0.1, 
        learning_rate: float = 1e-4
    ):
        """
        Args:
            in_channels (int): Number of feature channels from the T0 Extractor.
            lambda_pde (float): Weight of the physics loss vs data loss.
            learning_rate (float): Optimizer learning rate.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize our custom physics loss
        self.physics_loss_fn = FisherKolmogorovLoss(voxel_spacing=1.0)
        self.data_loss_fn = nn.MSELoss()
        
        # 3D Convolutional Decoder
        # Input channels = spatial features + 1 time channel
        conv_input_channels = in_channels + 1 
        
        self.decoder = nn.Sequential(
            nn.Conv3d(conv_input_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )
        
        # Heads for the 4 required variables of the PDE
        self.head_u = nn.Conv3d(32, 1, kernel_size=1)      # Tumor concentration
        self.head_dudt = nn.Conv3d(32, 1, kernel_size=1)   # Temporal derivative
        self.head_D = nn.Conv3d(32, 1, kernel_size=1)      # Diffusion map
        self.head_rho = nn.Conv3d(32, 1, kernel_size=1)    # Proliferation map

    def forward(self, spatial_features: torch.Tensor, t: torch.Tensor):
        """
        Forward pass of the PINN.
        
        Args:
            spatial_features (torch.Tensor): Latent features from T0 [B, C, D, H, W]
            t (torch.Tensor): Time scalar per batch item [B, 1]
            
        Returns:
            Tuple containing u, du_dt, D, rho tensors.
        """
        B, C, Depth, Height, Width = spatial_features.shape
        
        # 1. Inject Time (t) into the spatial dimensions
        # Broadcast the scalar 't' to a full 3D volume [B, 1, D, H, W]
        t_volume = t.view(B, 1, 1, 1, 1).expand(B, 1, Depth, Height, Width)
        
        # Concatenate spatial features with the time volume along the channel axis
        x = torch.cat([spatial_features, t_volume], dim=1)
        
        # 2. Pass through the shared decoder
        features = self.decoder(x)
        
        # 3. Branch out to the specific physical heads with appropriate activations
        # 'u' is concentration [0, 1]
        u = torch.sigmoid(self.head_u(features))
        
        # 'du/dt' can be positive (growth) or negative (shrinkage/necrosis)
        du_dt = self.head_dudt(features) 
        
        # 'D' and 'rho' are physical coefficients and must be strictly positive
        D = nn.functional.softplus(self.head_D(features))
        rho = nn.functional.softplus(self.head_rho(features))
        
        return u, du_dt, D, rho

    def training_step(self, batch, batch_idx):
        """
        Lightning training loop step.
        """
        # Unpack the batch (assuming our DataLoader provides these)
        # spatial_features: baseline MRI embeddings
        # t: the time delta (e.g., 30 days)
        # target_u: the actual ground truth MRI mask at time 't'
        spatial_features, t, target_u = batch
        
        # Forward pass
        pred_u, pred_dudt, pred_D, pred_rho = self(spatial_features, t)
        
        # 1. Data Loss (Supervised): Does the prediction match the real MRI at time t?
        loss_data = self.data_loss_fn(pred_u, target_u)
        
        # 2. Physics Loss (Unsupervised): Does the prediction obey Fisher-Kolmogorov?
        loss_physics = self.physics_loss_fn(pred_u, pred_dudt, pred_D, pred_rho)
        
        # 3. Composite Loss
        loss_total = loss_data + (self.hparams.lambda_pde * loss_physics)
        
        # Logging to Weights & Biases / TensorBoard
        self.log("train/loss_data", loss_data, prog_bar=True)
        self.log("train/loss_physics", loss_physics, prog_bar=True)
        self.log("train/loss_total", loss_total)
        
        return loss_total

    def configure_optimizers(self):
        """
        Standard Adam optimizer setup.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer