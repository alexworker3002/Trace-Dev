import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ProposalGenerator(nn.Module):
    """
    Proposal Generator G_phi.
    Estimates coarse clean proposal, adjacency/safety/homogeneous masks,
    trustworthy structure detail weight, noise scale modulation, residual injection scaling,
    and context reliability.
    
    P_phi(x_h, u_minus, u_plus, c_h) -> (p_h, w_adj, w_safety, w_hom, k_str, sigma_h, A_h, g_ctx)
    """
    def __init__(self, in_channels: int = 19):
        super().__init__()
        # 19 channels = 1 (x_h) + 1 (u_minus) + 1 (u_plus) + 16 (c_h context)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 8, kernel_size=3, padding=1)
        )

    def forward(self, x_h: torch.Tensor, u_minus: torch.Tensor, u_plus: torch.Tensor, c_h: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Runs the proposal generator network.
        
        Args:
            x_h: Center slice noisy patch, shape [B, 1, H, W]
            u_minus: Restricted low/mid frequency adjacent slice (-1), shape [B, 1, H, W]
            u_plus: Restricted low/mid frequency adjacent slice (+1), shape [B, 1, H, W]
            c_h: Context encoder features, shape [B, 16, H, W]
            
        Returns:
            p_h: Coarse clean proposal, shape [B, 1, H, W]
            w_adj: Adjacency weight, shape [B, 1, H, W], range [0, 1]
            w_safety: Safety weight, shape [B, 1, H, W], range [0, 1]
            w_hom: Homogeneous mask, shape [B, 1, H, W], range [0, 1]
            k_str: Structure detail weight, shape [B, 1, H, W], range [0, 1]
            sigma_h: Noise scale estimate, shape [B, 1, H, W], range > 0
            A_h: Residual injection scaling, shape [B, 1, H, W], range [0, 1]
            g_ctx: Context reliability gate, shape [B, 1, H, W], range [0, 1]
        """
        inputs = torch.cat([x_h, u_minus, u_plus, c_h], dim=1)
        assert inputs.shape[1] == 19, f"ProposalGenerator input channels must be 19, got {inputs.shape[1]}"
        out = self.net(inputs)
        
        smooth = F.avg_pool2d(x_h, kernel_size=11, stride=1, padding=5)
        dx = F.pad(torch.abs(x_h[:, :, :, 1:] - x_h[:, :, :, :-1]), (0, 1, 0, 0))
        dy = F.pad(torch.abs(x_h[:, :, 1:, :] - x_h[:, :, :-1, :]), (0, 0, 0, 1))
        edge_gate = torch.sigmoid(40.0 * (dx + dy - 0.45))
        coarse_prior = smooth + edge_gate * (x_h - smooth)
        p_h = coarse_prior + 0.02 * torch.tanh(out[:, 0:1, :, :])
        w_adj = torch.sigmoid(out[:, 1:2, :, :])
        w_safety = torch.sigmoid(out[:, 2:3, :, :])
        w_hom = torch.sigmoid(out[:, 3:4, :, :])
        k_str = torch.sigmoid(out[:, 4:5, :, :])
        sigma_h = F.softplus(out[:, 5:6, :, :]) + 1e-6
        A_h = torch.sigmoid(out[:, 6:7, :, :])
        g_ctx = torch.sigmoid(out[:, 7:8, :, :])
        
        return p_h, w_adj, w_safety, w_hom, k_str, sigma_h, A_h, g_ctx
