import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DenoiserAuditOutput:
    noise_estimate: torch.Tensor
    denoise_gate: torch.Tensor
    removed_residual: torch.Tensor
    s_hat: torch.Tensor

class Denoiser(nn.Module):
    """
    Denoiser D_theta for TRACE-CT.
    Implements: D_theta(y_h^M, x_h, p_h, c_h) = p_h + k_str * R_theta(y_h^M, x_h, p_h, c_h)
    """
    def __init__(self, in_channels: int = 19, out_channels: int = 1):
        super().__init__()
        # 19 channels = 1 (y_h^M) + 1 (x_h) + 1 (p_h) + 16 (c_h context)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(
        self, 
        y_h_M: torch.Tensor, 
        x_h: torch.Tensor = None, 
        p_h: torch.Tensor = None, 
        c_h: torch.Tensor = None, 
        k_str: torch.Tensor = None,
        denoise_gate: torch.Tensor = None
    ) -> torch.Tensor:
        """
        D_theta forward pass.
        
        Args:
            y_h_M: Masked/perturbed center slice noisy patch, shape [B, 1, H, W]
            x_h: Original unperturbed center slice noisy patch, shape [B, 1, H, W]. Defaults to y_h_M if None.
            p_h: Coarse proposal from P_phi, shape [B, 1, H, W]. Defaults to zero.
            c_h: Context features, shape [B, C, H, W]. Defaults to zero of size [B, 16, H, W].
            k_str: Structure detail weight, shape [B, 1, H, W]. Defaults to one.
            denoise_gate: Denoising strength map, shape [B, 1, H, W]. Defaults to one.
        """
        B, _, H, W = y_h_M.shape
        
        if x_h is None:
            x_h = y_h_M
        
        if p_h is None:
            p_h = torch.zeros_like(y_h_M)
            
        if c_h is None:
            c_h = torch.zeros(B, 16, H, W, device=y_h_M.device)
            
        if k_str is None:
            k_str = torch.ones_like(y_h_M)
            
        return self.forward_with_audit(y_h_M, x_h=x_h, p_h=p_h, c_h=c_h, k_str=k_str, denoise_gate=denoise_gate).s_hat

    def forward_with_audit(
        self,
        y_h_M: torch.Tensor,
        x_h: torch.Tensor = None,
        p_h: torch.Tensor = None,
        c_h: torch.Tensor = None,
        k_str: torch.Tensor = None,
        denoise_gate: torch.Tensor = None,
    ) -> DenoiserAuditOutput:
        B, _, H, W = y_h_M.shape
        if x_h is None:
            x_h = y_h_M
        if p_h is None:
            p_h = torch.zeros_like(y_h_M)
        if c_h is None:
            c_h = torch.zeros(B, 16, H, W, device=y_h_M.device)
        if k_str is None:
            k_str = torch.ones_like(y_h_M)
        if denoise_gate is None:
            denoise_gate = torch.ones_like(y_h_M)

        inputs = torch.cat([y_h_M, x_h, p_h, c_h], dim=1)
        assert inputs.shape[1] == 19, f"Denoiser input channels must be 19, got {inputs.shape[1]}"
        raw = self.net(inputs)

        # Preserve the historical refinement output while exposing the auditable
        # D-as-noise-estimator quantities required by G4.5. The explicit noise
        # estimate is derived from the network's direct noise prediction raw.
        gate = torch.clamp(denoise_gate * k_str, 0.0, 1.0)
        s_hat = x_h - gate * raw
        removed_residual = x_h - s_hat
        noise_estimate = raw

        return DenoiserAuditOutput(
            noise_estimate=noise_estimate,
            denoise_gate=gate,
            removed_residual=removed_residual,
            s_hat=s_hat,
        )

