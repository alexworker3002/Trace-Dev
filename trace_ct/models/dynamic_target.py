import torch

class DynamicTargetAggregator:
    """
    Dynamic Target Gating (W_fb).
    Calculates the regional proximal target for the denoiser training loop.
    Implements: t_h = [ x_h + W_fb * (p_h - x_h) + m_t * momentum ].detach()
    Where: W_fb = w_safety * w_adj * w_benefit
    """
    def __init__(self):
        pass
        
    def aggregate(
        self, 
        x_h: torch.Tensor, 
        p_h: torch.Tensor, 
        w_safety: torch.Tensor, 
        w_adj: torch.Tensor, 
        w_benefit: torch.Tensor,
        momentum: torch.Tensor = None,
        m_t: float = 0.0
    ) -> torch.Tensor:
        """
        Aggregates inputs to generate the dynamic training target for D_theta.
        
        Args:
            x_h: Clean-ish REG slice baseline, shape [B, C, H, W]
            p_h: Coarse clean proposal from P_phi, shape [B, C, H, W]
            w_safety: Structural safety weight, shape [B, 1, H, W]
            w_adj: Adjacency weight, shape [B, 1, H, W]
            w_benefit: Benefit/warmup weight, shape [B, 1, H, W]
            momentum: Optional momentum tensor, shape [B, C, H, W]
            m_t: Momentum scaling factor
            
        Returns:
            t_h: Denoised target detached from computational graph, shape [B, C, H, W]
        """
        # Calculate localized feedback gate W_fb
        w_fb = w_safety * w_adj * w_benefit
        
        # Proximal target formulation
        t_h = x_h + w_fb * (p_h - x_h)
        
        if momentum is not None and m_t > 0.0:
            t_h = t_h + m_t * momentum
            
        # Enforce stop-gradient
        return t_h.detach()
        
    def compute_drift_q95(self, t_h: torch.Tensor, x_h: torch.Tensor) -> float:
        """
        Computes the 95th percentile (Q95) of the absolute drift between target t_h and x_h.
        """
        abs_diff = torch.abs(t_h - x_h)
        if abs_diff.numel() == 0:
            return 0.0
        return float(torch.quantile(abs_diff, 0.95).item())
