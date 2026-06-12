import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

def compute_pearson_correlation(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> float:
    """Computes the Pearson correlation coefficient between two tensors, optionally restricted to a mask."""
    # Ensure channel dimensions match by averaging if needed
    if x.shape[1] > 1 and y.shape[1] == 1:
        x = torch.mean(x, dim=1, keepdim=True)
    elif y.shape[1] > 1 and x.shape[1] == 1:
        y = torch.mean(y, dim=1, keepdim=True)
        
    x_flat = x.detach().cpu().numpy().flatten()
    y_flat = y.detach().cpu().numpy().flatten()
    
    if mask is not None:
        mask_flat = mask.detach().cpu().numpy().flatten()
        indices = np.where(mask_flat > 0.5)[0]
        if len(indices) < 2:
            return 0.0
        x_flat = x_flat[indices]
        y_flat = y_flat[indices]
        
    x_std = np.std(x_flat)
    y_std = np.std(y_flat)
    
    if x_std < 1e-8 or y_std < 1e-8:
        return 0.0
        
    return float(np.corrcoef(x_flat, y_flat)[0, 1])

def audit_g1_masked_baseline(
    pred: torch.Tensor, 
    noisy: torch.Tensor, 
    mask: torch.Tensor, 
    homo_mask: torch.Tensor = None
) -> Dict[str, float]:
    """
    Computes G1 validation metrics:
    - copy_attack_correlation: Correlation inside the masked region between prediction and noise-perturbed input.
    - masked_only_loss_ratio: Ratio of loss on masked vs unmasked regions.
    - homogeneous_noise_reduction: Percentage reduction in standard deviation on homogeneous ROI.
    """
    # 1. Copy-attack correlation inside masked region
    copy_corr = compute_pearson_correlation(pred, noisy, mask)
    
    # 2. Masked-only loss ratio
    diff_sq = (pred - noisy) ** 2
    loss_masked = float((diff_sq * mask).sum().item() / (mask.sum().item() + 1e-8))
    
    unmask = 1.0 - mask
    loss_unmasked = float((diff_sq * unmask).sum().item() / (unmask.sum().item() + 1e-8))
    
    loss_ratio = loss_masked / (loss_unmasked + 1e-8)
    
    # 3. Noise reduction on homogeneous region
    noise_reduction = 0.0
    if homo_mask is not None and homo_mask.sum() > 0:
        # Check standard deviation reduction on the homogeneous ROI
        # For simplicity, extract flat regions where homo_mask is 1
        pred_flat = pred.detach().cpu().numpy().flatten()
        noisy_flat = noisy.detach().cpu().numpy().flatten()
        h_flat = homo_mask.detach().cpu().numpy().flatten()
        
        indices = np.where(h_flat > 0.5)[0]
        if len(indices) > 1:
            std_noisy = np.std(noisy_flat[indices])
            std_pred = np.std(pred_flat[indices])
            if std_noisy > 1e-6:
                noise_reduction = float((std_noisy - std_pred) / std_noisy)
                
    return {
        "copy_attack_correlation": copy_corr,
        "masked_only_loss_ratio": loss_ratio,
        "homogeneous_noise_reduction": noise_reduction
    }

def audit_g2_context(
    context_features: torch.Tensor, 
    adjacent_slice: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluates G2 context leakage metrics:
    - High-frequency leakage ratio in context features.
    - High-frequency correlation with adjacent slices.
    """
    # High-pass filter: HP(x) = x - LP(x)
    lp_ctx = F.avg_pool2d(context_features, kernel_size=3, stride=1, padding=1)
    hp_ctx = context_features - lp_ctx
    
    lp_adj = F.avg_pool2d(adjacent_slice, kernel_size=3, stride=1, padding=1)
    hp_adj = adjacent_slice - lp_adj
    
    # HF Leakage Ratio: HF variance / Total variance
    ctx_var = float(context_features.var().item() + 1e-8)
    hp_var = float(hp_ctx.var().item())
    leakage_ratio = hp_var / ctx_var
    
    # Correlation of high-frequency components
    hf_correlation = compute_pearson_correlation(hp_ctx, hp_adj)
    
    return {
        "high_frequency_leakage_ratio": leakage_ratio,
        "high_frequency_correlation": hf_correlation
    }
