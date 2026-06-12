import torch
import torch.nn.functional as F
from typing import Tuple

def apply_block_mask(x: torch.Tensor, block_size: int = 4, mask_ratio: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Partitions the input x of shape [B, 1, H, W] into block_size x block_size blocks,
    randomly masks blocks with a ratio of mask_ratio, and returns (masked_x, mask).
    In masked areas, values are replaced with random Gaussian noise.
    
    Args:
        x: Input image tensor, shape [B, 1, H, W]
        block_size: Size of block to mask
        mask_ratio: Fraction of pixels to mask (0 to 1)
        
    Returns:
        masked_x: Input x with masked blocks replaced by noise, shape [B, 1, H, W]
        mask: Binary mask where 1 represents masked blocks, shape [B, 1, H, W]
    """
    B, C, H, W = x.shape
    device = x.device
    
    num_blocks_h = H // block_size
    num_blocks_w = W // block_size
    
    if num_blocks_h == 0 or num_blocks_w == 0:
        # Fallback to pixel-wise masking if image is smaller than block_size
        rand = torch.rand(B, 1, H, W, device=device)
        mask = (rand < mask_ratio).float()
    else:
        # Low resolution random values for blocks
        rand = torch.rand(B, 1, num_blocks_h, num_blocks_w, device=device)
        flat_rand = rand.view(B, -1)
        k = max(1, int(mask_ratio * flat_rand.shape[1]))
        
        # Get threshold value for the top k values
        val, _ = torch.topk(flat_rand, k, dim=1)
        thresholds = val[:, -1].view(B, 1, 1, 1)
        block_mask = (rand >= thresholds).float()
        
        # Upsample to full resolution
        mask = F.interpolate(block_mask, size=(H, W), mode='nearest')
        
    # Generate random Gaussian noise with same standard deviation as input or 0.5
    noise_std = float(x.std().item()) if x.std() > 1e-4 else 0.5
    noise = torch.randn_like(x) * noise_std
    
    masked_x = x * (1.0 - mask) + noise * mask
    
    return masked_x, mask
