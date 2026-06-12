import torch
from typing import Dict, Tuple

class SyntheticPhantom:
    """
    Generates structured synthetic phantom tensors for Level 1 tests.
    Includes edges, lesions (high contrast), homogeneous regions, and low/high frequency noise.
    """
    def __init__(self, shape: Tuple[int, int, int] = (1, 64, 64), device: str = "cpu"):
        self.shape = shape
        self.device = device
        
    def generate(self) -> Dict[str, torch.Tensor]:
        """Generates the phantom data and masks."""
        B, H, W = self.shape
        
        # Base background
        clean = torch.zeros(self.shape, device=self.device)
        
        # 1. Create edge (step function at W/2)
        clean[:, :, W//2:] = 1.0
        edge_mask = torch.zeros(self.shape, device=self.device)
        edge_mask[:, :, W//2 - 2 : W//2 + 2] = 1.0
        
        # 2. Create lesion (small high-contrast circle at H/4, W/4)
        lesion_mask = torch.zeros(self.shape, device=self.device)
        cy, cx = H//4, W//4
        radius = min(H, W) // 8
        y, x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
        dist = (y - cy)**2 + (x - cx)**2
        lesion_mask[:, dist <= radius**2] = 1.0
        clean[lesion_mask == 1.0] = 2.0
        
        # 3. Homogeneous region
        homo_mask = torch.zeros(self.shape, device=self.device)
        homo_mask[:, 3*H//4 - radius : 3*H//4 + radius, 3*W//4 - radius : 3*W//4 + radius] = 1.0
        
        # Add high frequency noise to create noisy input
        hf_noise = torch.randn(self.shape, device=self.device) * 0.5
        noisy = clean + hf_noise
        
        # Create an adjacent slice (similar clean structure, different noise)
        adjacent_noisy = clean + torch.randn(self.shape, device=self.device) * 0.5
        
        # Generate low-frequency contaminated residual
        # Simulate LF by heavily blurring noise (just adding a smooth gradient for simplicity)
        lf_residual = torch.linspace(-1, 1, W, device=self.device).view(1, 1, W).expand(self.shape)
        
        # Generate structure-contaminated residual (residual contains part of the edge)
        structure_residual = edge_mask.clone()
        
        return {
            "clean": clean,
            "noisy": noisy,
            "adjacent_noisy": adjacent_noisy,
            "edge_mask": edge_mask,
            "lesion_mask": lesion_mask,
            "homogeneous_mask": homo_mask,
            "negative_controls": {
                "lf_contaminated_residual": lf_residual,
                "structure_contaminated_residual": structure_residual,
                "all_ones_W_fb": torch.ones_like(clean)
            }
        }
