import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextEncoder(nn.Module):
    """
    Context encoder G_psi. Extracts low-frequency structural priors from adjacent slices.
    Must NOT leak high-frequency details.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 16):
        super().__init__()
        self.lowpass_kernel_size = 7
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
    def _lowpass(self, x: torch.Tensor) -> torch.Tensor:
        padding = self.lowpass_kernel_size // 2
        return F.avg_pool2d(x, kernel_size=self.lowpass_kernel_size, stride=1, padding=padding)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self._lowpass(x)
        return self._lowpass(self.net(low))
