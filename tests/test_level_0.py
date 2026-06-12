import pytest
import torch
from trace_ct.models.denoiser import Denoiser
from trace_ct.models.context import ContextEncoder
from trace_ct.utils.hashing import compute_architecture_hash

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def test_denoiser_initialization_and_forward(device):
    """Level 0: Validates tensor shape, forward/backward execution, and module initialization ONLY."""
    # Instantiate with strict channel contract (19 channels)
    model = Denoiser(in_channels=19, out_channels=1).to(device)
    
    # Calculate architecture hash and assert it is not empty
    arch_hash = compute_architecture_hash(model)
    assert len(arch_hash) == 64
    
    y_h_M = torch.randn(2, 1, 64, 64, device=device)
    x_h = torch.randn(2, 1, 64, 64, device=device)
    p_h = torch.randn(2, 1, 64, 64, device=device)
    c_h = torch.randn(2, 16, 64, 64, device=device)
    
    # Forward
    out = model(y_h_M, x_h, p_h, c_h)
    assert out.shape == y_h_M.shape
    
    # Backward
    loss = out.sum()
    loss.backward()
    
    # Check grads
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad

def test_context_encoder_initialization_and_forward(device):
    """Level 0: Context encoder smoke test."""
    model = ContextEncoder(in_channels=1, out_channels=16).to(device)
    x = torch.randn(2, 1, 64, 64, device=device)
    
    out = model(x)
    assert out.shape == (2, 16, 64, 64)
    
    loss = out.sum()
    loss.backward()
    
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad
