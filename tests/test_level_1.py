import pytest
import torch
import torch.nn.functional as F
from trace_ct.data.phantom import SyntheticPhantom

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def phantom(device):
    return SyntheticPhantom(shape=(1, 64, 64), device=device)

def test_phantom_generation(phantom):
    """Level 1: Verifies the phantom generates all required structural controls."""
    data = phantom.generate()
    
    assert "clean" in data
    assert "noisy" in data
    assert "edge_mask" in data
    assert "lesion_mask" in data
    
    # Check edge is correctly positioned
    assert data["edge_mask"].sum() > 0
    
    # Check lesion is present
    assert data["lesion_mask"].sum() > 0
    
    # Check noisy has different std than clean
    assert data["noisy"].std() > data["clean"].std()
    
def test_phantom_negative_controls(phantom):
    """Level 1: Verifies negative controls exist and are properly formatted."""
    data = phantom.generate()
    nc = data["negative_controls"]
    
    assert "lf_contaminated_residual" in nc
    assert "structure_contaminated_residual" in nc
    assert "all_ones_W_fb" in nc
    
    # W_fb should be exactly all ones
    assert torch.all(nc["all_ones_W_fb"] == 1.0)
    
    # Structure contaminated residual should have overlap with edge
    overlap = (nc["structure_contaminated_residual"] * data["edge_mask"]).sum()
    assert overlap > 0
