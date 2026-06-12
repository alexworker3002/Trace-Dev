import json

import torch
import torch.nn.functional as F

from trace_ct.audit.denoising_strength_audit import audit_denoising_strength
from trace_ct.data.phantom import SyntheticPhantom
from trace_ct.models.denoiser import Denoiser
from trace_ct.models.denoising_strength import DenoisingStrengthController
from trace_ct.training.stages import compute_gradient


def _batch():
    phantom = SyntheticPhantom(shape=(1, 64, 64), device="cpu")
    data = phantom.generate()
    noisy = data["noisy"].unsqueeze(0)
    return noisy, data["homogeneous_mask"].unsqueeze(0), data["edge_mask"].unsqueeze(0), data["lesion_mask"].unsqueeze(0)


def test_denoiser_output_includes_audit_fields():
    noisy, _, _, _ = _batch()
    out = Denoiser().forward_with_audit(noisy, x_h=noisy)
    assert out.noise_estimate.shape == noisy.shape
    assert out.denoise_gate.shape == noisy.shape
    assert out.removed_residual.shape == noisy.shape
    assert out.s_hat.shape == noisy.shape


def test_g45_fails_identity_mapping(tmp_path):
    noisy, homo, edge, lesion = _batch()
    report = audit_denoising_strength(noisy, noisy.clone(), homo, edge, lesion, run_dir=tmp_path)
    assert report["flags"]["release_D"] is False
    assert report["flags"]["identity_collapse"] is True
    assert (tmp_path / "reports" / "denoising_strength_audit.json").exists()


def test_g45_fails_edge_correlated_residual():
    noisy, homo, edge, lesion = _batch()
    output = noisy - 0.25 * edge
    report = audit_denoising_strength(noisy, output, homo, edge, lesion, write_json=False)
    assert report["flags"]["release_D"] is False
    assert report["flags"]["structure_deletion"] is True


def test_g45_passes_controlled_denoising_case():
    noisy, homo, edge, lesion = _batch()
    output = noisy * (1.0 - 0.25 * (1.0 - lesion))
    edge = torch.zeros_like(edge)
    report = audit_denoising_strength(noisy, output, homo, edge, lesion, controller=DenoisingStrengthController(), write_json=False)
    assert report["flags"]["release_D"] is True
    for key in ["r_D_hom_mean", "e_D_mean", "A_NPS_mean", "d_shape_mean", "eta_res_edge_mean", "c_lesion_mean"]:
        assert key in report["metrics"]
