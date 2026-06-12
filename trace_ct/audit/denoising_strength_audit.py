import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from trace_ct.audit.proposal_audit import compute_gradient
from trace_ct.models.denoising_strength import DenoisingStrengthController


def _masked_values(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = x[mask > 0.5]
    if values.numel() == 0:
        return x.flatten()
    return values


def _safe_std(x: torch.Tensor, mask: torch.Tensor) -> float:
    values = _masked_values(x, mask)
    if values.numel() <= 1:
        return 0.0
    return float(values.std(unbiased=False).item())


def _nps(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        x = x[:, 0]
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    spec = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    return (spec.real ** 2 + spec.imag ** 2).mean(dim=0)


def _noise_band_mask(h: int, w: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=device), torch.linspace(-1, 1, w, device=device), indexing="ij")
    rr = torch.sqrt(xx ** 2 + yy ** 2)
    return (rr >= 0.25) & (rr <= 0.75)


def _nps_metrics(noisy: torch.Tensor, output: torch.Tensor) -> tuple[float, float]:
    nps_noisy = _nps(noisy)
    nps_output = _nps(output)
    band = _noise_band_mask(nps_noisy.shape[-2], nps_noisy.shape[-1], nps_noisy.device)
    amp_noisy = nps_noisy[band].sum() + 1e-8
    amp_output = nps_output[band].sum()
    amp_ratio = float((amp_output / amp_noisy).item())
    n_shape = nps_noisy[band] / amp_noisy
    o_shape = nps_output[band] / (amp_output + 1e-8)
    shape_distance = float(torch.mean(torch.abs(n_shape - o_shape)).item())
    return amp_ratio, shape_distance


def _corr_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    if a.numel() < 2 or float(a.std().item()) < 1e-8 or float(b.std().item()) < 1e-8:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    return float(torch.abs(torch.sum(a * b) / (torch.sqrt(torch.sum(a * a) * torch.sum(b * b)) + 1e-8)).item())


def audit_denoising_strength(
    noisy: torch.Tensor,
    output: torch.Tensor,
    homogeneous_mask: torch.Tensor,
    edge_mask: torch.Tensor | None = None,
    lesion_mask: torch.Tensor | None = None,
    controller: DenoisingStrengthController | None = None,
    checkpoint: str | None = None,
    dataset_split: str = "validation_or_audit",
    run_dir: str | Path | None = None,
    write_json: bool = True,
) -> Dict[str, Any]:
    controller = controller or DenoisingStrengthController()
    noisy = noisy.detach().float()
    output = output.detach().float()
    homogeneous_mask = homogeneous_mask.detach().float()
    edge_mask = edge_mask.detach().float() if edge_mask is not None else (compute_gradient(noisy) > torch.quantile(compute_gradient(noisy), 0.90)).float()
    lesion_mask = lesion_mask.detach().float() if lesion_mask is not None else torch.zeros_like(noisy)

    removed = noisy - output
    r_D = _safe_std(output, homogeneous_mask) / (_safe_std(noisy, homogeneous_mask) + 1e-8)
    e_D = float((torch.norm(_masked_values(removed, homogeneous_mask)) / (torch.norm(_masked_values(noisy, homogeneous_mask)) + 1e-8)).item())
    eta = _corr_abs(compute_gradient(removed) * edge_mask, compute_gradient(noisy) * edge_mask)
    amp, shape = _nps_metrics(noisy, output)

    if float(lesion_mask.sum().item()) > 1:
        c_lesion = _safe_std(output, lesion_mask) / (_safe_std(noisy, lesion_mask) + 1e-8)
    else:
        c_lesion = 1.0

    metrics = {
        "r_D_hom_mean": float(r_D),
        "r_D_hom_p05": float(r_D),
        "r_D_hom_p50": float(r_D),
        "r_D_hom_p95": float(r_D),
        "e_D_mean": float(e_D),
        "A_NPS_mean": float(amp),
        "d_shape_mean": float(shape),
        "eta_res_edge_mean": float(eta),
        "c_lesion_mean": float(c_lesion),
    }
    decision = controller.evaluate(metrics)
    report = {
        "stage": "G4.5",
        "checkpoint": checkpoint or "uncheckpointed",
        "dataset_split": dataset_split,
        "num_volumes": 1,
        "num_patches": int(noisy.shape[0]),
        "metrics": metrics,
        "flags": decision["flags"],
        "thresholds": controller.thresholds_dict(),
        "recommended_action": decision["recommended_action"],
        "failure_reasons": decision["reasons"],
    }

    if write_json and run_dir is not None:
        reports_dir = Path(run_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "denoising_strength_audit.json", "w") as f:
            json.dump(report, f, indent=2)
    return report
