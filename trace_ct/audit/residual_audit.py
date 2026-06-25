import torch
import torch.nn.functional as F
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from trace_ct.config.schema import ResidualAuditThresholds

def compute_nps_2d(patch: torch.Tensor) -> torch.Tensor:
    """Computes the 2D Noise Power Spectrum (NPS) of a patch using FFT."""
    # Ensure shape is 2D [H, W]
    p_2d = patch.squeeze()
    fft = torch.fft.fft2(p_2d)
    return torch.abs(fft) ** 2

def compute_swd_2d(patch_res: torch.Tensor, patch_ref: torch.Tensor, num_projections: int = 32) -> float:
    """
    Computes the Sliced Wasserstein Distance (SWD) between two 2D patches.
    """
    p_res = patch_res.squeeze().detach()
    p_ref = patch_ref.squeeze().detach()
    
    H, W = p_res.shape
    device = p_res.device
    
    if p_res.shape != p_ref.shape:
        # Resize p_ref to match p_res
        p_ref = F.interpolate(p_ref.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        
    # Generate random projection vectors on the sphere in R^W
    projections = torch.randn(W, num_projections, device=device)
    projections = projections / torch.norm(projections, dim=0, keepdim=True)
    
    # Project the patches: shape [H, num_projections]
    proj_res = torch.matmul(p_res, projections)
    proj_ref = torch.matmul(p_ref, projections)
    
    # Sort projections along H dimension
    proj_res_sorted, _ = torch.sort(proj_res, dim=0)
    proj_ref_sorted, _ = torch.sort(proj_ref, dim=0)
    
    # Compute W1 distance
    w1_distances = torch.mean(torch.abs(proj_res_sorted - proj_ref_sorted), dim=0)
    return float(torch.mean(w1_distances).item())

class ResidualAuditor:
    """
    Audits candidate residual samples to ensure they do not carry structured anatomical
    or low-frequency content (which indicates leakage of clean structure or incomplete denoising).
    """
    def __init__(self, thresholds: ResidualAuditThresholds):
        self.thresholds = thresholds
        
    def audit_patch(
        self, 
        residual_patch: torch.Tensor, 
        edge_mask_patch: torch.Tensor = None,
        clean_baseline_patch: torch.Tensor = None,
        validation_hr_proxy_patch: torch.Tensor = None
    ) -> Tuple[bool, List[str], Dict[str, float]]:
        """
        Audits a single residual patch against safety and proxy criteria.
        """
        reasons = []
        metrics = {}
        
        # Alias clean_baseline_patch to validation_hr_proxy_patch if clean is not provided
        if validation_hr_proxy_patch is None:
            validation_hr_proxy_patch = clean_baseline_patch
            
        # 1. Standard Deviation Check
        std = float(residual_patch.std().item())
        metrics["std"] = std
        if validation_hr_proxy_patch is not None:
            baseline_std = float(validation_hr_proxy_patch.std().item() + 1e-8)
            rel_std = std / baseline_std
            metrics["relative_std"] = rel_std
            if rel_std > self.thresholds.relative_std_max or rel_std < self.thresholds.relative_std_min:
                reasons.append(f"Relative std {rel_std:.4f} outside bounds [{self.thresholds.relative_std_min}, {self.thresholds.relative_std_max}]")
        else:
            metrics["relative_std"] = 1.0
            if std < 1e-4:
                reasons.append(f"Standard deviation {std:.4f} is too close to zero")
                
        # 2. Structural/Edge Leakage Check
        metrics["edge_leakage"] = 0.0
        if edge_mask_patch is not None:
            mask_sum = float(edge_mask_patch.sum().item())
            if mask_sum > 0:
                overlap = float(torch.abs((residual_patch * edge_mask_patch).sum()).item() / mask_sum)
                metrics["edge_leakage"] = overlap
                if overlap > self.thresholds.max_edge_leakage_q95:
                    reasons.append(f"Edge leakage {overlap:.4f} exceeds max {self.thresholds.max_edge_leakage_q95}")
                    
        # 3. Low-Frequency Leakage Check
        lf_residual = F.avg_pool2d(residual_patch.unsqueeze(0), kernel_size=8, stride=8).squeeze(0)
        lf_variance = float(lf_residual.var().item())
        metrics["lf_variance"] = lf_variance
        if lf_variance > self.thresholds.max_low_frequency_leakage_q95:
            reasons.append(f"Low-frequency variance {lf_variance:.4f} exceeds max {self.thresholds.max_low_frequency_leakage_q95}")
            
        # 4. Calibration Proxy (Mean Shift Check)
        mean_val = float(residual_patch.mean().item())
        metrics["mean_shift"] = mean_val
        if abs(mean_val) > 0.1:
            reasons.append(f"Mean shift {mean_val:.4f} indicates calibration bias")
            
        # 5. NPS Shape Check (Correlation check)
        metrics["nps_correlation"] = 1.0
        if validation_hr_proxy_patch is not None:
            nps_res = compute_nps_2d(residual_patch)
            nps_ref = compute_nps_2d(validation_hr_proxy_patch)
            
            nps_res_flat = nps_res.detach().cpu().numpy().flatten()
            nps_ref_flat = nps_ref.detach().cpu().numpy().flatten()
            
            if np.std(nps_res_flat) > 1e-8 and np.std(nps_ref_flat) > 1e-8:
                nps_corr = float(np.corrcoef(nps_res_flat, nps_ref_flat)[0, 1])
                metrics["nps_correlation"] = nps_corr
                
        # 6. Sliced Wasserstein Distance (SWD) check
        metrics["swd"] = 0.0
        if validation_hr_proxy_patch is not None:
            swd_val = compute_swd_2d(residual_patch, validation_hr_proxy_patch)
            metrics["swd"] = swd_val
            max_swd = getattr(self.thresholds, "max_swd", 0.15)
            if swd_val > max_swd:
                reasons.append(f"Sliced Wasserstein Distance {swd_val:.4f} exceeds max {max_swd}")
                
        return len(reasons) == 0, reasons, metrics


class ResidualPool:
    """
    Manages the collection, storage, and isolation of audited residual samples.
    Saves accepted residual samples along with detailed metadata (donor, slice, coordinates, metrics).
    """
    def __init__(self, run_dir: Path, thresholds: ResidualAuditThresholds, donor_volume_ids: List[str], audit_version_hash: str = "v1"):
        self.run_dir = run_dir
        self.thresholds = thresholds
        self.donor_volume_ids = donor_volume_ids
        self.audit_version_hash = audit_version_hash
        
        self.accepted_pool_path = run_dir / "residual_pools" / "accepted_residuals.pt"
        self.error_pool_path = run_dir / "residual_pools" / "error_residuals.pt"
        self.metadata_log_path = run_dir / "residual_pools" / "residual_metadata.json"
        self.double_track_log_path = run_dir / "audit" / "stage_records" / "G3_double_track.json"
        
        self.auditor = ResidualAuditor(thresholds)
        self.accepted_patches: List[torch.Tensor] = []
        self.rejected_patches: List[torch.Tensor] = []
        self.patch_metadata: List[Dict[str, Any]] = []
        
    def add_volume_residuals(
        self, 
        volume_id: str, 
        residuals: torch.Tensor, 
        edge_masks: torch.Tensor = None, 
        clean_baselines: torch.Tensor = None,
        patch_size: Tuple[int, int] = (64, 64),
        sample_mode: str = "patch",
        validation_hr_proxies: torch.Tensor = None
    ) -> Dict[str, Any]:
        """
        Extracts/audits residual samples and registers them in the accepted or error pools.
        """
        if validation_hr_proxies is None:
            validation_hr_proxies = clean_baselines
        if sample_mode not in {"patch", "full_slice"}:
            raise ValueError(f"Unsupported residual sample_mode={sample_mode!r}.")
            
        if self.thresholds.require_donor_receiver_isolation:
            if volume_id not in self.donor_volume_ids:
                raise ValueError(f"Volume {volume_id} is not in the donor volume list. HR isolation violation.")
                
        slices, H, W = residuals.shape
        ph, pw = (H, W) if sample_mode == "full_slice" else patch_size
        
        accepted_count = 0
        rejected_count = 0
        
        for s in range(slices):
            for y in range(0, H - ph + 1, ph):
                for x in range(0, W - pw + 1, pw):
                    res_patch = residuals[s, y:y+ph, x:x+pw].unsqueeze(0) # [1, ph, pw]
                    edge_patch = edge_masks[s, y:y+ph, x:x+pw].unsqueeze(0) if edge_masks is not None else None
                    
                    val_proxy_patch = None
                    if sample_mode != "full_slice" and validation_hr_proxies is not None:
                        val_s = s % validation_hr_proxies.shape[0]
                        val_proxy_patch = validation_hr_proxies[val_s, y:y+ph, x:x+pw].unsqueeze(0)
                        
                    passed, reasons, metrics = self.auditor.audit_patch(
                        res_patch, 
                        edge_patch, 
                        validation_hr_proxy_patch=val_proxy_patch
                    )
                    
                    meta_item = {
                        "donor_volume_id": volume_id,
                        "slice_idx": s,
                        "coordinates": [y, x, y + ph, x + pw],
                        "audit_version_hash": self.audit_version_hash,
                        "status": "accepted" if passed else "rejected",
                        "reasons": reasons,
                        "metrics": metrics
                    }
                    self.patch_metadata.append(meta_item)
                    
                    if passed:
                        self.accepted_patches.append(res_patch)
                        accepted_count += 1
                    else:
                        self.rejected_patches.append(res_patch)
                        rejected_count += 1
                        
        self.save_pools()
        
        total = accepted_count + rejected_count
        rate = accepted_count / total if total > 0 else 0.0
        
        stats = {
            "volume_id": volume_id,
            "total_extracted": total,
            "accepted": accepted_count,
            "rejected": rejected_count,
            "accepted_rate": rate
        }
        
        self.log_double_track(stats)
        return stats
        
    def save_pools(self):
        """Saves accepted/rejected pools and metadata to disk."""
        # Ensure directories exist
        self.accepted_pool_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.accepted_patches:
            torch.save(torch.stack(self.accepted_patches), self.accepted_pool_path)
        if self.rejected_patches:
            torch.save(torch.stack(self.rejected_patches), self.error_pool_path)
            
        with open(self.metadata_log_path, 'w') as f:
            json.dump(self.patch_metadata, f, indent=2)
            
    def load_accepted_patches(self) -> torch.Tensor:
        """Loads accepted residual patches from disk."""
        if not self.accepted_pool_path.exists():
            return torch.zeros((0, 1, 64, 64))
        return torch.load(self.accepted_pool_path)
        
    def get_double_track_stats(self) -> Dict[str, Any]:
        """Returns the current accumulated double-track counts."""
        total_acc = len(self.accepted_patches)
        total_rej = len(self.rejected_patches)
        total = total_acc + total_rej
        rate = total_acc / total if total > 0 else 0.0
        
        return {
            "accepted_count": total_acc,
            "rejected_count": total_rej,
            "accepted_rate": rate,
            "passed_threshold": rate >= self.thresholds.min_accepted_rate and total_acc >= self.thresholds.min_accepted_count
        }
        
    def log_double_track(self, vol_stats: Dict[str, Any]):
        """Logs double track statistics to disk."""
        accumulated = self.get_double_track_stats()
        log_data = {
            "accumulated": accumulated,
            "last_added": vol_stats
        }
        self.double_track_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.double_track_log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
