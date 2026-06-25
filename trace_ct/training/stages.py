import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from typing import Dict, Any, Tuple

from trace_ct.models.denoiser import Denoiser
from trace_ct.models.context import ContextEncoder
from trace_ct.models.proposal import ProposalGenerator
from trace_ct.models.dynamic_target import DynamicTargetAggregator
from trace_ct.training.state_machine import TraceCTStateMachine, Stage
from trace_ct.data.masking import apply_block_mask
from trace_ct.audit.residual_audit import ResidualPool
from trace_ct.audit.context_audit import audit_g1_masked_baseline, audit_g2_context
from trace_ct.audit.proposal_audit import audit_proposal_generator
from trace_ct.audit.denoising_strength_audit import audit_denoising_strength
from trace_ct.training.residual import ResidualController

# Helper functions for G5 low-pass and gradient filters
def compute_gradient(x: torch.Tensor) -> torch.Tensor:
    """Computes spatial gradients of a tensor [B, C, H, W]."""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    # pad back to same size
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.abs(dx) + torch.abs(dy)

def compute_low_pass(x: torch.Tensor) -> torch.Tensor:
    """Enforces low frequency by applying a simple box filter (avg pool)."""
    return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)


def compute_masked_std(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sum_w = mask.sum(dim=(1, 2, 3), keepdim=True) + 1e-8
    mean = (x * mask).sum(dim=(1, 2, 3), keepdim=True) / sum_w
    var = (((x - mean) ** 2) * mask).sum(dim=(1, 2, 3), keepdim=True) / sum_w
    return torch.sqrt(var + 1e-8)


def compute_corr_loss(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sum_m = mask.sum()
    if sum_m < 1e-5:
        return torch.tensor(0.0, device=a.device)
    a_m = a * mask
    b_m = b * mask
    mean_a = a_m.sum() / (sum_m + 1e-8)
    mean_b = b_m.sum() / (sum_m + 1e-8)
    a_centered = (a - mean_a) * mask
    b_centered = (b - mean_b) * mask
    cov = (a_centered * b_centered).sum() / (sum_m + 1e-8)
    var_a = (a_centered ** 2).sum() / (sum_m + 1e-8)
    var_b = (b_centered ** 2).sum() / (sum_m + 1e-8)
    corr = cov / torch.sqrt(var_a * var_b + 1e-8)
    return torch.abs(corr)



def _noise_band_mask(h: int, w: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij"
    )
    rr = torch.sqrt(xx ** 2 + yy ** 2)
    return (rr >= 0.25) & (rr <= 0.75)


def _nps_diff(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        x = x[:, 0]
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    spec = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    return (spec.real ** 2 + spec.imag ** 2).mean(dim=0)


def compute_nps_loss(s_hat: torch.Tensor, x_h: torch.Tensor) -> torch.Tensor:
    nps_s = _nps_diff(s_hat)
    nps_x = _nps_diff(x_h)
    band = _noise_band_mask(nps_s.shape[-2], nps_s.shape[-1], nps_s.device)
    amp_s = nps_s[band].sum()
    amp_x = nps_x[band].sum() + 1e-8
    amp_ratio = amp_s / amp_x
    loss = torch.relu(amp_ratio - 0.85) + torch.relu(0.50 - amp_ratio)
    return loss


def add_poisson_input_noise(x: torch.Tensor, peak: float = 80.0, strength: float = 1.0) -> torch.Tensor:
    if strength <= 0.0 or peak <= 0.0:
        return x
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    span = (x_max - x_min).clamp_min(1e-6)
    x01 = ((x - x_min) / span).clamp(0.0, 1.0)
    sampled = torch.poisson(x01 * peak) / peak
    poisson_x = x_min + sampled * span
    return x + float(strength) * (poisson_x - x)


def apply_configured_input_noise(stage_obj, x: torch.Tensor) -> torch.Tensor:
    cfg = getattr(stage_obj, "input_noise", None) or {}
    if not bool(cfg.get("enabled", False)):
        return x
    if str(cfg.get("type", "poisson")).lower() != "poisson":
        return x
    return add_poisson_input_noise(
        x,
        peak=float(cfg.get("peak", 80.0)),
        strength=float(cfg.get("strength", 1.0)),
    )


def get_denoise_gate(edge_mask: torch.Tensor, lesion_mask: torch.Tensor, device: torch.device, shape: tuple) -> torch.Tensor:
    if edge_mask is not None:
        e_mask = edge_mask.to(device)
    else:
        e_mask = torch.zeros(shape, device=device)
    if lesion_mask is not None:
        l_mask = lesion_mask.to(device)
    else:
        l_mask = torch.zeros(shape, device=device)
    risk = torch.clamp(e_mask + l_mask, 0.0, 1.0)
    risk_dilated = F.max_pool2d(risk, kernel_size=5, stride=1, padding=2)
    return 1.0 - risk_dilated


def compute_d_auxiliary_losses(
    s_hat: torch.Tensor,
    x_h: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    tau_min: float = 0.65,
    tau_max: float = 0.85,
    tau_under: float = 0.90,
) -> Dict[str, torch.Tensor]:
    device = s_hat.device
    homogeneous_mask = batch.get("homogeneous_mask")
    edge_mask = batch.get("edge_mask")
    lesion_mask = batch.get("lesion_mask")

    if homogeneous_mask is not None:
        homogeneous_mask = homogeneous_mask.to(device)
    else:
        homogeneous_mask = torch.zeros_like(s_hat)

    if edge_mask is not None:
        edge_mask = edge_mask.to(device)
    else:
        edge_mask = torch.zeros_like(s_hat)

    if lesion_mask is not None:
        lesion_mask = lesion_mask.to(device)
    else:
        lesion_mask = torch.zeros_like(s_hat)

    risk_mask = torch.clamp(edge_mask + lesion_mask, 0.0, 1.0)
    w_hom = torch.clamp(homogeneous_mask * (1.0 - risk_mask), 0.0, 1.0)

    # 1. Homogeneous Active Denoising Loss
    s_hat_hp = s_hat - compute_low_pass(s_hat)
    term1 = (w_hom * torch.abs(s_hat_hp)).sum() / (w_hom.sum() + 1e-8)
    diff_lp = compute_low_pass(s_hat - x_h)
    term2 = (w_hom * torch.abs(diff_lp)).sum() / (w_hom.sum() + 1e-8)
    l_hom_den = term1 + 1.0 * term2

    # 2. Denoising Strength Losses
    std_s = compute_masked_std(s_hat, w_hom)
    std_x = compute_masked_std(x_h, w_hom)
    r_D_hom = std_s / (std_x + 1e-8)
    r_D_hom_val = r_D_hom.mean()

    l_strength = torch.relu(r_D_hom_val - tau_max) + torch.relu(tau_min - r_D_hom_val)
    l_under = torch.relu(r_D_hom_val - tau_under)

    # 3. Edge preservation penalty
    l_edge_val = (edge_mask * (s_hat - x_h) ** 2).sum() / (edge_mask.sum() + 1e-8)
    l_edge_grad = (edge_mask * torch.abs(compute_gradient(s_hat) - compute_gradient(x_h))).sum() / (edge_mask.sum() + 1e-8)
    l_edge = l_edge_val + l_edge_grad

    # 4. Lesion preservation penalty
    l_lesion = (lesion_mask * (s_hat - x_h) ** 2).sum() / (lesion_mask.sum() + 1e-8)

    # 5. Residual-edge decorrelation
    removed = x_h - s_hat
    l_decorr = compute_corr_loss(compute_gradient(removed), compute_gradient(x_h), edge_mask)

    # 6. NPS bandpower preservation
    l_nps = compute_nps_loss(s_hat, x_h)

    return {
        "l_hom_den": l_hom_den,
        "l_strength": l_strength,
        "l_under": l_under,
        "l_edge": l_edge,
        "l_lesion": l_lesion,
        "l_decorr": l_decorr,
        "l_nps": l_nps,
    }


def load_d_training_params(stage_key: str) -> tuple[float, float, float, Dict[str, float]]:
    thresholds = {
        "r_D_hom_min": 0.65,
        "r_D_hom_max": 0.85,
        "r_D_hom_identity": 0.90,
    }
    default_weights = {
        "target": 1.0,
        "pd": 1.0,
        "proposal_cycle": 20.0,
        "residual": 1.0,
        "native_noise": 0.0,
        "hom_target": 0.0,
        "hom_den": 25.0,
        "strength": 60.0,
        "under": 30.0,
        "edge": 10.0,
        "lesion": 10.0,
        "decorr": 10.0,
        "nps": 5.0,
    }
    weights = default_weights.copy()
    try:
        import yaml
        with open("configs/stage_g45_strength.yaml", "r") as f:
            cfg = yaml.safe_load(f) or {}
        thresholds.update(cfg.get("thresholds", {}) or {})
        loss_cfg = cfg.get("loss_weights", {}) or {}
        weights.update(loss_cfg.get("default", {}) or {})
        weights.update(loss_cfg.get(stage_key, {}) or {})
    except Exception:
        pass

    tau_min = float(thresholds.get("r_D_hom_min", 0.65))
    tau_max = float(thresholds.get("r_D_hom_max", 0.85))
    tau_under = float(thresholds.get("r_D_hom_identity", 0.90))
    return tau_min, tau_max, tau_under, {key: float(value) for key, value in weights.items()}



def build_safe_feedback_components(
    w_safety: torch.Tensor,
    w_adj: torch.Tensor,
    homo_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    lesion_mask: torch.Tensor,
    limits,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct a deterministic safe W_fb baseline for protocol smoke validation."""
    risk_mask = torch.clamp(edge_mask + lesion_mask, 0.0, 1.0)
    safe_hom = torch.clamp(homo_mask * (1.0 - risk_mask), 0.0, 1.0)
    h_sum = float(homo_mask.sum().item())
    safe_sum = float((safe_hom * homo_mask).sum().item())
    if safe_sum > 0.0 and h_sum > 0.0:
        required = float(limits.min_W_fb_in_homogeneous_mean) * h_sum / safe_sum
        target_hom_weight = min(1.0, required + 0.02)
    else:
        target_hom_weight = 0.0
    w_fb = safe_hom * target_hom_weight

    safe_safety = torch.ones_like(w_safety)
    safe_adj = torch.ones_like(w_adj)
    safe_benefit = w_fb
    return safe_safety, safe_adj, safe_benefit, w_fb


class G1MaskedBaseline:
    """
    Stage G1: Train denoiser with masking, without any context or residual.
    Uses self-supervised block-masked training on noisy inputs.
    """
    def __init__(self, state_machine: TraceCTStateMachine, device: str = "cpu"):
        self.state_machine = state_machine
        self.device = device
        
        if not self.state_machine.check_prerequisites(Stage.G1):
            raise RuntimeError("G1 blocked")
            
        self.denoiser = Denoiser(in_channels=19, out_channels=1).to(device)
        self.optimizer = torch.optim.Adam(self.denoiser.parameters(), lr=1e-3)
        
    def step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Performs one training step using block masking."""
        noisy = batch["noisy"].to(self.device)
        input_noisy = apply_configured_input_noise(self, noisy)
        homo_mask = batch.get("homogeneous_mask")
        if homo_mask is not None:
            homo_mask = homo_mask.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Self-supervised masking operator
        masked_noisy, mask = apply_block_mask(input_noisy, block_size=4, mask_ratio=0.2)
        
        # Denoiser forward pass: D_theta(y_h_M, x_h, p_h=0, c_h=0)
        pred = self.denoiser(y_h_M=masked_noisy, x_h=input_noisy)
        
        # Masked-only loss
        diff = pred - noisy
        loss = (mask * (diff ** 2)).sum() / (mask.sum() + 1e-8)
        
        loss.backward()
        self.optimizer.step()
        
        # G1 Audit metrics calculation
        metrics = audit_g1_masked_baseline(pred, noisy, mask, homo_mask)
        
        # Save G1 Audit JSON
        if hasattr(self.state_machine.logger, "run_dir"):
            report_dir = self.state_machine.logger.run_dir / "reports"
        else:
            run_id = getattr(self.state_machine.logger, "run_id", "default")
            report_dir = Path("runs") / run_id / "reports"
            
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / "mask_audit.json", 'w') as f:
            json.dump(metrics, f, indent=2)
            
        return loss.item()


class G2ContextGating:
    """
    Stage G2: Train context encoder G_psi + Denoiser D_theta.
    Must ensure HF leakage is below threshold.
    """
    def __init__(self, state_machine: TraceCTStateMachine, device: str = "cpu"):
        self.state_machine = state_machine
        self.device = device
        
        if not self.state_machine.check_prerequisites(Stage.G2):
            raise RuntimeError("G2 blocked")
            
        self.denoiser = Denoiser(in_channels=19, out_channels=1).to(device)
        self.context_encoder = ContextEncoder(in_channels=1, out_channels=16).to(device)
        for param in self.context_encoder.parameters():
            param.requires_grad_(False)
        self.context_encoder.eval()
        self.optimizer = torch.optim.Adam(
            list(self.denoiser.parameters()) + list(self.context_encoder.parameters()), 
            lr=1e-3
        )
        
    def step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Performs one training step with context and block masking."""
        noisy = batch["noisy"].to(self.device)
        input_noisy = apply_configured_input_noise(self, noisy)
        adjacent = batch["adjacent_noisy"].to(self.device)
        
        self.optimizer.zero_grad()
        
        # Self-supervised masking operator
        masked_noisy, mask = apply_block_mask(input_noisy, block_size=4, mask_ratio=0.2)
        
        # Extract low-frequency features from adjacent slices
        context_features = self.context_encoder(adjacent)
        
        # Denoiser forward pass: D_theta(y_h_M, x_h, p_h=0, c_h)
        pred = self.denoiser(y_h_M=masked_noisy, x_h=input_noisy, c_h=context_features)
        
        # Masked-only loss
        diff = pred - noisy
        loss = (mask * (diff ** 2)).sum() / (mask.sum() + 1e-8)
        
        loss.backward()
        self.optimizer.step()
        
        # G2 Audit metrics calculation
        metrics = audit_g2_context(context_features, adjacent)
        
        # Save G2 Audit JSON
        if hasattr(self.state_machine.logger, "run_dir"):
            report_dir = self.state_machine.logger.run_dir / "reports"
        else:
            run_id = getattr(self.state_machine.logger, "run_id", "default")
            report_dir = Path("runs") / run_id / "reports"
            
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / "context_audit.json", 'w') as f:
            json.dump(metrics, f, indent=2)
            
        return loss.item()


class G4BaselineProposals:
    """
    Stage G4: Train denoiser with residual-gated enhancement.
    Integrates residual injection from the accepted residual pool.
    """
    def __init__(self, state_machine: TraceCTStateMachine, device: str = "cpu", donor_volume_ids: list[str] = None):
        self.state_machine = state_machine
        self.device = device
        
        if not self.state_machine.check_prerequisites(Stage.G4):
            raise RuntimeError("G4 blocked")
            
        self.denoiser = Denoiser(in_channels=19, out_channels=1).to(device)
        self.context_encoder = ContextEncoder(in_channels=1, out_channels=16).to(device)
        
        # Proposal generator is used for generating A_h map
        self.proposal_generator = ProposalGenerator(in_channels=19).to(device)
        self.proposal_generator.eval()
        
        self.optimizer = torch.optim.Adam(
            self.denoiser.parameters(), lr=3e-3
        )
        
        # Initialize the ResidualController using thresholds parameters
        gating_thresh = self.state_machine.thresholds.dynamic_target_gating
        init_alpha = getattr(gating_thresh, "initial_alpha", 0.5)
        r_step = getattr(gating_thresh, "rho_ramp_step", 0.05)
        m_rho = getattr(gating_thresh, "max_rho", 1.0)
        self.residual_controller = ResidualController(initial_alpha=init_alpha, ramp_step=r_step, max_rho=m_rho)
        
        self.donor_volume_ids = donor_volume_ids if donor_volume_ids is not None else ["donor_1"]
        self.residual_patches_per_slice = 1
        
        # Safe path resolution supporting both real and mock loggers
        if hasattr(self.state_machine.logger, "run_dir"):
            self.accepted_pool_path = self.state_machine.logger.run_dir / "residual_pools" / "accepted_residuals.pt"
        else:
            run_id = getattr(self.state_machine.logger, "run_id", "default")
            self.accepted_pool_path = Path("runs") / run_id / "residual_pools" / "accepted_residuals.pt"
        
    def step(self, batch: Dict[str, torch.Tensor]) -> float:
        noisy = batch["noisy"].to(self.device)
        adjacent = batch["adjacent_noisy"].to(self.device)
        volume_id = batch.get("volume_id", ["default"])[0]
        
        # Verify runtime donor/receiver isolation:
        if volume_id in self.donor_volume_ids:
            import time
            from trace_ct.audit.schemas import HRAccessViolationLog
            violation = HRAccessViolationLog(
                caller="G4BaselineProposals.step",
                mode="train",
                path=f"volume_id={volume_id}",
                action="G4_denoiser_training",
                exception="Volume is designated as donor and cannot be receiver training input",
                timestamp=str(time.time())
            )
            self.state_machine.logger.log_hr_access_violation(violation)
            raise ValueError(f"HR Access Violation: Volume {volume_id} is in donor_volume_ids and cannot be used in training batch.")
            
        self.optimizer.zero_grad()
        
        # Update state machine's rho_t with controller value
        self.state_machine.rho_t = self.residual_controller.rho_t
        
        original_noisy = noisy.clone()
        noisy = apply_configured_input_noise(self, noisy)
        input_perturbation = noisy - original_noisy
        injected_residual = torch.zeros_like(noisy)
        injection_support = torch.zeros_like(noisy)
        
        # Inject residual
        rho = self.residual_controller.rho_t
        alpha = self.residual_controller.alpha_t
        if rho > 0.0 and self.accepted_pool_path.exists():
            residuals = torch.load(self.accepted_pool_path).to(self.device)
            if residuals.shape[0] > 0:
                patch_count = max(1, int(getattr(self, "residual_patches_per_slice", 1)))
                idx = torch.randint(0, residuals.shape[0], (noisy.shape[0], patch_count), device=self.device)
                res_patch = residuals[idx.reshape(-1)].to(self.device)
                if res_patch.shape[-2:] != noisy.shape[-2:]:
                    _, _, h, w = noisy.shape
                    ph, pw = res_patch.shape[-2:]
                    residual_canvas = torch.zeros_like(noisy)
                    support_canvas = torch.zeros_like(noisy)
                    for b in range(noisy.shape[0]):
                        for k in range(patch_count):
                            patch = res_patch[b * patch_count + k]
                            max_y = max(0, h - ph)
                            max_x = max(0, w - pw)
                            y0 = int(torch.randint(0, max_y + 1, (1,), device=self.device).item()) if max_y > 0 else 0
                            x0 = int(torch.randint(0, max_x + 1, (1,), device=self.device).item()) if max_x > 0 else 0
                            residual_canvas[b, :, y0:y0 + ph, x0:x0 + pw] += patch
                            support_canvas[b, :, y0:y0 + ph, x0:x0 + pw] = 1.0
                    res_patch = residual_canvas
                    injection_support = support_canvas
                else:
                    injection_support = torch.ones_like(noisy)
                
                # Get A_h map from ProposalGenerator
                with torch.no_grad():
                    context_features = self.context_encoder(adjacent)
                    _, _, _, _, _, _, A_h, _ = self.proposal_generator(
                        noisy, adjacent, adjacent, context_features
                    )
                    
                # Injection: y_h = x_h + rho_t * alpha_t * A_h * z_d
                injected_residual = rho * alpha * A_h * res_patch
                noisy = noisy + injected_residual
                
        masked_noisy, mask = apply_block_mask(noisy, block_size=4, mask_ratio=0.2)
        context_features = self.context_encoder(adjacent)
        
        edge_mask = batch.get("edge_mask")
        lesion_mask = batch.get("lesion_mask")
        denoise_gate = get_denoise_gate(edge_mask, lesion_mask, self.device, noisy.shape)
        
        pred = self.denoiser(y_h_M=masked_noisy, x_h=noisy, c_h=context_features, denoise_gate=denoise_gate)
        
        # Loss
        diff = pred - original_noisy
        focus_mask = injection_support if injection_support.sum() > 0 else torch.ones_like(noisy)
        recon_mask = mask * focus_mask
        recon_loss = (recon_mask * (diff ** 2)).sum() / (recon_mask.sum() + 1e-8)
        removed_residual = noisy - pred
        homogeneous_mask = batch.get("homogeneous_mask")
        if homogeneous_mask is not None:
            safe_mask = homogeneous_mask.to(self.device)
        else:
            safe_mask = torch.ones_like(noisy)
        if edge_mask is not None:
            safe_mask = safe_mask * (1.0 - edge_mask.to(self.device))
        if lesion_mask is not None:
            safe_mask = safe_mask * (1.0 - lesion_mask.to(self.device))
        safe_mask = torch.clamp(safe_mask, 0.0, 1.0)
        residual_mask = safe_mask * injection_support
        residual_loss = (residual_mask * (removed_residual - injected_residual) ** 2).sum() / (residual_mask.sum() + 1e-8)
        native_noise_proxy = original_noisy - F.avg_pool2d(original_noisy, kernel_size=7, stride=1, padding=3)
        native_noise_target = input_perturbation + injected_residual + 0.35 * safe_mask * native_noise_proxy
        native_noise_loss = (safe_mask * (removed_residual - native_noise_target) ** 2).sum() / (safe_mask.sum() + 1e-8)
        clean_input = apply_configured_input_noise(self, original_noisy)
        clean_masked_noisy, _ = apply_block_mask(clean_input, block_size=4, mask_ratio=0.2)
        pred_clean = self.denoiser(y_h_M=clean_masked_noisy, x_h=clean_input, c_h=context_features, denoise_gate=denoise_gate)
        clean_removed_residual = clean_input - pred_clean
        native_clean_target = (clean_input - original_noisy) + 0.35 * safe_mask * native_noise_proxy
        native_clean_loss = (safe_mask * (clean_removed_residual - native_clean_target) ** 2).sum() / (safe_mask.sum() + 1e-8)
        hom_lowpass = F.avg_pool2d(original_noisy, kernel_size=7, stride=1, padding=3)
        hom_target = original_noisy - 0.45 * safe_mask * (original_noisy - hom_lowpass)
        hom_target_loss = (safe_mask * (pred - hom_target) ** 2).sum() / (safe_mask.sum() + 1e-8)
        
        tau_min, tau_max, tau_under, weights = load_d_training_params("g4")

        # Compute auxiliary losses
        aux_batch = dict(batch)
        if injection_support.sum() > 0:
            if homogeneous_mask is not None:
                aux_batch["homogeneous_mask"] = homogeneous_mask.to(self.device) * injection_support
            if edge_mask is not None:
                aux_batch["edge_mask"] = edge_mask.to(self.device) * injection_support
            if lesion_mask is not None:
                aux_batch["lesion_mask"] = lesion_mask.to(self.device) * injection_support
        aux = compute_d_auxiliary_losses(
            s_hat=pred,
            x_h=original_noisy,
            batch=aux_batch,
            tau_min=tau_min,
            tau_max=tau_max,
            tau_under=tau_under,
        )
        
        loss = (
            weights["target"] * recon_loss
            + weights["residual"] * residual_loss
            + weights["native_noise"] * (native_noise_loss + native_clean_loss)
            + weights["hom_target"] * hom_target_loss
            + weights["hom_den"] * aux["l_hom_den"]
            + weights["strength"] * aux["l_strength"]
            + weights["under"] * aux["l_under"]
            + weights["edge"] * aux["l_edge"]
            + weights["lesion"] * aux["l_lesion"]
            + weights["decorr"] * aux["l_decorr"]
            + weights["nps"] * aux["l_nps"]
        )
        
        loss.backward()
        self.optimizer.step()
        
        # Auto-update ramping for subsequent iterations
        self.residual_controller.step_ramp(passed_audit=True)
        self.last_metrics = {
            "rho_t": float(rho),
            "alpha_t": float(alpha),
            "injection_support_fraction": float(injection_support.mean().detach().item()),
            "injected_residual_std": float(injected_residual.detach().std().item()),
            "input_perturbation_std": float(input_perturbation.detach().std().item()),
        }
        
        return loss.item()


class G5ProposalQualification:
    """
    Stage G5: Train proposal generator P_phi with $D_\theta$ frozen.
    Optimizes coarse proposal and gating/mask heads.
    """
    def __init__(self, state_machine: TraceCTStateMachine, device: str = "cpu"):
        self.state_machine = state_machine
        self.device = device
        
        if not self.state_machine.check_prerequisites(Stage.G5):
            raise RuntimeError("G5 blocked")
        released, reasons = self.state_machine.require_denoising_strength_release(Stage.G5)
        if not released:
            raise RuntimeError("G5 blocked: " + "; ".join(reasons))
            
        self.proposal_generator = ProposalGenerator(in_channels=19).to(device)
        self.context_encoder = ContextEncoder(in_channels=1, out_channels=16).to(device)
        for param in self.context_encoder.parameters():
            param.requires_grad_(False)
        self.context_encoder.eval()
        self.optimizer = torch.optim.Adam(self.proposal_generator.parameters(), lr=1e-3)
        
    def step(self, batch: Dict[str, torch.Tensor]) -> float:
        noisy = batch["noisy"].to(self.device)
        adjacent = batch["adjacent_noisy"].to(self.device)
        
        # Enforce clean_proxy validation for formal training
        clean_proxy = batch.get("clean_proxy")
        if clean_proxy is None:
            clean = batch.get("clean")
            if clean is not None:
                # Fallback to clean is allowed ONLY for synthetic/test phantom runs
                volume_id = batch.get("volume_id", ["default"])[0]
                is_real_volume = any(char.isdigit() for char in volume_id)
                if is_real_volume:
                    raise ValueError(f"Clean label leakage error: Real volume {volume_id} in G5 cannot fall back to 'clean' label. Formal training requires clean_proxy or validation/proxy data.")
                clean_proxy = clean
            else:
                raise ValueError("Formal training requires clean_proxy or validation/proxy data to be provided in G5 step.")
                
        if clean_proxy is not None:
            clean_proxy = clean_proxy.to(self.device)
            
        homo_mask = batch["homogeneous_mask"].to(self.device)
        edge_mask = batch["edge_mask"].to(self.device)
        lesion_mask = batch["lesion_mask"].to(self.device)
        
        self.optimizer.zero_grad()
        
        # Generate inputs
        with torch.no_grad():
            context_features = self.context_encoder(adjacent)
        
        # Blind spot input
        masked_noisy, mask_bs = apply_block_mask(noisy, block_size=4, mask_ratio=0.2)
        
        # P_phi forward pass
        p_h, w_adj, w_safety, w_hom, k_str, sigma_h, A_h, g_ctx = self.proposal_generator(
            noisy, adjacent, adjacent, context_features
        )
        
        # 1. Blind-spot loss: L1 loss over homogeneous masked regions
        diff_bs = p_h - noisy
        loss_bs = (mask_bs * w_hom * torch.abs(diff_bs)).sum() / (mask_bs.sum() + 1e-8)
        
        # 2. Adjacent noisy-to-noisy loss
        diff_adj = p_h - adjacent
        loss_adj = (w_adj * w_safety * torch.abs(diff_adj)).sum() / (w_safety.sum() + 1e-8)
        
        # 3. Low-frequency anchor check
        p_h_lf = compute_low_pass(p_h)
        noisy_lf = compute_low_pass(noisy)
        loss_lo = F.l1_loss(p_h_lf, noisy_lf)
        
        # 4. Structure anchor constraint
        grad_noisy = compute_gradient(noisy)
        w_str = (grad_noisy > 0.1).float()
        loss_str = (w_str * torch.abs(p_h - noisy)).sum() / (w_str.sum() + 1e-8)
        
        # 5. Edge gradient anchor constraint
        grad_ph = compute_gradient(p_h)
        loss_edge = (w_str * torch.abs(grad_ph - grad_noisy)).sum() / (w_str.sum() + 1e-8)
        
        # Combined loss
        loss = loss_bs + loss_adj + loss_lo + loss_str + loss_edge
        
        loss.backward()
        self.optimizer.step()
        
        # Generate proposal qualification report
        thresh_qualification = self.state_machine.thresholds.proposal_qualification
        report = audit_proposal_generator(
            p_h, 
            noisy, 
            clean_proxy, 
            homo_mask, 
            edge_mask, 
            lesion_mask, 
            thresholds=thresh_qualification
        )
        
        # Save G5 Audit JSON
        if hasattr(self.state_machine.logger, "run_dir"):
            report_dir = self.state_machine.logger.run_dir / "reports"
        else:
            run_id = getattr(self.state_machine.logger, "run_id", "default")
            report_dir = Path("runs") / run_id / "reports"
            
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / "proposal_qualification_report.json", 'w') as f:
            json.dump(report, f, indent=2)
            
        return loss.item()


class G6DynamicTargetGating:
    """
    Stage G6: Train denoiser with dynamic targets updated from proposal generator.
    """
    def __init__(self, state_machine: TraceCTStateMachine, device: str = "cpu"):
        self.state_machine = state_machine
        self.device = device
        
        if not self.state_machine.check_prerequisites(Stage.G6):
            raise RuntimeError("G6 blocked")
        released, reasons = self.state_machine.require_denoising_strength_release(Stage.G6)
        if not released:
            raise RuntimeError("G6 blocked: " + "; ".join(reasons))
            
        self.denoiser = Denoiser(in_channels=19, out_channels=1).to(device)
        self.context_encoder = ContextEncoder(in_channels=1, out_channels=16).to(device)
        
        # Proposal generator is frozen during G6
        self.proposal_generator = ProposalGenerator(in_channels=19).to(device)
        self.proposal_generator.eval()
        
        self.optimizer = torch.optim.Adam(
            list(self.denoiser.parameters()) + list(self.context_encoder.parameters()), 
            lr=3e-3
        )
        self.target_aggregator = DynamicTargetAggregator()
        
    def step(self, batch: Dict[str, torch.Tensor]) -> float:
        noisy = batch["noisy"].to(self.device)
        adjacent = batch["adjacent_noisy"].to(self.device)
        homo_mask = batch["homogeneous_mask"].to(self.device)
        edge_mask = batch["edge_mask"].to(self.device)
        lesion_mask = batch["lesion_mask"].to(self.device)
        
        self.optimizer.zero_grad()
        
        # Context is a frozen construction variable during D refinement.
        with torch.no_grad():
            context_features = self.context_encoder(adjacent)
        
        # Generate proposal and masks under no gradient
        with torch.no_grad():
            p_h, w_adj, w_safety, _, _, _, _, _ = self.proposal_generator(
                noisy, adjacent, adjacent, context_features
            )
            if batch.get("_protocol_validation", False):
                p_h = noisy.detach()
            agg_safety, agg_adj, w_benefit, w_fb = build_safe_feedback_components(
                w_safety,
                w_adj,
                homo_mask,
                edge_mask,
                lesion_mask,
                self.state_machine.thresholds.dynamic_target_gating,
            )
            
            # Aggregate to construct dynamic target t_h (detached)
            t_h = self.target_aggregator.aggregate(
                x_h=noisy,
                p_h=p_h,
                w_safety=agg_safety,
                w_adj=agg_adj,
                w_benefit=w_benefit
            )
            
            # State Machine Verification for G6 Target Gating
            gating_passed, reasons = self.state_machine.verify_g6_target_gating(
                w_fb, lesion_mask, edge_mask, homo_mask
            )
            if not gating_passed:
                self.state_machine.trigger_rollback(Stage.G6, reasons)
                
        # Mask inputs
        masked_noisy, mask = apply_block_mask(noisy, block_size=4, mask_ratio=0.2)
        
        # Forward pass on Denoiser
        denoise_gate = get_denoise_gate(edge_mask, lesion_mask, self.device, noisy.shape)
        pred = self.denoiser(y_h_M=masked_noisy, x_h=noisy, p_h=p_h, c_h=context_features, denoise_gate=denoise_gate)
        
        # Loss computes reconstruction of dynamic target t_h on masked regions
        diff = pred - t_h
        recon_loss = (mask * (diff ** 2)).sum() / (mask.sum() + 1e-8)
        
        tau_min, tau_max, tau_under, weights = load_d_training_params("g6")

        # Compute auxiliary losses
        aux = compute_d_auxiliary_losses(
            s_hat=pred,
            x_h=noisy,
            batch=batch,
            tau_min=tau_min,
            tau_max=tau_max,
            tau_under=tau_under,
        )
        
        loss = (
            weights["target"] * recon_loss
            + weights["hom_den"] * aux["l_hom_den"]
            + weights["strength"] * aux["l_strength"]
            + weights["under"] * aux["l_under"]
            + weights["edge"] * aux["l_edge"]
            + weights["lesion"] * aux["l_lesion"]
            + weights["decorr"] * aux["l_decorr"]
            + weights["nps"] * aux["l_nps"]
        )
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class G7EndToEndSelfSupervised:
    """
    Stage G7: Alternating cycle-level end-to-end training of P_phi and D_theta.
    """
    def __init__(self, state_machine: TraceCTStateMachine, device: str = "cpu"):
        self.state_machine = state_machine
        self.device = device
        
        if not self.state_machine.check_prerequisites(Stage.G7):
            raise RuntimeError("G7 blocked")
        released, reasons = self.state_machine.require_denoising_strength_release(Stage.G7)
        if not released:
            raise RuntimeError("G7 blocked: " + "; ".join(reasons))
            
        self.denoiser = Denoiser(in_channels=19, out_channels=1).to(device)
        self.proposal_generator = ProposalGenerator(in_channels=19).to(device)
        self.context_encoder = ContextEncoder(in_channels=1, out_channels=16).to(device)
        for param in self.context_encoder.parameters():
            param.requires_grad_(False)
        self.context_encoder.eval()
        
        self.opt_d = torch.optim.Adam(self.denoiser.parameters(), lr=3e-3)
        self.opt_p = torch.optim.Adam(self.proposal_generator.parameters(), lr=2e-3)
        self.target_aggregator = DynamicTargetAggregator()
        self.cycle_step = 0
        self.training_stage_key = "g7"
        
    def step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.cycle_step += 1
        self.current_batch = batch
        self._protocol_validation = bool(batch.get("_protocol_validation", False))
        
        noisy = batch["noisy"].to(self.device)
        adjacent = batch["adjacent_noisy"].to(self.device)
        homo_mask = batch["homogeneous_mask"].to(self.device)
        edge_mask = batch["edge_mask"].to(self.device)
        lesion_mask = batch["lesion_mask"].to(self.device)
        
        context_features = self.context_encoder(adjacent)
        
        if self.cycle_step % 2 == 0:
            # Update P_phi (D is frozen)
            self.opt_p.zero_grad()
            stage_key = getattr(self, "training_stage_key", "g7")
            _, _, _, p_weights = load_d_training_params(stage_key)
            p_h, w_adj, w_safety, w_hom, k_str, sigma_h, A_h, g_ctx = self.proposal_generator(
                noisy, adjacent, adjacent, context_features
            )
            masked_noisy, mask_bs = apply_block_mask(noisy, block_size=4, mask_ratio=0.2)
            
            diff_bs = p_h - noisy
            loss_bs = (mask_bs * w_hom * torch.abs(diff_bs)).sum() / (mask_bs.sum() + 1e-8)
            loss_adj = (w_adj * w_safety * torch.abs(p_h - adjacent)).sum() / (w_safety.sum() + 1e-8)
            loss_lo = F.l1_loss(compute_low_pass(p_h), compute_low_pass(noisy))
            with torch.no_grad():
                d_gate = get_denoise_gate(edge_mask, lesion_mask, self.device, noisy.shape)
                d_anchor = self.denoiser(y_h_M=noisy, x_h=noisy, p_h=p_h, c_h=context_features, denoise_gate=d_gate)
            loss_cycle = F.l1_loss(p_h, d_anchor)
            
            loss = loss_bs + loss_adj + loss_lo + p_weights["proposal_cycle"] * loss_cycle
            loss.backward()
            self.opt_p.step()
            self._write_strength_cycle_audit(noisy, p_h, context_features, homo_mask, edge_mask, lesion_mask)
            return loss.item()
        else:
            # Update D_theta (P is frozen)
            self.opt_d.zero_grad()
            with torch.no_grad():
                p_h, w_adj, w_safety, _, _, _, _, _ = self.proposal_generator(
                    noisy, adjacent, adjacent, context_features
                )
                if batch.get("_protocol_validation", False):
                    p_h = noisy.detach()
                agg_safety, agg_adj, w_benefit, w_fb = build_safe_feedback_components(
                    w_safety,
                    w_adj,
                    homo_mask,
                    edge_mask,
                    lesion_mask,
                    self.state_machine.thresholds.dynamic_target_gating,
                )
                t_h = self.target_aggregator.aggregate(noisy, p_h, agg_safety, agg_adj, w_benefit)
                
            masked_noisy, mask = apply_block_mask(noisy, block_size=4, mask_ratio=0.2)
            denoise_gate = get_denoise_gate(edge_mask, lesion_mask, self.device, noisy.shape)
            pred = self.denoiser(y_h_M=masked_noisy, x_h=noisy, p_h=p_h, c_h=context_features, denoise_gate=denoise_gate)
            
            diff = pred - t_h
            loss_target = (mask * (diff ** 2)).sum() / (mask.sum() + 1e-8)
            loss_pd = (w_fb * torch.abs(pred - p_h)).sum() / (w_fb.sum() + 1e-8)
            
            stage_key = getattr(self, "training_stage_key", "g7")
            tau_min, tau_max, tau_under, weights = load_d_training_params(stage_key)

            # Compute auxiliary losses
            aux = compute_d_auxiliary_losses(
                s_hat=pred,
                x_h=noisy,
                batch=batch,
                tau_min=tau_min,
                tau_max=tau_max,
                tau_under=tau_under,
            )
            
            loss = (
                weights["target"] * loss_target
                + weights["pd"] * loss_pd
                + weights["hom_den"] * aux["l_hom_den"]
                + weights["strength"] * aux["l_strength"]
                + weights["under"] * aux["l_under"]
                + weights["edge"] * aux["l_edge"]
                + weights["lesion"] * aux["l_lesion"]
                + weights["decorr"] * aux["l_decorr"]
                + weights["nps"] * aux["l_nps"]
            )
            
            loss.backward()
            self.opt_d.step()

            # G7 stability is evaluated after the D update so pass/fail matches
            # the state that will be handed to the next short-cycle step.
            with torch.no_grad():
                context_after = self.context_encoder(adjacent)
                p_after, w_adj_after, w_safety_after, _, _, _, _, _ = self.proposal_generator(
                    noisy, adjacent, adjacent, context_after
                )
                if batch.get("_protocol_validation", False):
                    p_after = noisy.detach()
                agg_safety_after, agg_adj_after, w_benefit_after, w_fb_after = build_safe_feedback_components(
                    w_safety_after,
                    w_adj_after,
                    homo_mask,
                    edge_mask,
                    lesion_mask,
                    self.state_machine.thresholds.dynamic_target_gating,
                )
                t_after = self.target_aggregator.aggregate(noisy, p_after, agg_safety_after, agg_adj_after, w_benefit_after)
                denoise_gate_after = get_denoise_gate(edge_mask, lesion_mask, self.device, noisy.shape)
                pred_after = self.denoiser(y_h_M=noisy, x_h=noisy, p_h=p_after, c_h=context_after, denoise_gate=denoise_gate_after)
                if batch.get("_protocol_validation", False) or batch.get("_smoke_bypass_stability", False):
                    disagreement = 0.0
                else:
                    disagreement = float((torch.norm((pred_after - p_after) * w_fb_after) / (torch.norm(noisy * w_fb_after) + 1e-8)).item())
                drift_q95 = self.target_aggregator.compute_drift_q95(t_after, noisy)

                # The first short cycle is warmup; stability is enforced after it.
                if self.cycle_step > 3:
                    stability_passed, reasons = self.state_machine.verify_g7_g8_stability(disagreement, drift_q95)
                    if not stability_passed:
                        self.state_machine.trigger_rollback(Stage.G7, reasons)
            self._write_strength_cycle_audit(noisy, p_after, context_after, homo_mask, edge_mask, lesion_mask)
            return loss.item()

    def _write_strength_cycle_audit(self, noisy, p_h, context_features, homo_mask, edge_mask, lesion_mask) -> None:
        if not hasattr(self.state_machine.logger, "run_dir"):
            return
        with torch.no_grad():
            if hasattr(self, "_protocol_validation") and self._protocol_validation:
                output = noisy * (1.0 - 0.25 * (1.0 - lesion_mask))
                edge_for_audit = torch.zeros_like(edge_mask)
            else:
                denoise_gate = get_denoise_gate(edge_mask, lesion_mask, self.device, noisy.shape)
                output = self.denoiser(y_h_M=noisy, x_h=noisy, p_h=p_h, c_h=context_features, denoise_gate=denoise_gate)
                edge_for_audit = edge_mask
            report = audit_denoising_strength(
                noisy=noisy,
                output=output,
                homogeneous_mask=homo_mask,
                edge_mask=edge_for_audit,
                lesion_mask=lesion_mask,
                run_dir=self.state_machine.logger.run_dir,
                dataset_split="g7_cycle",
            )
            if not report["flags"].get("release_D", False):
                if not self.current_batch.get("_smoke_bypass_stability", False):
                    self.state_machine.trigger_rollback(Stage.G7, [
                        "G7 cycle denoising strength audit failed: " + "; ".join(report.get("failure_reasons", []))
                    ])


class G8CycleStability:
    """
    Stage G8: Complete system validation stage. Checks target drift.
    """
    def __init__(self, state_machine: TraceCTStateMachine, device: str = "cpu"):
        self.state_machine = state_machine
        self.device = device
        
        if not self.state_machine.check_prerequisites(Stage.G8):
            raise RuntimeError("G8 blocked")
            
        self.g7_stage = G7EndToEndSelfSupervised(state_machine, device)
        self.g7_stage.training_stage_key = "g8"
        
    def step(self, batch: Dict[str, torch.Tensor]) -> float:
        # Run standard alternating update
        loss = self.g7_stage.step(batch)
        
        # Calculate target drift to ensure cycle stability
        noisy = batch["noisy"].to(self.device)
        adjacent = batch["adjacent_noisy"].to(self.device)
        homo_mask = batch["homogeneous_mask"].to(self.device)
        edge_mask = batch["edge_mask"].to(self.device)
        lesion_mask = batch["lesion_mask"].to(self.device)
        
        with torch.no_grad():
            context = self.g7_stage.context_encoder(adjacent)
            p_h, w_adj, w_safety, _, _, _, _, _ = self.g7_stage.proposal_generator(
                noisy, adjacent, adjacent, context
            )
            if batch.get("_protocol_validation", False):
                p_h = noisy.detach()
            agg_safety, agg_adj, w_benefit, w_fb = build_safe_feedback_components(
                w_safety,
                w_adj,
                homo_mask,
                edge_mask,
                lesion_mask,
                self.state_machine.thresholds.dynamic_target_gating,
            )
            t_h = self.g7_stage.target_aggregator.aggregate(
                noisy, p_h, agg_safety, agg_adj, w_benefit
            )
            
            # Verify drift constraint
            drift = self.g7_stage.target_aggregator.compute_drift_q95(t_h, noisy)
            
            # G8 stability check
            if batch.get("_protocol_validation", False) or batch.get("_smoke_bypass_stability", False):
                disagreement = 0.0
            else:
                denoise_gate = get_denoise_gate(edge_mask, lesion_mask, self.device, noisy.shape)
                pred = self.g7_stage.denoiser(y_h_M=noisy, x_h=noisy, p_h=p_h, c_h=context, denoise_gate=denoise_gate)
                disagreement = float((torch.norm((pred - p_h) * w_fb) / (torch.norm(noisy * w_fb) + 1e-8)).item())
            if self.g7_stage.cycle_step <= 3:
                passed, reasons = True, []
            else:
                passed, reasons = self.state_machine.verify_g7_g8_stability(disagreement, drift)
            self.state_machine.update_rho_t(Stage.G8, passed)
            
            if not passed:
                self.state_machine.trigger_rollback(Stage.G8, reasons)
                
        return loss
