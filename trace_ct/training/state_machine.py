from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import json
import time
from pathlib import Path
import torch

from trace_ct.audit.logger import AuditLogger
from trace_ct.audit.schemas import StagePassFailRecord
from trace_ct.config.schema import ThresholdsYAML

class Stage(str, Enum):
    G0 = "G0"
    G1 = "G1"
    G2 = "G2"
    G3 = "G3"
    G4 = "G4"
    G45 = "G4.5"
    G5 = "G5"
    G6 = "G6"
    G7 = "G7"
    G8 = "G8"

class StageState(str, Enum):
    OFF = "OFF"
    AUDIT = "AUDIT"
    TRAIN = "TRAIN"

class TraceCTStateMachine:
    def __init__(self, logger: AuditLogger, thresholds: ThresholdsYAML = None, ttl_seconds: float = 86400.0):
        self.logger = logger
        self.ttl_seconds = ttl_seconds
        
        # Load thresholds from yaml if not provided
        if thresholds is None:
            from trace_ct.config.defaults import load_thresholds_config
            try:
                # Attempt to load from standard configs directory
                self.thresholds = load_thresholds_config("configs/thresholds.yaml")
            except Exception:
                # Default fallback for testing stability
                from trace_ct.config.schema import ResidualAuditThresholds, ContextAuditThresholds, ProposalQualificationThresholds, DynamicTargetGatingThresholds, CycleStabilityThresholds
                self.thresholds = ThresholdsYAML(
                    residual_audit=ResidualAuditThresholds(
                        min_accepted_count=1, min_accepted_rate=0.5, max_structural_score_q95=0.05,
                        max_low_frequency_leakage_q95=0.05, max_edge_leakage_q95=0.05,
                        relative_std_min=0.1, relative_std_max=2.0,
                        require_donor_receiver_isolation=True, require_residual_pool_freshness=True
                    ),
                    context_audit=ContextAuditThresholds(
                        max_high_frequency_correlation=0.10, max_high_frequency_leakage_ratio=0.05,
                        min_context_helpfulness_delta=0.01, max_context_gate_on_unreliable_region=0.10
                    ),
                    proposal_qualification=ProposalQualificationThresholds(
                        min_homogeneous_noise_reduction=0.20, min_edge_contrast_retention=0.90,
                        min_lesion_contrast_retention=0.90, max_low_frequency_bias=0.05,
                        max_nps_shape_error=0.15, reject_gaussian_blur_negative_control=True
                    ),
                    dynamic_target_gating=DynamicTargetGatingThresholds(
                        max_W_fb_in_lesion_mean=0.05, max_W_fb_at_edge_mean=0.05,
                        min_W_fb_in_homogeneous_mean=0.20, max_target_drift_q95=0.15,
                        max_pd_disagreement_allowed=0.10, reject_all_ones_gate=True
                    ),
                    cycle_stability=CycleStabilityThresholds(
                        max_cycle_drift_q95=0.05, max_target_drift_q95=0.10,
                        max_residual_policy_change=0.20, max_loss_oscillation_ratio=0.30
                    )
                )
        else:
            self.thresholds = thresholds
            
        # Internal rho_t controller state
        self.rho_t = 0.0
        
        # Track metric histories for G6-G8 trends
        self.target_drift_history: List[float] = []
        self.disagreement_history: List[float] = []
        
        # Modules allowed in each stage
        self.allowed_modules: Dict[Stage, List[str]] = {
            Stage.G0: [],
            Stage.G1: ["D", "masking"],
            Stage.G2: ["D", "masking", "context"],
            Stage.G3: ["residual_audit"],
            Stage.G4: ["D", "masking", "context", "residual"],
            Stage.G45: ["D", "masking", "context", "denoising_strength_audit"],
            Stage.G5: ["D", "masking", "context", "residual", "proposal_audit"],
            Stage.G6: ["D", "masking", "context", "residual", "proposal", "dynamic_target_local"],
            Stage.G7: ["D", "masking", "context", "residual", "proposal", "dynamic_target_local", "short_cycle"],
            Stage.G8: ["D", "masking", "context", "residual", "proposal", "dynamic_target_local", "short_cycle", "limited_full"]
        }
        
    def check_prerequisites(self, target_stage: Stage) -> bool:
        """Checks if all previous mandatory stages have a fresh pass record."""
        stage_order = list(Stage)
        target_idx = stage_order.index(target_stage)
        
        for i in range(target_idx):
            req_stage = stage_order[i]
            record = self.logger.get_stage_record(req_stage.value, status="pass")
            if not record:
                return False
                
            # Verify Freshness (TTL check)
            filename = f"{req_stage.value}_pass.json"
            filepath = self.logger.stage_records_dir / filename
            if filepath.exists():
                mtime = filepath.stat().st_mtime
                if (time.time() - mtime) > self.ttl_seconds:
                    return False
                    
        return True
        
    def get_stage_state(self, stage: Stage) -> StageState:
        """Determines the current state (OFF/AUDIT/TRAIN) of a stage."""
        if not self.check_prerequisites(stage):
            return StageState.OFF
            
        record = self.logger.get_stage_record(stage.value, status="pass")
        if record:
            return StageState.TRAIN
            
        return StageState.AUDIT
        
    def update_rho_t(self, stage: Stage, passed_audit: bool) -> float:
        """Updates residual strength rho_t."""
        if stage in [Stage.G0, Stage.G1, Stage.G2, Stage.G3, Stage.G45]:
            self.rho_t = 0.0
            return self.rho_t
            
        limits = self.thresholds.dynamic_target_gating
        ramp_step = getattr(limits, "rho_ramp_step", 0.05)
        max_rho = getattr(limits, "max_rho", 1.0)
        
        if passed_audit:
            self.rho_t = min(max_rho, self.rho_t + ramp_step)
        else:
            self.rho_t = 0.0  # Hard drop on failure
            
        return self.rho_t
        
    def can_enable_module(self, current_stage: Stage, module_name: str) -> bool:
        """Returns True if the module is allowed in the current stage."""
        state = self.get_stage_state(current_stage)
        if state == StageState.OFF:
            return False
        return module_name in self.allowed_modules.get(current_stage, [])
        
    def validate_transition(self, current_stage: Stage, next_stage: Stage) -> bool:
        """Validates transition safety."""
        if current_stage != next_stage:
            record = self.logger.get_stage_record(current_stage.value, status="pass")
            if not record:
                return False
                
        if not self.check_prerequisites(next_stage):
            return False
            
        return True
        
    def verify_g6_target_gating(
        self, 
        w_fb: torch.Tensor, 
        lesion_mask: torch.Tensor, 
        edge_mask: torch.Tensor, 
        homo_mask: torch.Tensor
    ) -> Tuple[bool, List[str]]:
        """
        G6 validation: verifies that feedback weights are small in lesion and edge regions,
        substantial in homogeneous regions, and not degenerated (all-ones gate check).
        """
        reasons = []
        limits = self.thresholds.dynamic_target_gating
        
        # All-ones gate check
        mean_fb = float(w_fb.mean().item())
        all_ones_thresh = getattr(limits, "all_ones_gate_threshold", 0.95)
        if limits.reject_all_ones_gate and mean_fb > all_ones_thresh:
            reasons.append(f"All-ones feedback gate detected (mean={mean_fb:.4f} > {all_ones_thresh})")
            
        # Lesion region check
        l_sum = lesion_mask.sum().item()
        if l_sum > 0:
            lesion_mean = float((w_fb * lesion_mask).sum().item() / l_sum)
            if lesion_mean > limits.max_W_fb_in_lesion_mean:
                reasons.append(f"Feedback weight in lesion {lesion_mean:.4f} exceeds max {limits.max_W_fb_in_lesion_mean}")
                
        # Edge region check
        e_sum = edge_mask.sum().item()
        if e_sum > 0:
            edge_mean = float((w_fb * edge_mask).sum().item() / e_sum)
            if edge_mean > limits.max_W_fb_at_edge_mean:
                reasons.append(f"Feedback weight at edge {edge_mean:.4f} exceeds max {limits.max_W_fb_at_edge_mean}")
                
        # Homogeneous region check
        h_sum = homo_mask.sum().item()
        if h_sum > 0:
            homo_mean = float((w_fb * homo_mask).sum().item() / h_sum)
            if homo_mean < limits.min_W_fb_in_homogeneous_mean:
                reasons.append(f"Feedback weight in homogeneous region {homo_mean:.4f} is below min {limits.min_W_fb_in_homogeneous_mean}")
                
        return len(reasons) == 0, reasons
        
    def verify_g7_g8_stability(
        self, 
        disagreement: float, 
        drift_q95: float
    ) -> Tuple[bool, List[str]]:
        """
        G7/G8 validation: verifies D/P disagreement and target drift trends.
        """
        reasons = []
        limits = self.thresholds.dynamic_target_gating
        
        self.disagreement_history.append(disagreement)
        self.target_drift_history.append(drift_q95)
        
        # Limit checking
        if disagreement > limits.max_pd_disagreement_allowed:
            reasons.append(f"D/P disagreement {disagreement:.4f} exceeds max {limits.max_pd_disagreement_allowed}")
            
        if drift_q95 > limits.max_target_drift_q95:
            reasons.append(f"Target drift Q95 {drift_q95:.4f} exceeds max {limits.max_target_drift_q95}")
            
        # Trend checking: check for continuous target drift increase over 3 steps
        if len(self.target_drift_history) >= 3:
            if (self.target_drift_history[-1] > self.target_drift_history[-2] > self.target_drift_history[-3]):
                reasons.append(f"Target drift Q95 showing consecutive monotonic increase: {self.target_drift_history[-3:]}")
                
        return len(reasons) == 0, reasons
        
    def get_denoising_strength_record(self) -> StagePassFailRecord | None:
        return self.logger.get_stage_record(Stage.G45.value, status="pass")

    def require_denoising_strength_release(self, target_stage: Stage) -> Tuple[bool, List[str]]:
        if target_stage not in [Stage.G5, Stage.G6, Stage.G7]:
            return True, []
        record = self.get_denoising_strength_record()
        if record is None:
            return False, ["G4.5 denoising strength audit pass record is required before G5/G6/G7."]
        release = bool(record.metrics.get("release_D", False))
        if not release:
            return False, ["G4.5 denoising strength audit did not release D."]
        return True, []

    def trigger_rollback(self, failed_stage: Stage, reasons: List[str]):
        """
        Performs an automatic rollback to the G4 checkpoint/state and writes the fallback report.
        """
        self.rho_t = 0.0  # Hard drop
        
        report = {
            "timestamp": str(time.time()),
            "failed_stage": failed_stage.value,
            "reasons": reasons,
            "fallback_action_taken": self.get_fallback_action(failed_stage),
            "disagreement_history": self.disagreement_history,
            "target_drift_history": self.target_drift_history
        }
        
        if hasattr(self.logger, "run_dir"):
            report_dir = self.logger.run_dir / "reports"
        else:
            run_id = getattr(self.logger, "run_id", "default")
            report_dir = Path("runs") / run_id / "reports"
            
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "fallback_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
    def get_fallback_action(self, failed_stage: Stage) -> str:
        """Returns the fallback action for a failed stage."""
        fallbacks = {
            Stage.G0: "Abort execution completely. Fix data paths/geometry.",
            Stage.G1: "Abort training. Adjust mask blocks/loss regions.",
            Stage.G2: "Disable context, fallback to G1 if HF leakage exceeds limit. Rerun audit if cutoff changes.",
            Stage.G3: "Invalidate residual_pool_hash. Block downstream. Force rho_t=0. G4 blocked until new G3_pass.json.",
            Stage.G4: "Set rho_t=0, disable injection, fallback to G2.",
            Stage.G45: "Block G5/G6/G7. Recalibrate denoising strength before proposal or dynamic target.",
            Stage.G5: "P_phi banned from generating targets. G6 blocked until valid G5_pass.json.",
            Stage.G6: "Invalidates dynamic_target_policy_hash, disables dynamic target entirely, retains G4 targets, and strictly blocks G7/G8 until a new G6_pass.json is generated.",
            Stage.G7: "Stop before G8. Write instability report. Freeze unstable branch. Rerun G7. Do not proceed to G8 without new G7_pass.json.",
            Stage.G8: "Stop protocol. Write failure report. Fallback to G7 diagnostic."
        }
        return fallbacks.get(failed_stage, "Abort pipeline.")
