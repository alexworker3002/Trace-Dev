import torch
from typing import Dict, Any, List

class ResidualController:
    """
    Manages residual injection parameter state and gating (G4).
    Tracks: rho_t, alpha_t, pi_pass, E_acc, train_infer_shift, and pool freshness.
    """
    def __init__(self, initial_alpha: float = 0.5, ramp_step: float = 0.05, max_rho: float = 1.0):
        self.rho_t = 0.0
        self.alpha_t = initial_alpha
        self.ramp_step = ramp_step
        self.max_rho = max_rho
        
        # Validation statistics
        self.pi_pass = 1.0 # Pass rate of audits
        self.E_acc = 0     # Accumulated audit failures
        self.train_infer_shift = 0.0
        
        self.failures_history: List[str] = []
        
    def step_ramp(self, passed_audit: bool) -> float:
        """
        Updates rho_t based on audit outcome.
        Ramps up if passed, instantly drops to 0.0 if failed.
        """
        if passed_audit:
            self.rho_t = min(self.max_rho, self.rho_t + self.ramp_step)
        else:
            self.rho_t = 0.0
            self.E_acc += 1
            self.failures_history.append(f"Audit failed at rho_t={self.rho_t:.3f}")
            
        return self.rho_t
        
    def record_validation_stats(self, pass_count: int, total_count: int, train_loss: float, infer_loss: float):
        """
        Records verification metrics and calculates Train-Infer Shift.
        """
        if total_count > 0:
            self.pi_pass = float(pass_count / total_count)
        else:
            self.pi_pass = 1.0
            
        self.train_infer_shift = float(abs(train_loss - infer_loss))
        
    def verify_isolation(self, incoming_volume_id: str, donor_volume_ids: List[str]) -> bool:
        """
        Enforces donor/receiver isolation in loader loops.
        """
        return incoming_volume_id in donor_volume_ids
