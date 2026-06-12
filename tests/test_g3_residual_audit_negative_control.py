import pytest
import torch
import time
from pathlib import Path

from trace_ct.audit.residual_audit import ResidualAuditor, ResidualPool
from trace_ct.config.schema import ResidualAuditThresholds
from trace_ct.data.phantom import SyntheticPhantom
from trace_ct.training.state_machine import TraceCTStateMachine, Stage
from trace_ct.audit.logger import AuditLogger
from trace_ct.audit.schemas import StagePassFailRecord, GlobalHashes, ArchitectureHashes, StageHashes

@pytest.fixture
def state_machine_and_run_dir(tmp_path):
    run_id = "test_g3"
    run_dir = tmp_path / "runs" / run_id
    audit_dir = run_dir / "audit" / "stage_records"
    security_dir = run_dir / "audit" / "security"
    pool_dir = run_dir / "residual_pools"
    
    audit_dir.mkdir(parents=True)
    security_dir.mkdir(parents=True)
    pool_dir.mkdir(parents=True)
    
    class MockLogger(AuditLogger):
        def __init__(self):
            self.run_id = run_id
            self.stage_records_dir = audit_dir
            self.run_dir = run_dir
            self.security_dir = security_dir
            self.accepted_pool_path = pool_dir / "accepted_residuals.pt"
            
    logger = MockLogger()
    sm = TraceCTStateMachine(logger)
    
    # Mock G0, G1, G2 passes
    for s in [Stage.G0, Stage.G1, Stage.G2]:
        record = StagePassFailRecord(
            run_id=run_id, stage=s.value, status="pass", timestamp=str(time.time()),
            global_hashes=GlobalHashes(dataset_config_hash="a", normalization_config_hash="b", split_config_hash="c", code_hash="d"),
            architecture_hashes=ArchitectureHashes(),
            stage_hashes=StageHashes(policy_hash="p")
        )
        logger.log_stage_record(record)
        
    return sm, run_dir

def test_g3_residual_audit(state_machine_and_run_dir):
    """Level 1: G3 Residual Audit test ensuring contaminated residuals are rejected."""
    state_machine, run_dir = state_machine_and_run_dir
    assert state_machine.check_prerequisites(Stage.G3)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    phantom = SyntheticPhantom(shape=(1, 64, 64), device=device)
    data = phantom.generate()
    nc = data["negative_controls"]
    
    thresholds = ResidualAuditThresholds(
        min_accepted_count=1,
        min_accepted_rate=0.5,
        max_structural_score_q95=0.05,
        max_low_frequency_leakage_q95=0.05,
        max_edge_leakage_q95=0.05,
        relative_std_min=0.1,
        relative_std_max=2.0,
        require_donor_receiver_isolation=True,
        require_residual_pool_freshness=True
    )
    
    auditor = ResidualAuditor(thresholds)
    
    # 1. Test Clean/Proper Residual (pure noise, reasonable std, zero mean)
    clean_residual = torch.randn(1, 64, 64, device=device) * 0.1
    passed, reasons, metrics = auditor.audit_patch(clean_residual, data["edge_mask"], data["clean"])
    assert passed, f"Clean residual failed: {reasons}"
    
    # 2. Test High STD Failure
    high_std_residual = torch.randn(1, 64, 64, device=device) * 5.0
    passed, reasons, metrics = auditor.audit_patch(high_std_residual, data["edge_mask"], data["clean"])
    assert not passed, "High std residual should have failed"
    assert any("std" in r or "Relative std" in r for r in reasons)
    
    # 3. Test Structural Contamination Failure (edge leakage)
    struct_residual = nc["structure_contaminated_residual"]
    passed, reasons, metrics = auditor.audit_patch(struct_residual, data["edge_mask"], data["clean"])
    assert not passed, "Structure contaminated residual should have failed"
    assert any("Edge leakage" in r for r in reasons)
    
    # 4. Test LF Contamination Failure
    lf_residual = nc["lf_contaminated_residual"]
    passed, reasons, metrics = auditor.audit_patch(lf_residual, data["edge_mask"], data["clean"])
    assert not passed, "LF contaminated residual should have failed"
    assert any("low-frequency variance" in r or "Low-frequency variance" in r for r in reasons)
    
    # 5. Test Mean Shift Failure
    shifted_residual = torch.randn(1, 64, 64, device=device) * 0.1 + 0.5
    passed, reasons, metrics = auditor.audit_patch(shifted_residual, data["edge_mask"], data["clean"])
    assert not passed, "Shifted mean residual should have failed"
    assert any("Mean shift" in r or "mean shift" in r for r in reasons)

def test_g3_residual_pool(state_machine_and_run_dir):
    """Verifies that ResidualPool registers and filters residuals based on donor isolation."""
    state_machine, run_dir = state_machine_and_run_dir
    
    thresholds = ResidualAuditThresholds(
        min_accepted_count=2,
        min_accepted_rate=0.5,
        max_structural_score_q95=0.05,
        max_low_frequency_leakage_q95=0.5,
        max_edge_leakage_q95=0.5,
        relative_std_min=0.05,
        relative_std_max=2.0,
        require_donor_receiver_isolation=True,
        require_residual_pool_freshness=True
    )
    
    pool = ResidualPool(
        run_dir=run_dir,
        thresholds=thresholds,
        donor_volume_ids=["vol_donor_1"]
    )
    
    # Generate mock residual volume [1, 64, 64]
    residuals = torch.randn(1, 64, 64) * 0.1
    edge_masks = torch.zeros(1, 64, 64)
    # Give clean_baselines a non-zero std
    clean_baselines = torch.randn(1, 64, 64) * 0.1
    
    # Adding residual from a non-donor should fail if isolation is enabled
    with pytest.raises(ValueError, match="not in the donor volume list"):
        pool.add_volume_residuals(
            volume_id="vol_receiver_1",
            residuals=residuals,
            edge_masks=edge_masks,
            clean_baselines=clean_baselines
        )
        
    # Adding residual from donor should succeed
    stats = pool.add_volume_residuals(
        volume_id="vol_donor_1",
        residuals=residuals,
        edge_masks=edge_masks,
        clean_baselines=clean_baselines
    )
    
    assert stats["accepted"] > 0
    assert pool.accepted_pool_path.exists()
    
    loaded = pool.load_accepted_patches()
    assert loaded.shape[0] > 0
