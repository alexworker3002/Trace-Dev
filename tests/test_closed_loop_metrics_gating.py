import pytest
import torch
import tempfile
import json
import numpy as np
from pathlib import Path

from trace_ct.audit.context_audit import audit_g1_masked_baseline, audit_g2_context
from trace_ct.audit.proposal_audit import audit_proposal_generator
from trace_ct.audit.residual_audit import ResidualAuditor, ResidualPool, compute_nps_2d
from trace_ct.training.residual import ResidualController
from trace_ct.training.state_machine import TraceCTStateMachine, Stage
from trace_ct.audit.logger import AuditLogger
from trace_ct.config.schema import ResidualAuditThresholds

@pytest.fixture
def run_dir(tmp_path):
    r_dir = tmp_path / "runs" / "test_closed_loop"
    r_dir.mkdir(parents=True)
    return r_dir

@pytest.fixture
def state_machine(run_dir):
    class MockLogger(AuditLogger):
        def __init__(self):
            self.run_id = "test_closed_loop"
            self.run_dir = run_dir
            self.stage_records_dir = run_dir / "audit" / "stage_records"
            self.security_dir = run_dir / "audit" / "security"
            self.stage_records_dir.mkdir(parents=True, exist_ok=True)
            self.security_dir.mkdir(parents=True, exist_ok=True)
            
    logger = MockLogger()
    return TraceCTStateMachine(logger)

def test_g1_g2_audit_metrics():
    pred = torch.ones(1, 1, 16, 16)
    noisy = torch.ones(1, 1, 16, 16) * 1.5
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, 0:4, 0:4] = 1.0
    homo_mask = torch.zeros(1, 1, 16, 16)
    homo_mask[:, :, 8:12, 8:12] = 1.0
    
    # G1 Audit
    g1_metrics = audit_g1_masked_baseline(pred, noisy, mask, homo_mask)
    assert "copy_attack_correlation" in g1_metrics
    assert "masked_only_loss_ratio" in g1_metrics
    assert "homogeneous_noise_reduction" in g1_metrics
    
    # G2 Audit
    context_features = torch.randn(1, 16, 16, 16)
    adjacent = torch.randn(1, 1, 16, 16)
    g2_metrics = audit_g2_context(context_features, adjacent)
    assert "high_frequency_leakage_ratio" in g2_metrics
    assert "high_frequency_correlation" in g2_metrics

def test_g3_closed_loop_metadata_and_nps(run_dir):
    thresholds = ResidualAuditThresholds(
        min_accepted_count=1,
        min_accepted_rate=0.5,
        max_structural_score_q95=0.5,
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
        donor_volume_ids=["donor_1"],
        audit_version_hash="test_version_123"
    )
    
    residuals = torch.randn(1, 64, 64) * 0.1
    edge_masks = torch.zeros(1, 64, 64)
    clean_baselines = torch.randn(1, 64, 64) * 0.1
    
    stats = pool.add_volume_residuals(
        volume_id="donor_1",
        residuals=residuals,
        edge_masks=edge_masks,
        clean_baselines=clean_baselines
    )
    
    assert stats["accepted"] > 0
    assert pool.metadata_log_path.exists()
    
    # Verify metadata details are logged
    with open(pool.metadata_log_path, 'r') as f:
        meta = json.load(f)
        assert len(meta) > 0
        item = meta[0]
        assert item["donor_volume_id"] == "donor_1"
        assert "coordinates" in item
        assert item["audit_version_hash"] == "test_version_123"
        assert "nps_correlation" in item["metrics"]

def test_g4_residual_controller():
    controller = ResidualController(initial_alpha=0.5, ramp_step=0.05, max_rho=1.0)
    
    # Test ramp-up
    rho1 = controller.step_ramp(passed_audit=True)
    assert rho1 == 0.05
    
    # Test fallback to 0.0 on audit failure
    rho2 = controller.step_ramp(passed_audit=False)
    assert rho2 == 0.0
    assert controller.E_acc == 1
    
    # Test Train-Infer shift calculation
    controller.record_validation_stats(pass_count=8, total_count=10, train_loss=0.25, infer_loss=0.21)
    assert controller.pi_pass == 0.8
    assert controller.train_infer_shift == pytest.approx(0.04)

def test_g5_proposal_report():
    proposal = torch.randn(1, 1, 64, 64) * 0.1
    noisy = torch.randn(1, 1, 64, 64) * 0.5
    clean = torch.ones(1, 1, 64, 64) * 0.1
    homo_mask = torch.ones(1, 1, 64, 64)
    edge_mask = torch.zeros(1, 1, 64, 64)
    lesion_mask = torch.zeros(1, 1, 64, 64)
    
    report = audit_proposal_generator(proposal, noisy, clean, homo_mask, edge_mask, lesion_mask)
    assert "homogeneous_noise_reduction" in report
    
    # Test blur negative control detection
    report_blur = audit_proposal_generator(proposal, noisy, clean, homo_mask, edge_mask, lesion_mask, is_negative_control=True)
    assert not report_blur["passed"]

def test_g6_g8_fallbacks(state_machine, run_dir):
    # Mock validation values
    w_fb_all_ones = torch.ones(1, 1, 16, 16)
    lesion_mask = torch.zeros(1, 1, 16, 16)
    edge_mask = torch.zeros(1, 1, 16, 16)
    homo_mask = torch.zeros(1, 1, 16, 16)
    
    # 1. Test all-ones gate detection
    passed, reasons = state_machine.verify_g6_target_gating(w_fb_all_ones, lesion_mask, edge_mask, homo_mask)
    assert not passed
    assert any("All-ones" in r for r in reasons)
    
    # 2. Test auto-rollback JSON fallback report output
    state_machine.trigger_rollback(Stage.G6, reasons)
    report_path = run_dir / "reports" / "fallback_report.json"
    assert report_path.exists()
    
    with open(report_path, 'r') as f:
        rep = json.load(f)
        assert rep["failed_stage"] == "G6"
        assert "All-ones" in rep["reasons"][0]
        
    # 3. Test G7/G8 target drift trend check (continuous increase)
    passed1, _ = state_machine.verify_g7_g8_stability(disagreement=0.02, drift_q95=0.05)
    passed2, _ = state_machine.verify_g7_g8_stability(disagreement=0.03, drift_q95=0.08)
    passed3, reasons_drift = state_machine.verify_g7_g8_stability(disagreement=0.04, drift_q95=0.12)
    
    # Should flag consecutive monotonic increase
    assert not passed3
    assert any("monotonic increase" in r for r in reasons_drift)

def test_g0_hashes_and_chunk_checks():
    from trace_ct.utils.hashing import compute_metadata_hash, compute_sample_hash
    # Test metadata hash
    meta = {"volume_id": "test_01", "spacing": [1.0, 1.0, 1.0]}
    h_meta = compute_metadata_hash(meta)
    assert len(h_meta) == 64
    
    # Test sample hash
    vol = np.zeros((5, 16, 16))
    vol[2, 8, 8] = 42.0
    h_sample = compute_sample_hash(vol)
    assert len(h_sample) == 64
    
    # Test MockArray chunk compatibility checks
    from trace_ct.data.zarr_io import get_zarr_array
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        zarr_json = tmp_path / "zarr.json"
        with open(zarr_json, 'w') as f:
            json.dump({
                "shape": [5, 160, 160],
                "chunk_grid": {
                    "configuration": {
                        "chunk_shape": [1, 32, 32]
                    }
                }
            }, f)
            
        arr = get_zarr_array(tmp_path, "")
        assert arr.shape == (5, 160, 160)
        assert arr.chunks == (1, 32, 32)

def test_swd_metric_calculation():
    torch.manual_seed(0)
    from trace_ct.audit.residual_audit import compute_swd_2d
    p1 = torch.ones(1, 16, 16) * 0.5
    p2 = torch.ones(1, 16, 16) * 0.5
    # SWD of identical patches should be very close to 0
    swd_same = compute_swd_2d(p1, p2)
    assert swd_same == pytest.approx(0.0, abs=1e-5)
    
    p3 = torch.randn(1, 16, 16) * 0.1
    p4 = torch.randn(1, 16, 16) * 0.1 + 0.5
    swd_diff = compute_swd_2d(p3, p4)
    assert swd_diff > 0.3

def test_g4_ah_modulation_and_isolation(state_machine, tmp_path):
    from trace_ct.training.stages import G4BaselineProposals
    from trace_ct.audit.schemas import GlobalHashes, ArchitectureHashes, StageHashes, StagePassFailRecord
    import time
    
    # Mock prerequisites so G4 constructor doesn't raise error
    global_hashes = GlobalHashes(dataset_config_hash="a", normalization_config_hash="b", split_config_hash="c", code_hash="d")
    arch_hashes = ArchitectureHashes()
    stage_hashes = StageHashes(policy_hash="p")
    for s in [Stage.G0, Stage.G1, Stage.G2, Stage.G3]:
        record = StagePassFailRecord(
            run_id=state_machine.logger.run_id, stage=s.value, status="pass", timestamp=str(time.time()),
            global_hashes=global_hashes, architecture_hashes=arch_hashes, stage_hashes=stage_hashes
        )
        state_machine.logger.log_stage_record(record)
        
    # Instantiate G4 with donor isolation list
    g4_stage = G4BaselineProposals(state_machine, donor_volume_ids=["donor_volume_1"])
    
    # Verify that a training batch with volume_id in donor split raises ValueError
    violating_batch = {
        "noisy": torch.randn(2, 1, 16, 16),
        "adjacent_noisy": torch.randn(2, 1, 16, 16),
        "volume_id": ["donor_volume_1", "donor_volume_1"]
    }
    with pytest.raises(ValueError, match="HR Access Violation"):
        g4_stage.step(violating_batch)
        
    # Verify that security log exists
    violation_file = state_machine.logger.security_dir / "hr_access_violations.jsonl"
    assert violation_file.exists()
    
    # Verify G4 A_h modulation:
    # Set up accepted residual pool
    pool_dir = state_machine.logger.run_dir / "residual_pools"
    pool_dir.mkdir(parents=True, exist_ok=True)
    residuals = torch.randn(5, 1, 16, 16) * 0.1
    torch.save(residuals, pool_dir / "accepted_residuals.pt")
    
    # Set rho_t > 0
    g4_stage.residual_controller.rho_t = 0.5
    g4_stage.residual_controller.alpha_t = 0.8
    
    # Step with safe batch (receiver volume)
    safe_batch = {
        "noisy": torch.zeros(2, 1, 16, 16),
        "adjacent_noisy": torch.zeros(2, 1, 16, 16),
        "volume_id": ["receiver_volume_2", "receiver_volume_2"]
    }
    
    # Let's run a forward pass
    loss = g4_stage.step(safe_batch)
    assert loss > 0.0
