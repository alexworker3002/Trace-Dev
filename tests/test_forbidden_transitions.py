import pytest
import os
import shutil
import json
from pathlib import Path

from trace_ct.training.state_machine import TraceCTStateMachine, Stage
from trace_ct.audit.logger import AuditLogger
from trace_ct.audit.schemas import StagePassFailRecord, GlobalHashes, ArchitectureHashes, StageHashes

@pytest.fixture
def run_dir(tmp_path):
    run_id = "test_run_transitions"
    # Create the run dir manually for test to avoid hardcoding path inside logger
    audit_dir = tmp_path / "runs" / run_id / "audit" / "stage_records"
    audit_dir.mkdir(parents=True)
    
    # We patch the logger's directory for tests
    class MockLogger(AuditLogger):
        def __init__(self):
            self.run_id = run_id
            self.stage_records_dir = audit_dir
            
    return MockLogger()

def test_g4_requires_g3_pass(run_dir):
    machine = TraceCTStateMachine(run_dir)
    
    # G0-G2 passes
    global_hashes = GlobalHashes(dataset_config_hash="a", normalization_config_hash="b", split_config_hash="c", code_hash="d")
    arch_hashes = ArchitectureHashes()
    stage_hashes = StageHashes(policy_hash="p")
    
    for stage in [Stage.G0, Stage.G1, Stage.G2]:
        record = StagePassFailRecord(
            run_id=run_dir.run_id, stage=stage.value, status="pass", timestamp="123",
            global_hashes=global_hashes, architecture_hashes=arch_hashes, stage_hashes=stage_hashes
        )
        run_dir.log_stage_record(record)
        
    # G3 is missing
    assert not machine.check_prerequisites(Stage.G4)
    
    # G3 failed
    record = StagePassFailRecord(
        run_id=run_dir.run_id, stage=Stage.G3.value, status="fail", timestamp="123",
        global_hashes=global_hashes, architecture_hashes=arch_hashes, stage_hashes=stage_hashes,
        failure_reasons=["Residual pool contaminated"]
    )
    run_dir.log_stage_record(record)
    assert not machine.check_prerequisites(Stage.G4)
    
    # G3 passes
    record.status = "pass"
    run_dir.log_stage_record(record)
    assert machine.check_prerequisites(Stage.G4)

def test_module_isolation(run_dir):
    machine = TraceCTStateMachine(run_dir)
    
    # Mock G0 pass so Stage.G1 is AUDIT or TRAIN instead of OFF
    global_hashes = GlobalHashes(dataset_config_hash="a", normalization_config_hash="b", split_config_hash="c", code_hash="d")
    arch_hashes = ArchitectureHashes()
    stage_hashes = StageHashes(policy_hash="p")
    
    record = StagePassFailRecord(
        run_id=run_dir.run_id, stage=Stage.G0.value, status="pass", timestamp="123",
        global_hashes=global_hashes, architecture_hashes=arch_hashes, stage_hashes=stage_hashes
    )
    run_dir.log_stage_record(record)
    
    # In G1, only D and masking are allowed
    assert machine.can_enable_module(Stage.G1, "D")
    assert machine.can_enable_module(Stage.G1, "masking")
    assert not machine.can_enable_module(Stage.G1, "context")
    assert not machine.can_enable_module(Stage.G1, "residual")
    
    # Mock G0-G3 passes to check G4 residual enablement
    for stage in [Stage.G1, Stage.G2, Stage.G3]:
        r = StagePassFailRecord(
            run_id=run_dir.run_id, stage=stage.value, status="pass", timestamp="123",
            global_hashes=global_hashes, architecture_hashes=arch_hashes, stage_hashes=stage_hashes
        )
        run_dir.log_stage_record(r)
        
    # In G4, residual is allowed, but dynamic target is not
    assert machine.can_enable_module(Stage.G4, "residual")
    assert not machine.can_enable_module(Stage.G4, "dynamic_target_local")
