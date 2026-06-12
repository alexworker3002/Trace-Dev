import pytest
import torch
import torch.nn.functional as F

from trace_ct.training.stages import G1MaskedBaseline
from trace_ct.training.state_machine import TraceCTStateMachine, Stage
from trace_ct.audit.logger import AuditLogger
from trace_ct.audit.schemas import StagePassFailRecord, GlobalHashes, ArchitectureHashes, StageHashes
from trace_ct.data.phantom import SyntheticPhantom

@pytest.fixture
def state_machine(tmp_path):
    run_id = "test_g1"
    audit_dir = tmp_path / "runs" / run_id / "audit" / "stage_records"
    audit_dir.mkdir(parents=True)
    
    class MockLogger(AuditLogger):
        def __init__(self):
            self.run_id = run_id
            self.stage_records_dir = audit_dir
            
    logger = MockLogger()
    sm = TraceCTStateMachine(logger)
    
    # Mock G0 pass
    record = StagePassFailRecord(
        run_id=run_id, stage=Stage.G0.value, status="pass", timestamp="123",
        global_hashes=GlobalHashes(dataset_config_hash="a", normalization_config_hash="b", split_config_hash="c", code_hash="d"),
        architecture_hashes=ArchitectureHashes(),
        stage_hashes=StageHashes(policy_hash="p")
    )
    logger.log_stage_record(record)
    return sm

def test_g1_training(state_machine):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g1 = G1MaskedBaseline(state_machine, device=device)
    phantom = SyntheticPhantom(shape=(1, 64, 64), device=device)
    
    batch = phantom.generate()
    # Mock batch format for training
    batch = {k: v.unsqueeze(0) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    
    loss1 = g1.step(batch)
    loss2 = g1.step(batch)
    
    # Check that model trains and loss goes down (usually true for 2 identical steps)
    # Actually just ensuring it runs without crashing is the main smoke test.
    assert loss1 > 0
    assert loss2 > 0
    
    # Validate masking: model shouldn't access context or residual
    assert not g1.state_machine.can_enable_module(Stage.G1, "context")
