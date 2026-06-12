import pytest
import torch
import torch.nn.functional as F

from trace_ct.training.stages import G2ContextGating
from trace_ct.training.state_machine import TraceCTStateMachine, Stage
from trace_ct.audit.logger import AuditLogger
from trace_ct.audit.schemas import StagePassFailRecord, GlobalHashes, ArchitectureHashes, StageHashes
from trace_ct.data.phantom import SyntheticPhantom

@pytest.fixture
def state_machine(tmp_path):
    run_id = "test_g2"
    audit_dir = tmp_path / "runs" / run_id / "audit" / "stage_records"
    audit_dir.mkdir(parents=True)
    
    class MockLogger(AuditLogger):
        def __init__(self):
            self.run_id = run_id
            self.stage_records_dir = audit_dir
            
    logger = MockLogger()
    sm = TraceCTStateMachine(logger)
    
    # Mock G0 and G1 passes
    for s in [Stage.G0, Stage.G1]:
        record = StagePassFailRecord(
            run_id=run_id, stage=s.value, status="pass", timestamp="123",
            global_hashes=GlobalHashes(dataset_config_hash="a", normalization_config_hash="b", split_config_hash="c", code_hash="d"),
            architecture_hashes=ArchitectureHashes(),
            stage_hashes=StageHashes(policy_hash="p")
        )
        logger.log_stage_record(record)
        
    return sm

def test_g2_context_leakage(state_machine):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g2 = G2ContextGating(state_machine, device=device)
    phantom = SyntheticPhantom(shape=(1, 64, 64), device=device)
    
    # Generate phantom where adjacent slice has HF noise
    batch = phantom.generate()
    batch = {k: v.unsqueeze(0) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    
    # Step trains the network
    loss = g2.step(batch)
    assert loss > 0
    
    # HF Leakage Check:
    # Context encoder shouldn't pass HF features.
    # We can test this by passing pure HF noise to the context encoder and verifying the output variance is small.
    pure_noise = torch.randn(1, 1, 64, 64, device=device)
    g2.context_encoder.eval()
    with torch.no_grad():
        out_noise = g2.context_encoder(pure_noise)
        
    # The output variance should be much lower than the input variance due to the stride=2 (low pass)
    assert out_noise.std() < pure_noise.std()
