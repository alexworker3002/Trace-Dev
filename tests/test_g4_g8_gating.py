import pytest
import time
from trace_ct.training.stages import G4BaselineProposals, G5ProposalQualification, G6DynamicTargetGating, G7EndToEndSelfSupervised, G8CycleStability
from trace_ct.training.state_machine import TraceCTStateMachine, Stage
from trace_ct.audit.logger import AuditLogger
from trace_ct.audit.schemas import StagePassFailRecord, GlobalHashes, ArchitectureHashes, StageHashes

@pytest.fixture
def state_machine(tmp_path):
    run_id = "test_g4_g8"
    audit_dir = tmp_path / "runs" / run_id / "audit" / "stage_records"
    audit_dir.mkdir(parents=True)
    
    class MockLogger(AuditLogger):
        def __init__(self):
            self.run_id = run_id
            self.stage_records_dir = audit_dir
            
    logger = MockLogger()
    sm = TraceCTStateMachine(logger)
    return sm

def test_g4_blocked_without_g3(state_machine):
    with pytest.raises(RuntimeError, match="G4 blocked"):
        G4BaselineProposals(state_machine)

def test_g4_g8_progression(state_machine):
    """Test full sequential progression G0 -> G8"""
    stages = [Stage.G0, Stage.G1, Stage.G2, Stage.G3, Stage.G4, Stage.G45, Stage.G5, Stage.G6, Stage.G7, Stage.G8]
    classes = [None, None, None, None, G4BaselineProposals, None, G5ProposalQualification, G6DynamicTargetGating, G7EndToEndSelfSupervised, G8CycleStability]
    
    for i, s in enumerate(stages):
        # Now it should be allowed if i >= 4
        if i >= 4 and classes[i] is not None:
            obj = classes[i](state_machine)
            assert obj is not None
            
        # Log pass for this stage so the next one unblocks
        record = StagePassFailRecord(
            run_id=state_machine.logger.run_id, stage=s.value, status="pass", timestamp=str(time.time()),
            global_hashes=GlobalHashes(dataset_config_hash="a", normalization_config_hash="b", split_config_hash="c", code_hash="d"),
            architecture_hashes=ArchitectureHashes(),
            stage_hashes=StageHashes(policy_hash="p"),
            metrics={"release_D": 1.0} if s == Stage.G45 else {}
        )
        state_machine.logger.log_stage_record(record)



def test_g5_blocked_without_g45_release(state_machine):
    for s in [Stage.G0, Stage.G1, Stage.G2, Stage.G3, Stage.G4]:
        record = StagePassFailRecord(
            run_id=state_machine.logger.run_id, stage=s.value, status="pass", timestamp=str(time.time()),
            global_hashes=GlobalHashes(dataset_config_hash="a", normalization_config_hash="b", split_config_hash="c", code_hash="d"),
            architecture_hashes=ArchitectureHashes(),
            stage_hashes=StageHashes(policy_hash="p")
        )
        state_machine.logger.log_stage_record(record)
    with pytest.raises(RuntimeError, match="G5 blocked"):
        G5ProposalQualification(state_machine)
