from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional

class GlobalHashes(BaseModel):
    dataset_config_hash: str
    normalization_config_hash: str
    split_config_hash: str
    code_hash: str

class ArchitectureHashes(BaseModel):
    denoiser_architecture_hash: Optional[str] = None
    context_architecture_hash: Optional[str] = None
    proposal_architecture_hash: Optional[str] = None

class StageHashes(BaseModel):
    policy_hash: str
    training_hash: Optional[str] = None
    output_hashes: Dict[str, str] = Field(default_factory=dict)

class StagePassFailRecord(BaseModel):
    run_id: str
    stage: str
    status: Literal["pass", "fail"]
    timestamp: str
    global_hashes: GlobalHashes
    architecture_hashes: ArchitectureHashes
    stage_hashes: StageHashes
    metrics: Dict[str, float] = Field(default_factory=dict)
    thresholds: Dict[str, float] = Field(default_factory=dict)
    failure_reasons: List[str] = Field(default_factory=list)
    fallback_action_taken: Optional[str] = None
    next_allowed_stage: Optional[str] = None

class HRAccessViolationLog(BaseModel):
    caller: str
    mode: str
    path: str
    action: str
    exception: str
    timestamp: str
