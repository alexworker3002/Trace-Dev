import json
from pathlib import Path
from trace_ct.utils.paths import get_run_dir
from trace_ct.audit.schemas import StagePassFailRecord, HRAccessViolationLog

class AuditLogger:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.run_dir = get_run_dir(run_id)
        self.stage_records_dir = self.run_dir / "audit" / "stage_records"
        self.security_dir = self.run_dir / "audit" / "security"
        
    def log_stage_record(self, record: StagePassFailRecord):
        """Logs a stage pass or fail record as a JSON file."""
        filename = f"{record.stage}_{record.status}.json"
        filepath = self.stage_records_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(record.model_dump(), f, indent=2)
            
    def log_hr_access_violation(self, violation: HRAccessViolationLog):
        """Appends an HR access violation to the security log."""
        filepath = self.security_dir / "hr_access_violations.jsonl"
        with open(filepath, 'a') as f:
            f.write(json.dumps(violation.model_dump()) + "\n")
            
    def get_stage_record(self, stage: str, status: str = "pass") -> StagePassFailRecord | None:
        """Retrieves a stage record if it exists."""
        filename = f"{stage}_{status}.json"
        filepath = self.stage_records_dir / filename
        if not filepath.exists():
            return None
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        return StagePassFailRecord(**data)
