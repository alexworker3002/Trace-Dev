import os
from pathlib import Path

def get_run_dir(run_id: str) -> Path:
    """Gets the base directory for a specific run."""
    run_dir = Path("runs") / run_id
    return run_dir

def setup_run_directories(run_id: str) -> Path:
    """
    Sets up the mandatory directory structure for a TRACE-CT run.
    Ensures that audit, checkpoints, reports, and residual_pools directories exist.
    """
    run_dir = get_run_dir(run_id)
    
    dirs_to_create = [
        run_dir / "audit",
        run_dir / "audit" / "stage_records",
        run_dir / "audit" / "security",
        run_dir / "checkpoints",
        run_dir / "reports",
        run_dir / "residual_pools"
    ]
    
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        
    return run_dir
