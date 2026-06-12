import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

def compute_sha256(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def compute_dict_hash(d: Dict[str, Any]) -> str:
    """Computes a stable hash for a dictionary."""
    return compute_sha256(json.dumps(d, sort_keys=True))

def compute_file_hash(filepath: str | Path) -> str:
    """Computes SHA256 of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_code_hash(project_root: str | Path) -> str:
    """Computes the hash of the codebase, ignoring specific directories."""
    project_root = Path(project_root)
    exclude_dirs = {'runs', 'datasets', 'tests/_tmp', 'checkpoints', '__pycache__', '.pytest_cache', '.git', '.venv'}
    hasher = hashlib.sha256()
    
    # Collect all python and yaml files
    files = []
    for root, dirs, filenames in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.endswith('.egg-info')]
        for f in filenames:
            if f.endswith('.py') or f.endswith('.yaml'):
                files.append(Path(root) / f)
                
    # Sort for stability
    files.sort()
    
    for filepath in files:
        # Include the relative path and content in the hash
        rel_path = filepath.relative_to(project_root)
        hasher.update(str(rel_path).encode('utf-8'))
        hasher.update(compute_file_hash(filepath).encode('utf-8'))
        
    return hasher.hexdigest()

def compute_architecture_hash(model: Any) -> str:
    """Computes a stable hash of the model structure (layer names and parameter shapes)."""
    desc = []
    for name, param in model.named_parameters():
        desc.append(f"{name}:{list(param.shape)}")
    desc_str = ";".join(desc)
    return hashlib.sha256(desc_str.encode('utf-8')).hexdigest()

def compute_metadata_hash(metadata: Dict[str, Any]) -> str:
    """Computes a stable hash for metadata dictionary."""
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode('utf-8')).hexdigest()

def compute_sample_hash(volume: Any, root_path: Any = None) -> str:
    """Computes a stable hash of a deterministic voxel sample grid from a numpy volume, or fallback to raw chunk hashing for Zarr V3 MockArray."""
    import numpy as np
    
    if volume.__class__.__name__ == "MockArray" and root_path is not None:
        array_dir = Path(root_path)
        chunk_dir = array_dir / "c"
        if chunk_dir.exists():
            chunk_files = sorted(list(chunk_dir.rglob("*")))
            chunk_files = [f for f in chunk_files if f.is_file()]
            hasher = hashlib.sha256()
            # Hash up to 10 chunks to be deterministic and fast
            for f in chunk_files[:10]:
                hasher.update(f.name.encode('utf-8'))
                with open(f, 'rb') as fh:
                    hasher.update(fh.read(4096))
            return hasher.hexdigest()
            
    slices, H, W = volume.shape
    
    # Select up to 10 slices deterministically
    slice_step = max(1, slices // 10)
    slice_indices = list(range(0, slices, slice_step))[:10]
    
    samples = []
    for s in slice_indices:
        # Deterministic 10x10 spatial grid
        y_indices = np.linspace(0, H - 1, 10, dtype=int)
        x_indices = np.linspace(0, W - 1, 10, dtype=int)
        for y in y_indices:
            for x in x_indices:
                samples.append(float(volume[s, y, x]))
                
    samples_str = ",".join(map(str, samples))
    return hashlib.sha256(samples_str.encode('utf-8')).hexdigest()

