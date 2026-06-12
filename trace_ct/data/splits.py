import json
from pathlib import Path
from typing import Dict, List, Set
from trace_ct.config.schema import SplitsConfig

def load_splits(filepath: str | Path, dataset_root: Path, config: SplitsConfig) -> Dict[str, List[str]]:
    """Loads splits from a file, optionally relative to dataset root."""
    path = Path(filepath)
    if config.split_file_relative_to_dataset_root:
        path = dataset_root / path
        
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
        
    with open(path, 'r') as f:
        splits = json.load(f)
        
    if config.require_train_val_volume_disjoint:
        train_set = set(splits.get("train", []))
        val_set = set(splits.get("val", []))
        intersection = train_set.intersection(val_set)
        if intersection:
            raise ValueError(f"Train and val splits are not disjoint. Overlap: {intersection}")
            
    return splits
