import numpy as np
from typing import Tuple, Dict, Any
from trace_ct.config.schema import NormalizationConfig

def calculate_volume_statistics(arr: np.ndarray, config: NormalizationConfig) -> Tuple[float, float]:
    """Calculates mean and std of a volume after applying clipping if configured."""
    data = arr
    if config.clip.enabled:
        data = np.clip(data, config.clip.min, config.clip.max)
        
    mean = float(np.mean(data))
    std = float(np.std(data))
    
    if std < config.safety.min_std:
        if config.safety.on_zero_std == "abort_audit":
            raise ValueError(f"Standard deviation ({std}) is below safety minimum ({config.safety.min_std}).")
            
    return mean, std

def apply_normalization(arr: np.ndarray, mean: float, std: float, config: NormalizationConfig) -> np.ndarray:
    """Applies normalization according to the mode."""
    data = arr
    if config.clip.enabled:
        data = np.clip(data, config.clip.min, config.clip.max)
        
    if config.scaling.mode == "zscore_after_clip":
        return (data - mean) / std
    elif config.scaling.mode == "minmax_after_clip":
        return (data - config.clip.min) / (config.clip.max - config.clip.min)
    else:
        raise ValueError(f"Unknown scaling mode {config.scaling.mode}")
