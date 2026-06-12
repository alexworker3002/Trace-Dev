import yaml
from pathlib import Path
from .schema import TraceCTDatasetYAML, ProtocolConfig, ThresholdsYAML

def load_dataset_config(path: str | Path) -> TraceCTDatasetYAML:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return TraceCTDatasetYAML(**data)

def load_protocol_config(path: str | Path) -> ProtocolConfig:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return ProtocolConfig(**data)

def load_thresholds_config(path: str | Path) -> ThresholdsYAML:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    # The yaml root is 'thresholds:'
    return ThresholdsYAML(**data['thresholds'])
