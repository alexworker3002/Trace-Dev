import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import zarr

def read_ome_zarr_metadata(path: str | Path) -> Dict[str, Any]:
    """Reads OME-Zarr metadata from root."""
    path_str = str(path)
    # Handle zarr v3
    if os.path.exists(os.path.join(path_str, "zarr.json")):
        with open(os.path.join(path_str, "zarr.json"), "r") as f:
            metadata = json.load(f)
        return metadata.get("attributes", {}).get("ome", {})
    else:
        # Fallback to zarr python API for v2
        store = zarr.DirectoryStore(path_str)
        root = zarr.group(store=store)
        return dict(root.attrs).get("ome", {})

def get_zarr_array(root_path: str | Path, sub_path: str) -> Any:
    """Returns the zarr array (or a mock object with shape) for a specific relative path."""
    import zarr.core
    target_path = Path(root_path) / sub_path
    
    # Check if zarr v3 zarr.json exists
    if (target_path / "zarr.json").exists():
        with open(target_path / "zarr.json", "r") as f:
            metadata = json.load(f)
            
        class MockArray:
            def __init__(self, shape, chunks=None):
                self.shape = tuple(shape)
                self.chunks = tuple(chunks) if chunks is not None else None
                
        chunks = None
        chunk_grid = metadata.get("chunk_grid", {})
        if isinstance(chunk_grid, dict):
            config = chunk_grid.get("configuration", {})
            if isinstance(config, dict):
                chunks = config.get("chunk_shape")
                
        return MockArray(metadata.get("shape", []), chunks)
        
    return zarr.open(str(target_path), mode='r')

def extract_physical_metadata(metadata: Dict[str, Any], target_path: str) -> Tuple[Optional[Tuple[float, ...]], Optional[Tuple[float, ...]]]:
    """Extracts spacing and origin from OME-Zarr metadata (multiscales) for a specific dataset path."""
    multiscales = metadata.get("multiscales", [])
    if not multiscales:
        return None, None
        
    datasets = multiscales[0].get("datasets", [])
    if not datasets:
        return None, None
        
    spacing = None
    origin = None
    
    for ds in datasets:
        if ds.get("path") == target_path:
            coord_transformations = ds.get("coordinateTransformations", [])
            for ct in coord_transformations:
                if ct.get("type") == "scale":
                    spacing = tuple(ct.get("scale", []))
                elif ct.get("type") == "translation":
                    origin = tuple(ct.get("translation", []))
            
            if spacing is not None and origin is None:
                origin = tuple([0.0] * len(spacing))
            break
            
    return spacing, origin

def verify_axes(metadata: Dict[str, Any], expected_axes: list[str]) -> bool:
    """Verifies that the axes match the expected configuration."""
    multiscales = metadata.get("multiscales", [])
    if not multiscales:
        return False
        
    axes = multiscales[0].get("axes", [])
    # axes can be dicts with "name" key
    if all(isinstance(ax, dict) for ax in axes):
        found_axes = [ax.get("name") for ax in axes]
    else:
        found_axes = list(axes)
        
    return found_axes == expected_axes
