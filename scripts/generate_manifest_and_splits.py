import json
import os
from pathlib import Path

def main():
    base_dir = Path("datasets/004-FBP")
    manifest_dir = Path("datasets/manifests")
    splits_dir = Path("datasets/splits")
    
    manifest_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    volumes = {}
    train_vols = []
    val_vols = []
    
    if base_dir.exists():
        dirs = sorted([d.name for d in base_dir.iterdir() if d.is_dir() and d.name.endswith("_ome.zarr")])
        for i, d in enumerate(dirs):
            vol_id = d.replace("_ome.zarr", "")
            volumes[vol_id] = {"path": str(base_dir / d)}
            
            # Simple split: first 80% train, rest val
            if i < len(dirs) * 0.8:
                train_vols.append(vol_id)
            else:
                val_vols.append(vol_id)
                
    with open(manifest_dir / "004_fbp_volumes.json", "w") as f:
        json.dump(volumes, f, indent=2)
        
    with open(splits_dir / "fbp_splits.json", "w") as f:
        json.dump({"train": train_vols, "val": val_vols}, f, indent=2)
        
if __name__ == "__main__":
    main()
