import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.utils.data

from trace_ct.config.defaults import load_dataset_config
from trace_ct.data.splits import load_splits
from trace_ct.cli.real_data_smoke import read_zarr_v3_metadata


class TraceCTDataset(torch.utils.data.Dataset):
    """
    TRACE-CT PyTorch Dataset supporting multi-volume, slice-level, and patch-level loading.
    Implements a lightweight in-memory cache for decoded Zarr v3 chunks to ensure fast I/O.
    """
    def __init__(self, dataset_config_path: str, split: str = "train", patches_per_volume: int = 256):
        self.dataset_config_path = dataset_config_path
        self.split = split
        self.patches_per_volume = patches_per_volume
        
        self.dataset_yaml = load_dataset_config(dataset_config_path)
        self.dataset_root = Path(self.dataset_yaml.dataset.root)
        self.dataset_dir = self.dataset_root / self.dataset_yaml.dataset.dataset_dir
        self.splits = load_splits(self.dataset_yaml.splits.split_file, self.dataset_root, self.dataset_yaml.splits)
        
        self.volume_ids = self.splits.get(split, [])
        if not self.volume_ids:
            raise ValueError(f"Split {split} has no volumes configured.")
            
        self.patch_size = tuple(self.dataset_yaml.dataset.patch_size)
        self.offsets = self.dataset_yaml.dataset.context_offsets or [-1, 1]
        
        # Cache for decoded chunks: Key is (array_path_str, chunk_coords_tuple), Value is np.ndarray
        self._chunk_cache = {}
        
        # Pre-generate sampling coordinates across all volumes in the split
        self.coords = []
        for vol_id in self.volume_ids:
            root_path = self.dataset_dir / f"{vol_id}_ome.zarr"
            reg_path = root_path / "REG" / self.dataset_yaml.dataset.reg_level
            meta = read_zarr_v3_metadata(reg_path)
            shape = tuple(meta["shape"])
            
            # Select slice indices evenly spaced out in [16, shape[0] - 17]
            z_min = 16
            z_max = shape[0] - 17
            num_slices = 16
            z_values = np.linspace(z_min, z_max, num_slices, dtype=int).tolist()
            
            # We want to distribute coordinates evenly across Y and X
            patches_per_slice = int(np.ceil(self.patches_per_volume / num_slices))
            grid_side = int(np.ceil(np.sqrt(patches_per_slice)))
            
            _, h, w = shape
            ph, pw = self.patch_size
            
            y_coords = np.linspace(64, h - ph - 64, grid_side, dtype=int).tolist()
            x_coords = np.linspace(64, w - pw - 64, grid_side, dtype=int).tolist()
            
            for z in z_values:
                for y in y_coords:
                    for x in x_coords:
                        self.coords.append((vol_id, int(z), int(y), int(x)))
                        
        # Limit total coords to match target count
        total_target = len(self.volume_ids) * self.patches_per_volume
        if len(self.coords) > total_target:
            self.coords = self.coords[:total_target]
            
        # Shuffle coordinates for random batching
        np.random.shuffle(self.coords)
        
    def _get_chunk(self, reg_path: Path, chunk_coords: Tuple[int, int, int], reg_meta: Dict[str, Any]) -> np.ndarray:
        key = (str(reg_path), chunk_coords)
        if key not in self._chunk_cache:
            if len(self._chunk_cache) >= 64:
                # Discard an arbitrary chunk to control memory usage
                self._chunk_cache.pop(next(iter(self._chunk_cache)))
            from trace_ct.cli.real_data_smoke import decode_zarr_v3_chunk
            self._chunk_cache[key] = decode_zarr_v3_chunk(reg_path, chunk_coords, reg_meta)
        return self._chunk_cache[key]
        
    def _read_patch(self, array_path: Path, z_index: int, y0: int, x0: int, metadata: Dict[str, Any]) -> np.ndarray:
        chunk_shape = tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
        cz, cy, cx = chunk_shape
        ph, pw = self.patch_size
        chunk_coords = (z_index // cz, y0 // cy, x0 // cx)
        z_in = z_index % cz
        y_in = y0 % cy
        x_in = x0 % cx
        chunk = self._get_chunk(array_path, chunk_coords, metadata)
        patch = chunk[z_in, y_in : y_in + ph, x_in : x_in + pw]
        return np.array(patch)
        
    def __len__(self) -> int:
        return len(self.coords)
        
    def __getitem__(self, index: int) -> Dict[str, Any]:
        from trace_ct.data.normalization import apply_normalization, calculate_volume_statistics
        from trace_ct.training.stages import compute_gradient
        
        # Rejection sampling loop to avoid empty air patches
        max_attempts = 10
        center_patch = None
        vol_id, z, y, x = None, None, None, None
        
        for _ in range(max_attempts):
            vol_id, z, y, x = self.coords[index]
            root_path = self.dataset_dir / f"{vol_id}_ome.zarr"
            reg_path = root_path / "REG" / self.dataset_yaml.dataset.reg_level
            reg_meta = read_zarr_v3_metadata(reg_path)
            
            center_patch = self._read_patch(reg_path, z, y, x, reg_meta).astype(np.float32)
            if center_patch.mean() > -950 and center_patch.std() > 10.0:
                break
            index = np.random.randint(0, len(self.coords))
            
        # If we failed to find a non-air patch, use whatever we have
        if center_patch is None:
            vol_id, z, y, x = self.coords[index]
            root_path = self.dataset_dir / f"{vol_id}_ome.zarr"
            reg_path = root_path / "REG" / self.dataset_yaml.dataset.reg_level
            reg_meta = read_zarr_v3_metadata(reg_path)
            center_patch = self._read_patch(reg_path, z, y, x, reg_meta).astype(np.float32)
            
        adjacent_z = min(max(z + self.offsets[0], 0), reg_meta["shape"][0] - 1)
        adjacent_patch = self._read_patch(reg_path, adjacent_z, y, x, reg_meta).astype(np.float32)
        
        # Load stats chunk for normalization calculations
        cz, cy, cx = reg_meta["chunk_grid"]["configuration"]["chunk_shape"]
        chunk_coords = (z // cz, y // cy, x // cx)
        chunk = self._get_chunk(reg_path, chunk_coords, reg_meta).astype(np.float32)
        mean, std = calculate_volume_statistics(chunk, self.dataset_yaml.normalization)
        
        norm_center = apply_normalization(center_patch, mean, std, self.dataset_yaml.normalization).astype(np.float32)
        norm_adjacent = apply_normalization(adjacent_patch, mean, std, self.dataset_yaml.normalization).astype(np.float32)
        
        noisy = torch.from_numpy(norm_center).unsqueeze(0)
        adjacent = torch.from_numpy(norm_adjacent).unsqueeze(0)
        
        # Build homogeneous_mask
        grad = compute_gradient(noisy.unsqueeze(0))
        homogeneous_mask = (grad < torch.quantile(grad, 0.60)).float().squeeze(0)
        
        batch = {
            "noisy": noisy,
            "adjacent_noisy": adjacent,
            "homogeneous_mask": homogeneous_mask,
            "volume_id": vol_id,
        }
        
        # Load HR target if validation split or explicitly requested for training
        if self.split == "val" or self.dataset_yaml.dataset.use_hr_for_training:
            hr_path = root_path / "HR" / self.dataset_yaml.dataset.hr_level
            hr_meta = read_zarr_v3_metadata(hr_path)
            hr_patch = self._read_patch(hr_path, z, y, x, hr_meta).astype(np.float32)
            norm_hr = apply_normalization(hr_patch, mean, std, self.dataset_yaml.normalization).astype(np.float32)
            clean_proxy = torch.from_numpy(norm_hr).unsqueeze(0)
            
            grad_proxy = compute_gradient(clean_proxy.unsqueeze(0))
            edge_mask = (grad_proxy > torch.quantile(grad_proxy, 0.90)).float().squeeze(0)
            lesion_mask = torch.zeros_like(edge_mask)
            
            batch["clean_proxy"] = clean_proxy
            batch["edge_mask"] = edge_mask
            batch["lesion_mask"] = lesion_mask
            
        return batch
