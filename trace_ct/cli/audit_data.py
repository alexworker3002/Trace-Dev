import argparse
import time
from pathlib import Path
import json

from trace_ct.config.defaults import load_dataset_config, load_thresholds_config
from trace_ct.utils.paths import setup_run_directories
from trace_ct.audit.logger import AuditLogger
from trace_ct.audit.schemas import StagePassFailRecord, GlobalHashes, ArchitectureHashes, StageHashes
from trace_ct.utils.hashing import compute_code_hash, compute_file_hash, compute_dict_hash, compute_sample_hash, compute_metadata_hash
from trace_ct.training.state_machine import TraceCTStateMachine, Stage
from trace_ct.data.zarr_io import read_ome_zarr_metadata, get_zarr_array, extract_physical_metadata, verify_axes
from trace_ct.data.alignment import verify_physical_alignment
from trace_ct.data.splits import load_splits

def parse_args():
    parser = argparse.ArgumentParser(description="G0 Bounded Data Audit")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--run-id", type=str, required=True, help="Unique Run ID")
    return parser.parse_args()

def run_g0_audit(args):
    dataset_yaml = load_dataset_config(args.config)
    
    # 1. Setup Directories & Logger
    project_root = Path.cwd()
    run_dir = setup_run_directories(args.run_id)
    logger = AuditLogger(args.run_id)
    state_machine = TraceCTStateMachine(logger)
    
    global_hashes = GlobalHashes(
        dataset_config_hash=compute_dict_hash(dataset_yaml.dataset.model_dump()),
        normalization_config_hash=compute_dict_hash(dataset_yaml.normalization.model_dump()),
        split_config_hash=compute_dict_hash(dataset_yaml.splits.model_dump()),
        code_hash=compute_code_hash(project_root)
    )
    
    record_output_hashes = {}
    policy_hash = compute_dict_hash(dataset_yaml.bounded_audit.model_dump())
    architecture_hashes = ArchitectureHashes()
    audited_volume_count = 0
    
    failure_reasons = []
    dataset_root = Path(dataset_yaml.dataset.root) / dataset_yaml.dataset.dataset_dir
    manifest_path = Path(dataset_yaml.dataset.root) / dataset_yaml.dataset.volume_manifest
    
    # Check dataset existence
    if not dataset_root.exists():
        failure_reasons.append(f"Dataset root {dataset_root} does not exist.")
    
    # Execute the bounded data split isolation checks
    try:
        # Check Splits Isolation
        if dataset_yaml.splits.split_file:
            load_splits(dataset_yaml.splits.split_file, Path(dataset_yaml.dataset.root), dataset_yaml.splits)
    except Exception as e:
        failure_reasons.append(f"Split Audit Failed: {str(e)}")
        
    # Load manifest and perform physical/safety checks on each volume within the bounded_audit limit
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            volumes = json.load(f)
            
        for vol_id in list(volumes.keys())[:dataset_yaml.bounded_audit.max_volumes]:
            reg_path = dataset_root / dataset_yaml.dataset.reg_rel_template.format(volume_id=vol_id)
            hr_path = dataset_root / dataset_yaml.dataset.hr_rel_template.format(volume_id=vol_id)
            audited_volume_count += 1
            
            try:
                # Read root metadata
                root_path = dataset_root / f"{vol_id}_ome.zarr"
                reg_meta = read_ome_zarr_metadata(root_path)
                hr_meta = read_ome_zarr_metadata(root_path)
                
                # OME-Zarr multiscales specify datasets relative to root, e.g. "REG/0"
                reg_rel = f"REG/{dataset_yaml.dataset.reg_level}"
                hr_rel = f"HR/{dataset_yaml.dataset.hr_level}"
                
                reg_sp, reg_or = extract_physical_metadata(reg_meta, reg_rel)
                hr_sp, hr_or = extract_physical_metadata(hr_meta, hr_rel)
                
                if not verify_axes(reg_meta, dataset_yaml.dataset.axes):
                    failure_reasons.append(f"Axes mismatch in {root_path}")
                    
                reg_arr = get_zarr_array(root_path, reg_rel)
                hr_arr = get_zarr_array(root_path, hr_rel)
                
                # Real Metadata & Sample Content Hashing
                vol_metadata_hash = compute_metadata_hash(reg_meta)
                vol_sample_hash = compute_sample_hash(reg_arr, root_path / reg_rel)
                
                if len(vol_metadata_hash) != 64:
                    failure_reasons.append(f"Invalid metadata hash for volume {vol_id}")
                if len(vol_sample_hash) != 64:
                    failure_reasons.append(f"Invalid sample hash for volume {vol_id}")
                    
                record_output_hashes[f"{vol_id}_metadata_hash"] = vol_metadata_hash
                record_output_hashes[f"{vol_id}_sample_hash"] = vol_sample_hash
                
                is_aligned, msg = verify_physical_alignment(
                    reg_sp, reg_or, reg_arr.shape,
                    hr_sp, hr_or, hr_arr.shape,
                    dataset_yaml.alignment
                )
                if not is_aligned:
                    failure_reasons.append(f"Alignment Failed for {vol_id}: {msg}")
                    
                # 1. Patch Sampling & Chunk relationship check
                slices = reg_arr.shape[0]
                H = reg_arr.shape[1]
                W = reg_arr.shape[2]
                ph, pw = dataset_yaml.dataset.patch_size
                
                if slices < 3:
                    failure_reasons.append(f"Volume {vol_id} has only {slices} slices (minimum 3 required for first/middle/last sampling).")
                if ph > H or pw > W:
                    failure_reasons.append(f"Patch size {ph}x{pw} exceeds volume dimension {H}x{W} for {vol_id}.")
                    
                # Verify chunk compatibility (chunk-not-used-as-patch)
                chunks = getattr(reg_arr, "chunks", None)
                if chunks is not None and len(chunks) == 3:
                    cz, cy, cx = chunks
                    if dataset_yaml.bounded_audit.verify_chunk_not_used_as_patch:
                        if cy == ph and cx == pw:
                            failure_reasons.append(f"Volume {vol_id} chunk size {cy}x{cx} is exactly equal to patch size {ph}x{pw}, violating verify_chunk_not_used_as_patch constraint.")
                        if ph > cy or pw > cx:
                            failure_reasons.append(f"Volume {vol_id} patch size {ph}x{pw} exceeds chunk size {cy}x{cx}.")
                            
                # 2. Slice continuity (Physical spacing check along Z axis)
                if reg_sp is not None and len(reg_sp) == 3:
                    # Spacing on the slice dimension (usually index 0)
                    z_spacing = reg_sp[0]
                    if z_spacing <= 0:
                        failure_reasons.append(f"Invalid physical spacing {z_spacing} along Z axis.")
                        
                # 3. HU Calibration & Normalization consistency check on real Zarr data
                # We check if indexable and not a MockArray
                if hasattr(reg_arr, "__getitem__") and not reg_arr.__class__.__name__ == "MockArray":
                    import numpy as np
                    from trace_ct.data.normalization import apply_normalization, calculate_volume_statistics
                    
                    # Read sample slices
                    mid_idx = slices // 2
                    reg_slice = reg_arr[mid_idx]
                    hr_slice = hr_arr[mid_idx]
                    
                    # Force conversion to numpy array if it's zarr Array
                    if not isinstance(reg_slice, np.ndarray):
                        reg_slice = np.array(reg_slice)
                    if not isinstance(hr_slice, np.ndarray):
                        hr_slice = np.array(hr_slice)
                        
                    # HU Calibration Check
                    reg_std = np.std(reg_slice)
                    reg_min = np.min(reg_slice)
                    reg_max = np.max(reg_slice)
                    
                    if reg_std < dataset_yaml.normalization.safety.min_std:
                        failure_reasons.append(f"HU Calibration error for {vol_id}: standard deviation {reg_std:.6f} below safety threshold.")
                    if reg_min > -500:
                        failure_reasons.append(f"HU Calibration error for {vol_id}: minimum HU {reg_min} is not in standard CT air range (should be <= -500).")
                    if reg_max < 0:
                        failure_reasons.append(f"HU Calibration error for {vol_id}: maximum HU {reg_max} is not in standard CT tissue range (should be >= 0).")
                        
                    # Normalization Consistency
                    reg_mean, reg_std_val = calculate_volume_statistics(reg_slice, dataset_yaml.normalization)
                    hr_mean, hr_std_val = calculate_volume_statistics(hr_slice, dataset_yaml.normalization)
                    
                    norm_reg = apply_normalization(reg_slice, reg_mean, reg_std_val, dataset_yaml.normalization)
                    norm_hr = apply_normalization(hr_slice, hr_mean, hr_std_val, dataset_yaml.normalization)
                    
                    if dataset_yaml.normalization.enforce_reg_hr_consistency:
                        mean_diff = abs(float(np.mean(norm_reg) - np.mean(norm_hr)))
                        if mean_diff > 0.5:
                            failure_reasons.append(f"Normalization inconsistency for {vol_id}: REG vs HR normalized mean diff {mean_diff:.4f} > 0.5")
            except Exception as e:
                failure_reasons.append(f"Zarr IO Failed for {vol_id}: {str(e)}")
    else:
        failure_reasons.append(f"Volume manifest not found at {manifest_path}")
        
    status = "pass" if not failure_reasons else "fail"
    fallback_action = state_machine.get_fallback_action(Stage.G0) if status == "fail" else None
    stage_hashes = StageHashes(
        policy_hash=policy_hash,
        output_hashes=record_output_hashes,
    )
    metrics = {
        "audited_volume_count": float(audited_volume_count),
        "output_hash_count": float(len(record_output_hashes)),
    }
    
    record = StagePassFailRecord(
        run_id=args.run_id,
        stage=Stage.G0.value,
        status=status,
        timestamp=str(time.time()),
        global_hashes=global_hashes,
        architecture_hashes=architecture_hashes,
        stage_hashes=stage_hashes,
        metrics=metrics,
        failure_reasons=failure_reasons,
        fallback_action_taken=fallback_action,
        next_allowed_stage=Stage.G1.value if status == "pass" else None
    )
    
    logger.log_stage_record(record)
    
    if status == "fail":
        print(f"G0 Audit FAILED: {failure_reasons}")
        print(f"Fallback Action: {fallback_action}")
        exit(1)
    else:
        print("G0 Audit PASSED.")
        
if __name__ == "__main__":
    args = parse_args()
    run_g0_audit(args)
