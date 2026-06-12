import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from numcodecs import Blosc

from trace_ct.audit.logger import AuditLogger
from trace_ct.audit.schemas import (
    ArchitectureHashes,
    GlobalHashes,
    StageHashes,
    StagePassFailRecord,
)
from trace_ct.cli.audit_data import run_g0_audit
from trace_ct.config.defaults import load_dataset_config, load_thresholds_config
from trace_ct.data.normalization import apply_normalization, calculate_volume_statistics
from trace_ct.data.splits import load_splits
from trace_ct.models.context import ContextEncoder
from trace_ct.models.denoiser import Denoiser
from trace_ct.models.proposal import ProposalGenerator
from trace_ct.training.stages import G1MaskedBaseline, G2ContextGating, compute_gradient
from trace_ct.training.state_machine import Stage, TraceCTStateMachine
from trace_ct.utils.hashing import compute_architecture_hash, compute_code_hash, compute_dict_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-data TRACE-CT G0/G1/G2 smoke validation.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--thresholds", default="configs/thresholds.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", default="G1-G2", choices=["G1", "G1-G2"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--volume-id", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _dtype_from_zarr(data_type: str) -> np.dtype:
    if data_type == "int16":
        return np.dtype("<i2")
    if data_type == "uint16":
        return np.dtype("<u2")
    if data_type == "int32":
        return np.dtype("<i4")
    if data_type == "uint32":
        return np.dtype("<u4")
    if data_type == "float32":
        return np.dtype("<f4")
    raise ValueError(f"Unsupported Zarr v3 data_type for smoke read: {data_type}")


def read_zarr_v3_metadata(array_path: Path) -> Dict[str, object]:
    with open(array_path / "zarr.json", "r") as f:
        return json.load(f)


def decode_zarr_v3_chunk(array_path: Path, chunk_coords: Tuple[int, int, int], metadata: Dict[str, object]) -> np.ndarray:
    chunk_shape = tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
    dtype = _dtype_from_zarr(str(metadata["data_type"]))
    chunk_path = array_path / "c" / str(chunk_coords[0]) / str(chunk_coords[1]) / str(chunk_coords[2])
    if not chunk_path.exists():
        return np.full(chunk_shape, metadata.get("fill_value", 0), dtype=dtype)

    raw = chunk_path.read_bytes()
    codecs = metadata.get("codecs", [])
    blosc_config = None
    for codec in codecs:
        if codec.get("name") == "blosc":
            blosc_config = codec.get("configuration", {})
            break
    if blosc_config is None:
        raise ValueError(f"Unsupported Zarr v3 codec stack in {array_path}: missing blosc codec")

    shuffle_name = blosc_config.get("shuffle", "noshuffle")
    shuffle = {
        "noshuffle": Blosc.NOSHUFFLE,
        "shuffle": Blosc.SHUFFLE,
        "bitshuffle": Blosc.BITSHUFFLE,
    }.get(shuffle_name)
    if shuffle is None:
        raise ValueError(f"Unsupported blosc shuffle mode: {shuffle_name}")

    decoder = Blosc(
        cname=blosc_config.get("cname", "lz4"),
        clevel=int(blosc_config.get("clevel", 3)),
        shuffle=shuffle,
        blocksize=int(blosc_config.get("blocksize", 0)),
    )
    decoded = decoder.decode(raw)
    return np.frombuffer(decoded, dtype=dtype).reshape(chunk_shape)


def read_patch(array_path: Path, z_index: int, y0: int, x0: int, patch_size: Tuple[int, int]) -> np.ndarray:
    metadata = read_zarr_v3_metadata(array_path)
    chunk_shape = tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
    cz, cy, cx = chunk_shape
    ph, pw = patch_size
    if ph > cy or pw > cx:
        raise ValueError(f"Smoke reader expects patch {patch_size} to fit inside one chunk {chunk_shape}.")
    chunk_coords = (z_index // cz, y0 // cy, x0 // cx)
    z_in = z_index % cz
    y_in = y0 % cy
    x_in = x0 % cx
    chunk = decode_zarr_v3_chunk(array_path, chunk_coords, metadata)
    patch = chunk[z_in, y_in : y_in + ph, x_in : x_in + pw]
    if patch.shape != (ph, pw):
        raise ValueError(f"Patch shape {patch.shape} does not match expected {(ph, pw)}")
    return np.array(patch)


def choose_patch_window(array_path: Path, patch_size: Tuple[int, int]) -> Tuple[int, int, int]:
    metadata = read_zarr_v3_metadata(array_path)
    shape = tuple(metadata["shape"])
    chunk_shape = tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
    _, cy, cx = chunk_shape
    ph, pw = patch_size
    z_index = shape[0] // 2
    y_chunk = (shape[1] // 2) // cy
    x_chunk = (shape[2] // 2) // cx
    y0 = y_chunk * cy + max(0, (cy - ph) // 2)
    x0 = x_chunk * cx + max(0, (cx - pw) // 2)
    return z_index, y0, x0


def make_real_batch(config_path: str, device: str, split: str, volume_id: str | None) -> tuple[Dict[str, torch.Tensor | List[str]], Dict[str, object]]:
    dataset_yaml = load_dataset_config(config_path)
    dataset_root = Path(dataset_yaml.dataset.root)
    dataset_dir = dataset_root / dataset_yaml.dataset.dataset_dir
    splits = load_splits(dataset_yaml.splits.split_file, dataset_root, dataset_yaml.splits)
    split_ids = splits.get(split, [])
    if not split_ids:
        raise ValueError(f"Split {split!r} is empty or missing.")
    selected_volume = volume_id or split_ids[0]
    if selected_volume not in split_ids:
        raise ValueError(f"Volume {selected_volume} is not in split {split}: {split_ids}")

    root_path = dataset_dir / f"{selected_volume}_ome.zarr"
    reg_path = root_path / "REG" / dataset_yaml.dataset.reg_level
    patch_size = tuple(dataset_yaml.dataset.patch_size)
    z_index, y0, x0 = choose_patch_window(reg_path, patch_size)

    metadata = read_zarr_v3_metadata(reg_path)
    chunk_shape = tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
    cz, cy, cx = chunk_shape
    offsets = dataset_yaml.dataset.context_offsets or [-1, 1]
    adjacent_z = min(max(z_index + offsets[0], 0), metadata["shape"][0] - 1)
    center_patch = read_patch(reg_path, z_index, y0, x0, patch_size).astype(np.float32)
    adjacent_patch = read_patch(reg_path, adjacent_z, y0, x0, patch_size).astype(np.float32)

    stat_block = decode_zarr_v3_chunk(
        reg_path,
        (z_index // cz, y0 // cy, x0 // cx),
        metadata,
    ).astype(np.float32)
    mean, std = calculate_volume_statistics(stat_block, dataset_yaml.normalization)
    norm_center = apply_normalization(center_patch, mean, std, dataset_yaml.normalization).astype(np.float32)
    norm_adjacent = apply_normalization(adjacent_patch, mean, std, dataset_yaml.normalization).astype(np.float32)

    noisy = torch.from_numpy(norm_center).unsqueeze(0).unsqueeze(0).to(device)
    adjacent = torch.from_numpy(norm_adjacent).unsqueeze(0).unsqueeze(0).to(device)
    grad = compute_gradient(noisy)
    homogeneous_mask = (grad < torch.quantile(grad, 0.60)).float()

    batch: Dict[str, torch.Tensor | List[str]] = {
        "noisy": noisy,
        "adjacent_noisy": adjacent,
        "homogeneous_mask": homogeneous_mask,
        "volume_id": [selected_volume],
    }
    metrics = {
        "volume_id": selected_volume,
        "z_index": z_index,
        "adjacent_z_index": adjacent_z,
        "y0": y0,
        "x0": x0,
        "patch_h": patch_size[0],
        "patch_w": patch_size[1],
        "raw_min": float(center_patch.min()),
        "raw_max": float(center_patch.max()),
        "raw_mean": float(center_patch.mean()),
        "raw_std": float(center_patch.std()),
        "norm_mean": float(norm_center.mean()),
        "norm_std": float(norm_center.std()),
    }
    return batch, metrics


def build_hash_context(config_path: str) -> tuple[GlobalHashes, ArchitectureHashes]:
    dataset_yaml = load_dataset_config(config_path)
    global_hashes = GlobalHashes(
        dataset_config_hash=compute_dict_hash(dataset_yaml.dataset.model_dump()),
        normalization_config_hash=compute_dict_hash(dataset_yaml.normalization.model_dump()),
        split_config_hash=compute_dict_hash(dataset_yaml.splits.model_dump()),
        code_hash=compute_code_hash(Path.cwd()),
    )
    architecture_hashes = ArchitectureHashes(
        denoiser_architecture_hash=compute_architecture_hash(Denoiser()),
        context_architecture_hash=compute_architecture_hash(ContextEncoder()),
        proposal_architecture_hash=compute_architecture_hash(ProposalGenerator()),
    )
    return global_hashes, architecture_hashes


def log_stage(logger: AuditLogger, stage: Stage, status: str, global_hashes: GlobalHashes, architecture_hashes: ArchitectureHashes, policy_hash: str, metrics: Dict[str, float] | None = None, failure_reasons: List[str] | None = None) -> None:
    next_stage = Stage.G2.value if stage == Stage.G1 and status == "pass" else None
    if stage == Stage.G2 and status == "pass":
        next_stage = Stage.G3.value
    record = StagePassFailRecord(
        run_id=logger.run_id,
        stage=stage.value,
        status=status,  # type: ignore[arg-type]
        timestamp=str(time.time()),
        global_hashes=global_hashes,
        architecture_hashes=architecture_hashes,
        stage_hashes=StageHashes(policy_hash=policy_hash),
        metrics=metrics or {},
        failure_reasons=failure_reasons or [],
        next_allowed_stage=next_stage,
    )
    logger.log_stage_record(record)


def run_smoke(args: argparse.Namespace) -> int:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_g0_audit(SimpleNamespace(config=args.config, run_id=args.run_id))
    logger = AuditLogger(args.run_id)
    thresholds = load_thresholds_config(args.thresholds)
    machine = TraceCTStateMachine(logger, thresholds=thresholds)
    global_hashes, architecture_hashes = build_hash_context(args.config)
    policy_hash = compute_dict_hash({"config": args.config, "thresholds": args.thresholds, "stages": args.stages, "split": args.split, "volume_id": args.volume_id, "seed": args.seed})

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    batch, batch_metrics = make_real_batch(args.config, args.device, args.split, args.volume_id)
    stages = [Stage.G1] if args.stages == "G1" else [Stage.G1, Stage.G2]
    for stage in stages:
        try:
            if stage == Stage.G1:
                stage_obj = G1MaskedBaseline(machine, device=args.device)
            elif stage == Stage.G2:
                stage_obj = G2ContextGating(machine, device=args.device)
            else:
                raise ValueError(f"Unsupported smoke stage: {stage}")
            loss = float(stage_obj.step(batch))
            if not np.isfinite(loss):
                raise ValueError(f"Non-finite loss: {loss}")
            metrics = {key: float(value) for key, value in batch_metrics.items() if isinstance(value, (int, float))}
            metrics["loss"] = loss
            if stage == Stage.G2:
                report_path = logger.run_dir / "reports" / "context_audit.json"
                with open(report_path, "r") as f:
                    context_report = json.load(f)
                metrics.update({key: float(value) for key, value in context_report.items() if isinstance(value, (int, float))})
                context_limits = thresholds.context_audit
                reasons = []
                hf_leakage = float(context_report.get("high_frequency_leakage_ratio", 0.0))
                hf_corr = float(context_report.get("high_frequency_correlation", 0.0))
                if hf_leakage > context_limits.max_high_frequency_leakage_ratio:
                    reasons.append(
                        f"G2 high-frequency leakage {hf_leakage:.4f} exceeds max {context_limits.max_high_frequency_leakage_ratio}"
                    )
                if abs(hf_corr) > context_limits.max_high_frequency_correlation:
                    reasons.append(
                        f"G2 absolute high-frequency correlation {abs(hf_corr):.4f} exceeds max {context_limits.max_high_frequency_correlation}"
                    )
                if reasons:
                    log_stage(
                        logger,
                        stage,
                        "fail",
                        global_hashes,
                        architecture_hashes,
                        policy_hash,
                        metrics=metrics,
                        failure_reasons=reasons,
                    )
                    print(f"Real data smoke failed at {stage.value}: {'; '.join(reasons)}")
                    print(f"Smoke artifacts: runs/{args.run_id}")
                    return 1
            log_stage(logger, stage, "pass", global_hashes, architecture_hashes, policy_hash, metrics=metrics)
        except Exception as exc:
            log_stage(logger, stage, "fail", global_hashes, architecture_hashes, policy_hash, failure_reasons=[str(exc)])
            print(f"Real data smoke failed at {stage.value}: {exc}")
            print(f"Smoke artifacts: runs/{args.run_id}")
            return 1

    print(f"Real data smoke complete: runs/{args.run_id}")
    return 0


def main() -> None:
    raise SystemExit(run_smoke(parse_args()))


if __name__ == "__main__":
    main()
