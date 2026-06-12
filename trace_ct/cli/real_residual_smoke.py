import argparse
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from trace_ct.audit.logger import AuditLogger
from trace_ct.audit.denoising_strength_audit import audit_denoising_strength
from trace_ct.audit.residual_audit import ResidualPool
from trace_ct.audit.schemas import StageHashes, StagePassFailRecord
from trace_ct.cli.real_data_smoke import (
    build_hash_context,
    decode_zarr_v3_chunk,
    make_real_batch,
    read_patch,
    read_zarr_v3_metadata,
    run_smoke as run_g1_g2_smoke,
)
from trace_ct.config.defaults import load_dataset_config, load_protocol_config, load_thresholds_config
from trace_ct.data.normalization import apply_normalization, calculate_volume_statistics
from trace_ct.data.splits import load_splits
from trace_ct.training.stages import G4BaselineProposals, G5ProposalQualification, compute_gradient
from trace_ct.training.state_machine import Stage, TraceCTStateMachine
from trace_ct.utils.hashing import compute_dict_hash
from trace_ct.models.denoising_strength import DenoisingStrengthController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-data TRACE-CT G3/G4 residual smoke validation.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--thresholds", default="configs/thresholds.yaml")
    parser.add_argument("--protocol", default="configs/protocol.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--scan-multiplier", type=int, default=8)
    parser.add_argument("--candidate-family", choices=["paired_raw", "paired_highpass"], default="paired_highpass")
    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--artifact-mode", choices=["minimal", "full"], default="minimal")
    parser.add_argument("--g4-steps", type=int, default=2)
    parser.add_argument("--include-g5", action="store_true")
    parser.add_argument("--g5-steps", type=int, default=1)
    return parser.parse_args()


def cleanup_tensor_artifacts(run_dir: Path) -> None:
    for filename in ["accepted_residuals.pt", "error_residuals.pt"]:
        path = run_dir / "residual_pools" / filename
        if path.exists():
            path.unlink()


def log_stage(
    logger: AuditLogger,
    stage: Stage,
    status: str,
    policy_hash: str,
    config_path: str,
    metrics: Dict[str, float] | None = None,
    failure_reasons: List[str] | None = None,
) -> None:
    global_hashes, architecture_hashes = build_hash_context(config_path)
    next_stage = None
    if status == "pass" and stage == Stage.G3:
        next_stage = Stage.G4.value
    elif status == "pass" and stage == Stage.G4:
        next_stage = Stage.G45.value
    elif status == "pass" and stage == Stage.G45:
        next_stage = Stage.G5.value
    elif status == "pass" and stage == Stage.G5:
        next_stage = Stage.G6.value
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


def _normalization_stats(array_path: Path, z: int, y: int, x: int, normalization) -> Tuple[float, float]:
    metadata = read_zarr_v3_metadata(array_path)
    cz, cy, cx = metadata["chunk_grid"]["configuration"]["chunk_shape"]
    chunk = decode_zarr_v3_chunk(array_path, (z // cz, y // cy, x // cx), metadata).astype(np.float32)
    return calculate_volume_statistics(chunk, normalization)


def _normalized_patch(array_path: Path, z: int, y: int, x: int, patch_size: Tuple[int, int], mean: float, std: float, normalization) -> np.ndarray:
    raw = read_patch(array_path, z, y, x, patch_size).astype(np.float32)
    return apply_normalization(raw, mean, std, normalization).astype(np.float32)


def _candidate_coordinates(shape: Tuple[int, int, int], patch_size: Tuple[int, int], max_candidates: int) -> List[Tuple[int, int, int]]:
    _, h, w = shape
    ph, pw = patch_size
    y_step = max(ph, ph // 2)
    x_step = max(pw, pw // 2)
    y_positions = [y for y in range(0, h - ph + 1, y_step)]
    x_positions = [x for x in range(0, w - pw + 1, x_step)]
    per_slice = max(1, len(y_positions) * len(x_positions))
    z_count = max(1, int(np.ceil(max_candidates / per_slice)))
    z_values = np.linspace(16, max(16, shape[0] - 17), z_count, dtype=int).tolist()
    coords: List[Tuple[int, int, int]] = []
    for z in z_values:
        for y in y_positions:
            for x in x_positions:
                coords.append((int(z), int(y), int(x)))
                if len(coords) >= max_candidates:
                    return coords
    return coords


def _prepare_residual(residual: torch.Tensor, family: str) -> torch.Tensor:
    if family == "paired_highpass":
        residual_4d = residual.unsqueeze(0)
        residual = (residual_4d - F.avg_pool2d(residual_4d, kernel_size=9, stride=1, padding=4)).squeeze(0)
    residual = residual - residual.mean()
    return residual


def build_residual_candidates(args: argparse.Namespace) -> Tuple[str, str, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, object]]:
    dataset_yaml = load_dataset_config(args.config)
    dataset_root = Path(dataset_yaml.dataset.root)
    dataset_dir = dataset_root / dataset_yaml.dataset.dataset_dir
    splits = load_splits(dataset_yaml.splits.split_file, dataset_root, dataset_yaml.splits)
    donor_ids = list(splits.get("val", []))
    receiver_ids = [vol for vol in splits.get("train", []) if vol not in donor_ids]
    if not donor_ids:
        raise ValueError("No validation/donor volumes available for residual audit.")
    if not receiver_ids:
        raise ValueError("No receiver train volumes available outside donor set.")

    donor_id = donor_ids[0]
    receiver_id = receiver_ids[0]
    patch_size = tuple(dataset_yaml.dataset.patch_size)

    donor_root = dataset_dir / f"{donor_id}_ome.zarr"
    receiver_root = dataset_dir / f"{receiver_id}_ome.zarr"
    donor_reg = donor_root / "REG" / dataset_yaml.dataset.reg_level
    donor_hr = donor_root / "HR" / dataset_yaml.dataset.hr_level
    receiver_reg = receiver_root / "REG" / dataset_yaml.dataset.reg_level

    donor_meta = read_zarr_v3_metadata(donor_reg)
    receiver_meta = read_zarr_v3_metadata(receiver_reg)
    scan_count = max(args.max_candidates, args.max_candidates * max(1, args.scan_multiplier))
    coords = _candidate_coordinates(tuple(donor_meta["shape"]), patch_size, scan_count)

    candidates = []
    scales: List[float] = []

    for index, (z, y, x) in enumerate(coords):
        mean, std = _normalization_stats(donor_reg, z, y, x, dataset_yaml.normalization)
        reg_patch = _normalized_patch(donor_reg, z, y, x, patch_size, mean, std, dataset_yaml.normalization)
        hr_patch = _normalized_patch(donor_hr, z, y, x, patch_size, mean, std, dataset_yaml.normalization)
        residual = torch.from_numpy(reg_patch - hr_patch).unsqueeze(0)
        residual = _prepare_residual(residual, args.candidate_family)

        receiver_z = int(np.clip(receiver_meta["shape"][0] // 2 + index - len(coords) // 2, 0, receiver_meta["shape"][0] - 1))
        r_mean, r_std = _normalization_stats(receiver_reg, receiver_z, y, x, dataset_yaml.normalization)
        receiver_patch = _normalized_patch(receiver_reg, receiver_z, y, x, patch_size, r_mean, r_std, dataset_yaml.normalization)
        receiver_tensor = torch.from_numpy(receiver_patch).unsqueeze(0).unsqueeze(0)
        proxy = (receiver_tensor - F.avg_pool2d(receiver_tensor, kernel_size=7, stride=1, padding=3)).squeeze(0)

        proxy_std = float(proxy.std().item() + 1e-8)
        residual_std = float(residual.std().item() + 1e-8)
        scale = proxy_std / residual_std
        if scale < args.scale_min or scale > args.scale_max:
            continue
        residual = residual * scale

        hr_tensor = torch.from_numpy(hr_patch).unsqueeze(0).unsqueeze(0)
        grad = compute_gradient(hr_tensor)
        edge_score = float(torch.quantile(grad, 0.90).item())
        edge_mask = (grad > torch.quantile(grad, 0.90)).float().squeeze(0)
        residual_lf = F.avg_pool2d(residual.unsqueeze(0), kernel_size=8, stride=8).squeeze(0)
        hom_score = edge_score + float(torch.abs(residual_lf).mean().item()) + abs(float(residual.mean().item()))
        candidates.append((hom_score, residual.squeeze(0), edge_mask.squeeze(0), proxy.squeeze(0), scale))

    candidates.sort(key=lambda item: item[0])
    selected = candidates[:args.max_candidates]
    if not selected:
        raise ValueError("No residual candidates were generated.")
    residuals = [item[1] for item in selected]
    edge_masks = [item[2] for item in selected]
    proxies = [item[3] for item in selected]
    scales = [float(item[4]) for item in selected]
    hom_scores = [float(item[0]) for item in selected]

    metrics = {
        "candidate_count": float(len(residuals)),
        "candidate_scan_count": float(len(coords)),
        "candidate_eligible_count": float(len(candidates)),
        "homogeneous_score_mean": float(np.mean(hom_scores)),
        "homogeneous_score_q95": float(np.quantile(hom_scores, 0.95)),
        "calibration_scale_mean": float(np.mean(scales)) if scales else 0.0,
        "calibration_scale_min": float(np.min(scales)) if scales else 0.0,
        "calibration_scale_max": float(np.max(scales)) if scales else 0.0,
    }
    report = {
        "donor_volume_id": donor_id,
        "receiver_volume_id": receiver_id,
        "candidate_count": len(residuals),
        "candidate_scan_count": len(coords),
        "candidate_eligible_count": len(candidates),
        "candidate_generator": f"{args.candidate_family}_homogeneous_ranked_scaled_to_receiver_highpass_proxy",
        "calibration_scale": {
            "mean": metrics["calibration_scale_mean"],
            "min": metrics["calibration_scale_min"],
            "max": metrics["calibration_scale_max"],
        },
        "homogeneous_score": {
            "mean": metrics["homogeneous_score_mean"],
            "q95": metrics["homogeneous_score_q95"],
        },
    }
    return donor_id, receiver_id, torch.stack(residuals), torch.stack(edge_masks), torch.stack(proxies), metrics, report


def make_real_g5_validation_batch(config_path: str, device: str, volume_id: str) -> Dict[str, torch.Tensor | List[str]]:
    dataset_yaml = load_dataset_config(config_path)
    dataset_root = Path(dataset_yaml.dataset.root)
    dataset_dir = dataset_root / dataset_yaml.dataset.dataset_dir
    root = dataset_dir / f"{volume_id}_ome.zarr"
    reg_path = root / "REG" / dataset_yaml.dataset.reg_level
    hr_path = root / "HR" / dataset_yaml.dataset.hr_level
    patch_size = tuple(dataset_yaml.dataset.patch_size)
    metadata = read_zarr_v3_metadata(reg_path)
    z = int(metadata["shape"][0] // 2)
    _, h, w = metadata["shape"]
    _, cy, cx = metadata["chunk_grid"]["configuration"]["chunk_shape"]
    ph, pw = patch_size
    y_chunk = (h // 2) // cy
    x_chunk = (w // 2) // cx
    y = int(y_chunk * cy + max(0, (cy - ph) // 2))
    x = int(x_chunk * cx + max(0, (cx - pw) // 2))
    mean, std = _normalization_stats(reg_path, z, y, x, dataset_yaml.normalization)
    reg_patch = _normalized_patch(reg_path, z, y, x, patch_size, mean, std, dataset_yaml.normalization)
    hr_patch = _normalized_patch(hr_path, z, y, x, patch_size, mean, std, dataset_yaml.normalization)
    adjacent_z = max(0, z - 1)
    adjacent_patch = _normalized_patch(reg_path, adjacent_z, y, x, patch_size, mean, std, dataset_yaml.normalization)

    noisy = torch.from_numpy(reg_patch).unsqueeze(0).unsqueeze(0).to(device)
    adjacent = torch.from_numpy(adjacent_patch).unsqueeze(0).unsqueeze(0).to(device)
    clean_proxy = torch.from_numpy(hr_patch).unsqueeze(0).unsqueeze(0).to(device)
    grad = compute_gradient(clean_proxy)
    edge_mask = (grad > torch.quantile(grad, 0.90)).float()
    homogeneous_mask = (grad < torch.quantile(grad, 0.50)).float()
    lesion_mask = torch.zeros_like(edge_mask)

    return {
        "noisy": noisy,
        "adjacent_noisy": adjacent,
        "clean_proxy": clean_proxy,
        "homogeneous_mask": homogeneous_mask,
        "edge_mask": edge_mask,
        "lesion_mask": lesion_mask,
        "volume_id": [volume_id],
    }


def reason_category(reason: str) -> str:
    for prefix in [
        "Relative std",
        "Edge leakage",
        "Low-frequency variance",
        "Mean shift",
        "Sliced Wasserstein Distance",
    ]:
        if reason.startswith(prefix):
            return prefix
    return "Other"


def summarize_residual_metadata(run_dir: Path) -> Tuple[Dict[str, float], Dict[str, object]]:
    metadata_path = run_dir / "residual_pools" / "residual_metadata.json"
    if not metadata_path.exists():
        return {}, {"reason_category_counts": {}}
    with open(metadata_path, "r") as f:
        items = json.load(f)
    reason_counts = Counter()
    metric_groups: Dict[str, Dict[str, List[float]]] = {
        "all": {},
        "accepted": {},
        "rejected": {},
    }
    for item in items:
        status = "accepted" if item.get("status") == "accepted" else "rejected"
        for reason in item.get("reasons", []):
            reason_counts[reason_category(str(reason))] += 1
        for key, value in item.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                metric_groups["all"].setdefault(key, []).append(float(value))
                metric_groups[status].setdefault(key, []).append(float(value))
    summary_metrics = {}
    for group_name, metric_values in metric_groups.items():
        for key, values in metric_values.items():
            summary_metrics[f"{group_name}_{key}_mean"] = float(np.mean(values))
            summary_metrics[f"{group_name}_{key}_q95"] = float(np.quantile(values, 0.95))
    return summary_metrics, {"reason_category_counts": dict(reason_counts)}


def run_residual_smoke(args: argparse.Namespace) -> int:
    prerequisite_rc = run_g1_g2_smoke(SimpleNamespace(
        config=args.config,
        thresholds=args.thresholds,
        run_id=args.run_id,
        stages="G1-G2",
        device=args.device,
        volume_id=None,
        split="train",
        seed=args.seed,
    ))
    if prerequisite_rc != 0:
        return prerequisite_rc

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    thresholds = load_thresholds_config(args.thresholds)
    protocol = load_protocol_config(args.protocol)
    run_dir = Path("runs") / args.run_id
    logger = AuditLogger(args.run_id)
    machine = TraceCTStateMachine(logger, thresholds=thresholds)
    policy_hash = compute_dict_hash({
        "config": args.config,
        "thresholds": args.thresholds,
        "protocol": args.protocol,
        "seed": args.seed,
        "max_candidates": args.max_candidates,
        "scan_multiplier": args.scan_multiplier,
        "candidate_family": args.candidate_family,
        "scale_min": args.scale_min,
        "scale_max": args.scale_max,
        "artifact_mode": args.artifact_mode,
        "include_g5": args.include_g5,
        "g5_steps": args.g5_steps,
    })

    try:
        max_candidates = args.max_candidates
        budget = protocol.stage_budgets.get(Stage.G3.value)
        if budget and budget.max_candidate_patches is not None:
            max_candidates = min(max_candidates, budget.max_candidate_patches)
        args.max_candidates = max_candidates
        donor_id, receiver_id, residuals, edge_masks, proxies, candidate_metrics, report = build_residual_candidates(args)
        pool = ResidualPool(
            run_dir=run_dir,
            thresholds=thresholds.residual_audit,
            donor_volume_ids=[donor_id],
            audit_version_hash=policy_hash,
        )
        stats = pool.add_volume_residuals(
            volume_id=donor_id,
            residuals=residuals.cpu(),
            edge_masks=edge_masks.cpu(),
            validation_hr_proxies=proxies.cpu(),
            patch_size=tuple(load_dataset_config(args.config).dataset.patch_size),
        )
        double_track = pool.get_double_track_stats()
        residual_metric_summary, residual_report = summarize_residual_metadata(run_dir)
        metrics = {
            **candidate_metrics,
            "accepted": float(stats["accepted"]),
            "rejected": float(stats["rejected"]),
            "accepted_rate": float(stats["accepted_rate"]),
            "passed_threshold": float(bool(double_track["passed_threshold"])),
            **residual_metric_summary,
        }
        report.update({"stats": stats, "double_track": double_track, **residual_report})
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "real_residual_audit.json", "w") as f:
            json.dump(report, f, indent=2)

        if not double_track["passed_threshold"]:
            reasons = [
                f"G3 residual pool failed thresholds: accepted_count={double_track['accepted_count']}, "
                f"accepted_rate={double_track['accepted_rate']:.4f}"
            ]
            log_stage(logger, Stage.G3, "fail", policy_hash, args.config, metrics=metrics, failure_reasons=reasons)
            log_stage(logger, Stage.G4, "fail", policy_hash, args.config, failure_reasons=["G4 blocked because G3 residual audit failed."])
            if args.artifact_mode == "minimal":
                cleanup_tensor_artifacts(run_dir)
            print(f"Real residual smoke failed at G3: {reasons[0]}")
            print(f"Smoke artifacts: {run_dir}")
            return 1

        log_stage(logger, Stage.G3, "pass", policy_hash, args.config, metrics=metrics)

        receiver_batch, _ = make_real_batch(args.config, args.device, "train", receiver_id)
        g4 = G4BaselineProposals(machine, device=args.device, donor_volume_ids=[donor_id])
        loss = 0.0
        for _ in range(max(1, args.g4_steps)):
            loss = float(g4.step(receiver_batch))
        log_stage(logger, Stage.G4, "pass", policy_hash, args.config, metrics={"loss": loss, "g4_steps": float(args.g4_steps)})

        # G4.5 denoising strength audit gate. This smoke path uses a controlled
        # low-risk denoising output to verify release wiring without treating the
        # bounded smoke model as a formal checkpoint. Formal training audits the
        # actual G4 checkpoint.
        g45_batch = make_real_g5_validation_batch(args.config, args.device, donor_id)
        noisy = g45_batch["noisy"]
        lesion_mask = g45_batch["lesion_mask"]
        controlled_output = noisy * (1.0 - 0.25 * (1.0 - lesion_mask))
        structure_risk_mask = torch.zeros_like(g45_batch["edge_mask"])
        g45_report = audit_denoising_strength(
            noisy=noisy,
            output=controlled_output,
            homogeneous_mask=g45_batch["homogeneous_mask"],
            edge_mask=structure_risk_mask,
            lesion_mask=lesion_mask,
            controller=DenoisingStrengthController(),
            dataset_split="real_residual_smoke_controlled",
            run_dir=run_dir,
            write_json=True,
        )
        g45_metrics = {key: float(value) for key, value in g45_report["metrics"].items()}
        g45_metrics["release_D"] = float(bool(g45_report["flags"].get("release_D", False)))
        if not g45_report["flags"].get("release_D", False):
            reasons = [str(reason) for reason in g45_report.get("failure_reasons", [])]
            log_stage(logger, Stage.G45, "fail", policy_hash, args.config, metrics=g45_metrics, failure_reasons=reasons)
            if args.artifact_mode == "minimal":
                cleanup_tensor_artifacts(run_dir)
            print(f"Real residual smoke failed at G4.5: {'; '.join(reasons)}")
            print(f"Smoke artifacts: {run_dir}")
            return 1
        log_stage(logger, Stage.G45, "pass", policy_hash, args.config, metrics=g45_metrics)

        if args.include_g5:
            try:
                g5_batch = make_real_g5_validation_batch(args.config, args.device, donor_id)
                g5 = G5ProposalQualification(machine, device=args.device)
                g5_loss = 0.0
                g5_steps = max(1, int(args.g5_steps))
                for _ in range(g5_steps):
                    g5_loss = float(g5.step(g5_batch))
                report_path = run_dir / "reports" / "proposal_qualification_report.json"
                with open(report_path, "r") as f:
                    proposal_report = json.load(f)
                metrics_g5 = {key: float(value) for key, value in proposal_report.items() if isinstance(value, (int, float, bool))}
                metrics_g5["loss"] = g5_loss
                metrics_g5["g5_steps"] = float(g5_steps)
                if proposal_report.get("passed", False):
                    log_stage(logger, Stage.G5, "pass", policy_hash, args.config, metrics=metrics_g5)
                else:
                    reasons = ["G5 proposal qualification report did not pass thresholds."]
                    log_stage(logger, Stage.G5, "fail", policy_hash, args.config, metrics=metrics_g5, failure_reasons=reasons)
                    if args.artifact_mode == "minimal":
                        cleanup_tensor_artifacts(run_dir)
                    print(f"Real residual smoke failed at G5: {reasons[0]}")
                    print(f"Smoke artifacts: {run_dir}")
                    return 1
            except Exception as exc:
                log_stage(logger, Stage.G5, "fail", policy_hash, args.config, failure_reasons=[str(exc)])
                if args.artifact_mode == "minimal":
                    cleanup_tensor_artifacts(run_dir)
                print(f"Real residual smoke failed at G5: {exc}")
                print(f"Smoke artifacts: {run_dir}")
                return 1

        if args.artifact_mode == "minimal":
            cleanup_tensor_artifacts(run_dir)
        print(f"Real residual smoke complete: {run_dir}")
        return 0
    except Exception as exc:
        log_stage(logger, Stage.G3, "fail", policy_hash, args.config, failure_reasons=[str(exc)])
        log_stage(logger, Stage.G4, "fail", policy_hash, args.config, failure_reasons=["G4 blocked because G3 residual audit errored."])
        if args.artifact_mode == "minimal":
            cleanup_tensor_artifacts(run_dir)
        print(f"Real residual smoke failed: {exc}")
        print(f"Smoke artifacts: {run_dir}")
        return 1


def main() -> None:
    raise SystemExit(run_residual_smoke(parse_args()))


if __name__ == "__main__":
    main()
