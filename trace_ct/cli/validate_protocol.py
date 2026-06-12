import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import torch

from trace_ct.audit.logger import AuditLogger
from trace_ct.audit.residual_audit import ResidualPool
from trace_ct.audit.schemas import (
    ArchitectureHashes,
    GlobalHashes,
    StageHashes,
    StagePassFailRecord,
)
from trace_ct.cli.audit_data import run_g0_audit
from trace_ct.cli.audit_g45_strength import run_audit as run_g45_audit
from trace_ct.config.defaults import (
    load_dataset_config,
    load_protocol_config,
    load_thresholds_config,
)
from trace_ct.data.phantom import SyntheticPhantom
from trace_ct.models.context import ContextEncoder
from trace_ct.models.denoiser import Denoiser
from trace_ct.models.proposal import ProposalGenerator
from trace_ct.training.stages import (
    G1MaskedBaseline,
    G2ContextGating,
    G4BaselineProposals,
    G5ProposalQualification,
    G6DynamicTargetGating,
    G7EndToEndSelfSupervised,
    G8CycleStability,
)
from trace_ct.training.state_machine import Stage, TraceCTStateMachine
from trace_ct.utils.hashing import (
    compute_architecture_hash,
    compute_code_hash,
    compute_dict_hash,
)
from trace_ct.utils.paths import setup_run_directories


STAGE_ORDER = [Stage.G0, Stage.G1, Stage.G2, Stage.G3, Stage.G4, Stage.G45, Stage.G5, Stage.G6, Stage.G7, Stage.G8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded TRACE-CT G0-G8 protocol validation.")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--thresholds", default="configs/thresholds.yaml")
    parser.add_argument("--protocol", default="configs/protocol.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", default="G0-G8", help="Stage range like G0-G8, G1-G3, or comma list G1,G2,G3.")
    parser.add_argument("--artifact-mode", choices=["minimal", "full"], default="minimal")
    parser.add_argument("--max-steps", type=int, default=None, help="Override each train stage step count.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phantom-size", type=int, default=64)
    return parser.parse_args()


def resolve_stages(spec: str) -> List[Stage]:
    spec = spec.strip()
    by_name = {stage.value: stage for stage in STAGE_ORDER}
    if "-" in spec and "," not in spec:
        start_name, end_name = [part.strip() for part in spec.split("-", 1)]
        start = STAGE_ORDER.index(by_name[start_name])
        end = STAGE_ORDER.index(by_name[end_name])
        if end < start:
            raise ValueError(f"Invalid stage range: {spec}")
        return STAGE_ORDER[start : end + 1]
    return [by_name[item.strip()] for item in spec.split(",") if item.strip()]


def make_batch(device: str, phantom_size: int, volume_id: str = "phantom_receiver") -> Dict[str, torch.Tensor | List[str]]:
    phantom = SyntheticPhantom(shape=(1, phantom_size, phantom_size), device=device)
    data = phantom.generate()
    batch: Dict[str, torch.Tensor | List[str]] = {
        key: value.unsqueeze(0) for key, value in data.items() if isinstance(value, torch.Tensor)
    }
    batch["clean_proxy"] = batch["clean"]
    batch["volume_id"] = [volume_id]
    batch["_protocol_validation"] = True
    return batch


def build_hash_context(dataset_config_path: str, protocol_path: str) -> tuple[GlobalHashes, ArchitectureHashes]:
    dataset_yaml = load_dataset_config(dataset_config_path)
    project_root = Path.cwd()
    global_hashes = GlobalHashes(
        dataset_config_hash=compute_dict_hash(dataset_yaml.dataset.model_dump()),
        normalization_config_hash=compute_dict_hash(dataset_yaml.normalization.model_dump()),
        split_config_hash=compute_dict_hash(dataset_yaml.splits.model_dump()),
        code_hash=compute_code_hash(project_root),
    )
    architecture_hashes = ArchitectureHashes(
        denoiser_architecture_hash=compute_architecture_hash(Denoiser()),
        context_architecture_hash=compute_architecture_hash(ContextEncoder()),
        proposal_architecture_hash=compute_architecture_hash(ProposalGenerator()),
    )
    return global_hashes, architecture_hashes


def log_stage(
    logger: AuditLogger,
    stage: Stage,
    status: str,
    global_hashes: GlobalHashes,
    architecture_hashes: ArchitectureHashes,
    policy_hash: str,
    metrics: Dict[str, float] | None = None,
    failure_reasons: List[str] | None = None,
) -> None:
    idx = STAGE_ORDER.index(stage)
    next_allowed = STAGE_ORDER[idx + 1].value if status == "pass" and idx + 1 < len(STAGE_ORDER) else None
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
        next_allowed_stage=next_allowed,
    )
    logger.log_stage_record(record)


def read_json_if_exists(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return {key: float(value) for key, value in data.items() if isinstance(value, (int, float))}


def run_steps(stage_obj: object, batch: Dict[str, object], steps: int) -> float:
    loss = 0.0
    for _ in range(max(1, steps)):
        loss = float(stage_obj.step(batch))  # type: ignore[attr-defined]
    return loss


def stage_budget_steps(protocol, stage: Stage, override: int | None, default: int = 1) -> int:
    if override is not None:
        return override
    budget = protocol.stage_budgets.get(stage.value)
    if budget and budget.max_steps is not None:
        return min(default, budget.max_steps)
    return default


def seed_residual_pool(run_dir: Path, thresholds, device: str, phantom_size: int) -> Dict[str, float]:
    phantom = SyntheticPhantom(shape=(1, phantom_size, phantom_size), device=device)
    data = phantom.generate()
    pool = ResidualPool(
        run_dir=run_dir,
        thresholds=thresholds.residual_audit,
        donor_volume_ids=["phantom_donor"],
        audit_version_hash=compute_dict_hash(thresholds.residual_audit.model_dump()),
    )
    residuals = torch.randn(1, phantom_size, phantom_size, device=device) * 0.1
    edge_masks = torch.zeros(1, phantom_size, phantom_size, device=device)
    validation_proxy = torch.randn(1, phantom_size, phantom_size, device=device) * 0.1
    stats = pool.add_volume_residuals(
        volume_id="phantom_donor",
        residuals=residuals.cpu(),
        edge_masks=edge_masks.cpu(),
        validation_hr_proxies=validation_proxy.cpu(),
        patch_size=(phantom_size, phantom_size),
    )
    double_track = pool.get_double_track_stats()
    return {
        "accepted": float(stats["accepted"]),
        "rejected": float(stats["rejected"]),
        "accepted_rate": float(stats["accepted_rate"]),
        "passed_threshold": float(bool(double_track["passed_threshold"])),
    }


def cleanup_minimal_artifacts(run_dir: Path) -> None:
    for filename in ["accepted_residuals.pt", "error_residuals.pt"]:
        path = run_dir / "residual_pools" / filename
        if path.exists():
            path.unlink()


def write_summary(run_dir: Path, run_id: str, stages: Iterable[Stage], artifact_mode: str) -> None:
    stage_records = {}
    for stage in stages:
        for status in ["pass", "fail"]:
            path = run_dir / "audit" / "stage_records" / f"{stage.value}_{status}.json"
            if path.exists():
                with open(path, "r") as f:
                    stage_records[stage.value] = json.load(f)
                break
    summary = {
        "run_id": run_id,
        "artifact_mode": artifact_mode,
        "stages": {key: value["status"] for key, value in stage_records.items()},
        "completed_at": str(time.time()),
    }
    with open(run_dir / "reports" / "protocol_summary.json", "w") as f:
        json.dump(summary, f, indent=2)



def stage_fallback_reasons(run_dir: Path, stage: Stage, previous_mtime: float | None) -> List[str]:
    path = run_dir / "reports" / "fallback_report.json"
    if not path.exists():
        return []
    current_mtime = path.stat().st_mtime
    if previous_mtime is not None and current_mtime <= previous_mtime:
        return []
    with open(path, "r") as f:
        report = json.load(f)
    if report.get("failed_stage") != stage.value:
        return []
    reasons = report.get("reasons", [])
    if isinstance(reasons, list):
        return [str(reason) for reason in reasons]
    return [str(reasons)]

def run_validation(args: argparse.Namespace) -> int:
    stages = resolve_stages(args.stages)
    run_dir = setup_run_directories(args.run_id)
    thresholds = load_thresholds_config(args.thresholds)
    protocol = load_protocol_config(args.protocol)
    logger = AuditLogger(args.run_id)
    machine = TraceCTStateMachine(logger, thresholds=thresholds)
    global_hashes, architecture_hashes = build_hash_context(args.dataset_config, args.protocol)
    policy_hash = compute_dict_hash(
        {
            "dataset_config": args.dataset_config,
            "thresholds": thresholds.model_dump(),
            "protocol": protocol.model_dump(),
            "stages": [stage.value for stage in stages],
            "artifact_mode": args.artifact_mode,
        }
    )

    if Stage.G0 in stages:
        run_g0_audit(SimpleNamespace(config=args.dataset_config, run_id=args.run_id))

    batch = make_batch(args.device, args.phantom_size)
    stage_objects = {
        Stage.G1: lambda: G1MaskedBaseline(machine, device=args.device),
        Stage.G2: lambda: G2ContextGating(machine, device=args.device),
        Stage.G4: lambda: G4BaselineProposals(machine, device=args.device, donor_volume_ids=["phantom_donor"]),
        Stage.G5: lambda: G5ProposalQualification(machine, device=args.device),
        Stage.G6: lambda: G6DynamicTargetGating(machine, device=args.device),
        Stage.G7: lambda: G7EndToEndSelfSupervised(machine, device=args.device),
        Stage.G8: lambda: G8CycleStability(machine, device=args.device),
    }

    for stage in stages:
        if stage == Stage.G0:
            continue
        try:

            if stage == Stage.G45:
                rc = run_g45_audit(SimpleNamespace(
                    config=args.dataset_config,
                    strength_config="configs/stage_g45_strength.yaml",
                    run_id=args.run_id,
                    device=args.device,
                    volume_id=None,
                    checkpoint=None,
                    synthetic_mode="controlled",
                    allow_fail=False,
                ))
                if rc != 0:
                    raise RuntimeError("G4.5 denoising strength audit failed.")
                continue

            if stage == Stage.G3:
                metrics = seed_residual_pool(run_dir, thresholds, args.device, args.phantom_size)
                if metrics["accepted"] <= 0:
                    raise RuntimeError("G3 residual pool produced no accepted patches.")
                log_stage(logger, stage, "pass", global_hashes, architecture_hashes, policy_hash, metrics=metrics)
                continue

            stage_obj = stage_objects[stage]()
            steps = stage_budget_steps(protocol, stage, args.max_steps)
            fallback_path = run_dir / "reports" / "fallback_report.json"
            previous_fallback_mtime = fallback_path.stat().st_mtime if fallback_path.exists() else None
            loss = run_steps(stage_obj, batch, steps)
            fallback_reasons = stage_fallback_reasons(run_dir, stage, previous_fallback_mtime)
            if fallback_reasons:
                raise RuntimeError("; ".join(fallback_reasons))
            metrics = {"loss": loss}
            if stage == Stage.G1:
                metrics.update(read_json_if_exists(run_dir / "reports" / "mask_audit.json"))
            elif stage == Stage.G2:
                metrics.update(read_json_if_exists(run_dir / "reports" / "context_audit.json"))
            elif stage == Stage.G5:
                metrics.update(read_json_if_exists(run_dir / "reports" / "proposal_qualification_report.json"))
            log_stage(logger, stage, "pass", global_hashes, architecture_hashes, policy_hash, metrics=metrics)
        except Exception as exc:
            log_stage(
                logger,
                stage,
                "fail",
                global_hashes,
                architecture_hashes,
                policy_hash,
                failure_reasons=[str(exc)],
            )
            if args.artifact_mode == "minimal":
                cleanup_minimal_artifacts(run_dir)
            write_summary(run_dir, args.run_id, stages, args.artifact_mode)
            print(f"Protocol validation failed at {stage.value}: {exc}")
            print(f"Validation artifacts: {run_dir}")
            return 1

    if args.artifact_mode == "minimal":
        cleanup_minimal_artifacts(run_dir)
    write_summary(run_dir, args.run_id, stages, args.artifact_mode)
    print(f"Protocol validation complete: {run_dir}")
    return 0


def main() -> None:
    raise SystemExit(run_validation(parse_args()))


if __name__ == "__main__":
    main()
