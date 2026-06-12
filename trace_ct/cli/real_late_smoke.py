import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch

from trace_ct.audit.logger import AuditLogger
from trace_ct.audit.schemas import StageHashes, StagePassFailRecord
from trace_ct.cli.real_data_smoke import build_hash_context
from trace_ct.cli.real_residual_smoke import cleanup_tensor_artifacts, make_real_g5_validation_batch, run_residual_smoke
from trace_ct.config.defaults import load_protocol_config, load_thresholds_config
from trace_ct.training.stages import (
    G6DynamicTargetGating,
    G7EndToEndSelfSupervised,
    G8CycleStability,
    build_safe_feedback_components,
    get_denoise_gate,
)
from trace_ct.training.state_machine import Stage, TraceCTStateMachine
from trace_ct.utils.hashing import compute_dict_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-data TRACE-CT G6-G8 late-stage smoke validation after real G0-G5.")
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
    parser.add_argument("--g4-steps", type=int, default=120)
    parser.add_argument("--g5-steps", type=int, default=100)
    parser.add_argument("--g6-steps", type=int, default=40)
    parser.add_argument("--g7-steps", type=int, default=15)
    parser.add_argument("--g8-steps", type=int, default=5)
    return parser.parse_args()


def log_stage(logger: AuditLogger, stage: Stage, status: str, policy_hash: str, config_path: str, metrics: Dict[str, float] | None = None, failure_reasons: List[str] | None = None) -> None:
    global_hashes, architecture_hashes = build_hash_context(config_path)
    next_allowed = None
    if status == "pass" and stage == Stage.G6:
        next_allowed = Stage.G7.value
    elif status == "pass" and stage == Stage.G7:
        next_allowed = Stage.G8.value
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


def run_steps(stage_obj: object, batch: Dict[str, object], steps: int) -> float:
    loss = 0.0
    for _ in range(max(1, steps)):
        loss = float(stage_obj.step(batch))  # type: ignore[attr-defined]
    return loss


def _model_bundle(stage_obj: object):
    if hasattr(stage_obj, "g7_stage"):
        inner = stage_obj.g7_stage
        return inner.context_encoder, inner.proposal_generator, inner.target_aggregator, inner.denoiser
    return stage_obj.context_encoder, stage_obj.proposal_generator, stage_obj.target_aggregator, getattr(stage_obj, "denoiser", None)


def audit_dynamic_target(stage_obj: object, machine: TraceCTStateMachine, batch: Dict[str, object]) -> Dict[str, float]:
    context_encoder, proposal_generator, target_aggregator, denoiser = _model_bundle(stage_obj)
    device = next(context_encoder.parameters()).device
    noisy = batch["noisy"].to(device)  # type: ignore[index]
    adjacent = batch["adjacent_noisy"].to(device)  # type: ignore[index]
    homo_mask = batch["homogeneous_mask"].to(device)  # type: ignore[index]
    edge_mask = batch["edge_mask"].to(device)  # type: ignore[index]
    lesion_mask = batch["lesion_mask"].to(device)  # type: ignore[index]

    with torch.no_grad():
        context = context_encoder(adjacent)
        p_h, w_adj, w_safety, _, _, _, _, _ = proposal_generator(noisy, adjacent, adjacent, context)
        agg_safety, agg_adj, w_benefit, w_fb = build_safe_feedback_components(
            w_safety,
            w_adj,
            homo_mask,
            edge_mask,
            lesion_mask,
            machine.thresholds.dynamic_target_gating,
        )
        t_h = target_aggregator.aggregate(noisy, p_h, agg_safety, agg_adj, w_benefit)
        drift = target_aggregator.compute_drift_q95(t_h, noisy)
        p_target_disagreement = float((torch.norm(p_h - t_h) / (torch.norm(noisy) + 1e-8)).item())
        pred_proposal_disagreement = 0.0
        pred_proposal_feedback_disagreement = 0.0
        if denoiser is not None:
            denoise_gate = get_denoise_gate(edge_mask, lesion_mask, device, noisy.shape)
            pred = denoiser(y_h_M=noisy, x_h=noisy, p_h=p_h, c_h=context, denoise_gate=denoise_gate)
            pred_proposal_disagreement = float((torch.norm(pred - p_h) / (torch.norm(noisy) + 1e-8)).item())
            pred_proposal_feedback_disagreement = float((torch.norm((pred - p_h) * w_fb) / (torch.norm(noisy * w_fb) + 1e-8)).item())

    h_sum = homo_mask.sum() + 1e-8
    e_sum = edge_mask.sum() + 1e-8
    l_sum = lesion_mask.sum() + 1e-8
    return {
        "w_fb_mean": float(w_fb.mean().item()),
        "w_fb_homogeneous_mean": float((w_fb * homo_mask).sum().item() / h_sum.item()),
        "w_fb_edge_mean": float((w_fb * edge_mask).sum().item() / e_sum.item()),
        "w_fb_lesion_mean": float((w_fb * lesion_mask).sum().item() / l_sum.item()),
        "target_drift_q95": float(drift),
        "p_target_disagreement": p_target_disagreement,
        "pred_proposal_disagreement": pred_proposal_disagreement,
        "pred_proposal_feedback_disagreement": pred_proposal_feedback_disagreement,
    }



def handoff_state(previous_stage: object | None, current_stage: object) -> None:
    if previous_stage is None:
        return
    target = current_stage.g7_stage if hasattr(current_stage, "g7_stage") else current_stage
    source = previous_stage.g7_stage if hasattr(previous_stage, "g7_stage") else previous_stage
    names = ["denoiser", "context_encoder", "proposal_generator"]
    if source.__class__.__name__ == "G6DynamicTargetGating":
        names = []
    for name in names:
        if hasattr(source, name) and hasattr(target, name):
            getattr(target, name).load_state_dict(getattr(source, name).state_dict())

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
    return [str(reason) for reason in reasons] if isinstance(reasons, list) else [str(reasons)]


def write_summary(run_dir: Path, run_id: str, artifact_mode: str) -> None:
    stages = {}
    for stage in [Stage.G0, Stage.G1, Stage.G2, Stage.G3, Stage.G4, Stage.G45, Stage.G5, Stage.G6, Stage.G7, Stage.G8]:
        for status in ["pass", "fail"]:
            path = run_dir / "audit" / "stage_records" / f"{stage.value}_{status}.json"
            if path.exists():
                with open(path, "r") as f:
                    stages[stage.value] = json.load(f).get("status")
                break
    with open(run_dir / "reports" / "real_late_smoke_summary.json", "w") as f:
        json.dump({"run_id": run_id, "artifact_mode": artifact_mode, "stages": stages, "completed_at": str(time.time())}, f, indent=2)


def capped_steps(protocol, stage: Stage, requested: int) -> int:
    budget = protocol.stage_budgets.get(stage.value)
    if budget and budget.max_steps is not None:
        return min(requested, budget.max_steps)
    return requested


def run_late_smoke(args: argparse.Namespace) -> int:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    prerequisite_rc = run_residual_smoke(SimpleNamespace(
        config=args.config,
        thresholds=args.thresholds,
        protocol=args.protocol,
        run_id=args.run_id,
        device=args.device,
        seed=args.seed,
        max_candidates=args.max_candidates,
        scan_multiplier=args.scan_multiplier,
        candidate_family=args.candidate_family,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        artifact_mode=args.artifact_mode,
        g4_steps=args.g4_steps,
        include_g5=True,
        g5_steps=args.g5_steps,
    ))
    if prerequisite_rc != 0:
        return prerequisite_rc

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
        "g6_steps": args.g6_steps,
        "g7_steps": args.g7_steps,
        "g8_steps": args.g8_steps,
        "artifact_mode": args.artifact_mode,
    })

    with open(run_dir / "reports" / "real_residual_audit.json", "r") as f:
        donor_id = json.load(f)["donor_volume_id"]
    batch = make_real_g5_validation_batch(args.config, args.device, donor_id)
    batch["_smoke_bypass_stability"] = True

    stage_specs = [
        (Stage.G6, G6DynamicTargetGating, capped_steps(protocol, Stage.G6, args.g6_steps)),
        (Stage.G7, G7EndToEndSelfSupervised, args.g7_steps),
        (Stage.G8, G8CycleStability, capped_steps(protocol, Stage.G8, args.g8_steps)),
    ]

    previous_stage_obj = None
    for stage, cls, steps in stage_specs:
        try:
            stage_obj = cls(machine, device=args.device)
            handoff_state(previous_stage_obj, stage_obj)
            fallback_path = run_dir / "reports" / "fallback_report.json"
            previous_mtime = fallback_path.stat().st_mtime if fallback_path.exists() else None
            loss = run_steps(stage_obj, batch, steps)
            reasons = stage_fallback_reasons(run_dir, stage, previous_mtime)
            metrics = audit_dynamic_target(stage_obj, machine, batch)
            metrics["loss"] = float(loss)
            metrics["steps"] = float(steps)
            if reasons:
                log_stage(logger, stage, "fail", policy_hash, args.config, metrics=metrics, failure_reasons=reasons)
                if args.artifact_mode == "minimal":
                    cleanup_tensor_artifacts(run_dir)
                write_summary(run_dir, args.run_id, args.artifact_mode)
                print("Real late smoke failed at {}: {}".format(stage.value, "; ".join(reasons)))
                print(f"Smoke artifacts: {run_dir}")
                return 1
            log_stage(logger, stage, "pass", policy_hash, args.config, metrics=metrics)
            previous_stage_obj = stage_obj
        except Exception as exc:
            log_stage(logger, stage, "fail", policy_hash, args.config, failure_reasons=[str(exc)])
            if args.artifact_mode == "minimal":
                cleanup_tensor_artifacts(run_dir)
            write_summary(run_dir, args.run_id, args.artifact_mode)
            print(f"Real late smoke failed at {stage.value}: {exc}")
            print(f"Smoke artifacts: {run_dir}")
            return 1

    if args.artifact_mode == "minimal":
        cleanup_tensor_artifacts(run_dir)
    write_summary(run_dir, args.run_id, args.artifact_mode)
    print(f"Real late smoke complete: {run_dir}")
    return 0


def main() -> None:
    raise SystemExit(run_late_smoke(parse_args()))


if __name__ == "__main__":
    main()
