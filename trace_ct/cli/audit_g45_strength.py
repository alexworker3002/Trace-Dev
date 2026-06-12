import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from trace_ct.audit.denoising_strength_audit import audit_denoising_strength
from trace_ct.audit.logger import AuditLogger
from trace_ct.audit.schemas import StageHashes, StagePassFailRecord
from trace_ct.cli.real_data_smoke import build_hash_context
from trace_ct.cli.real_residual_smoke import make_real_g5_validation_batch
from trace_ct.data.phantom import SyntheticPhantom
from trace_ct.models.denoiser import Denoiser
from trace_ct.models.denoising_strength import DenoisingStrengthController
from trace_ct.training.stages import compute_gradient, get_denoise_gate
from trace_ct.training.state_machine import Stage
from trace_ct.utils.hashing import compute_dict_hash
from trace_ct.utils.paths import setup_run_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRACE-CT G4.5 denoising strength audit.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--strength-config", default="configs/stage_g45_strength.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--volume-id", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--synthetic-mode", choices=["none", "identity", "oversmooth", "edge_deletion", "controlled"], default="none")
    parser.add_argument("--allow-fail", action="store_true")
    return parser.parse_args()


def load_controller(path: str) -> DenoisingStrengthController:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return DenoisingStrengthController.from_mapping(data.get("thresholds", {}))


def synthetic_batch(device: str):
    phantom = SyntheticPhantom(shape=(1, 64, 64), device=device)
    data = phantom.generate()
    noisy = data["noisy"].unsqueeze(0)
    clean = data["clean"].unsqueeze(0)
    homo = data["homogeneous_mask"].unsqueeze(0)
    edge = data["edge_mask"].unsqueeze(0)
    lesion = data["lesion_mask"].unsqueeze(0)
    return noisy, clean, homo, edge, lesion


def make_output(args: argparse.Namespace):
    if args.synthetic_mode != "none":
        noisy, clean, homo, edge, lesion = synthetic_batch(args.device)
        if args.synthetic_mode == "identity":
            output = noisy.clone()
        elif args.synthetic_mode == "oversmooth":
            output = F.avg_pool2d(noisy, kernel_size=15, stride=1, padding=7)
        elif args.synthetic_mode == "edge_deletion":
            output = noisy - 0.35 * edge
        else:
            output = noisy * (1.0 - 0.25 * (1.0 - lesion))
            edge = torch.zeros_like(edge)
        return noisy, output, homo, edge, lesion, "synthetic"

    volume_id = args.volume_id or "10"
    batch = make_real_g5_validation_batch(args.config, args.device, volume_id)
    noisy = batch["noisy"]
    homo = batch["homogeneous_mask"]
    edge = batch["edge_mask"]
    lesion = batch["lesion_mask"]
    denoiser = Denoiser().to(args.device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=args.device)
        state_dict = state.get("denoiser", state) if isinstance(state, dict) else state
        denoiser.load_state_dict(state_dict, strict=False)
    with torch.no_grad():
        denoise_gate = get_denoise_gate(edge, lesion, noisy.device, noisy.shape)
        output = denoiser(y_h_M=noisy, x_h=noisy, denoise_gate=denoise_gate)
    return noisy, output, homo, edge, lesion, "validation_or_audit"


def log_g45(args: argparse.Namespace, report: dict, status: str, reasons: list[str]) -> None:
    logger = AuditLogger(args.run_id)
    global_hashes, architecture_hashes = build_hash_context(args.config)
    metrics = {key: float(value) for key, value in report["metrics"].items() if isinstance(value, (int, float))}
    metrics["release_D"] = float(bool(report["flags"].get("release_D", False)))
    record = StagePassFailRecord(
        run_id=args.run_id,
        stage=Stage.G45.value,
        status=status,  # type: ignore[arg-type]
        timestamp=str(time.time()),
        global_hashes=global_hashes,
        architecture_hashes=architecture_hashes,
        stage_hashes=StageHashes(policy_hash=compute_dict_hash({"strength_config": args.strength_config, "checkpoint": args.checkpoint, "synthetic_mode": args.synthetic_mode})),
        metrics=metrics,
        failure_reasons=reasons,
        next_allowed_stage=Stage.G5.value if status == "pass" else None,
    )
    logger.log_stage_record(record)


def run_audit(args: argparse.Namespace) -> int:
    controller = load_controller(args.strength_config)
    run_dir = setup_run_directories(args.run_id)
    noisy, output, homo, edge, lesion, split = make_output(args)
    report = audit_denoising_strength(
        noisy=noisy,
        output=output,
        homogeneous_mask=homo,
        edge_mask=edge,
        lesion_mask=lesion,
        controller=controller,
        checkpoint=args.checkpoint,
        dataset_split=split,
        run_dir=run_dir,
        write_json=True,
    )
    release = bool(report["flags"].get("release_D", False))
    status = "pass" if release else "fail"
    reasons = [str(r) for r in report.get("failure_reasons", [])]
    log_g45(args, report, status, reasons)
    print(f"G4.5 denoising strength audit {status.upper()}: {run_dir}")
    if not release and not args.allow_fail:
        return 1
    return 0


def main() -> None:
    raise SystemExit(run_audit(parse_args()))


if __name__ == "__main__":
    main()
