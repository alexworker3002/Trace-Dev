import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml

from trace_ct.audit.denoising_strength_audit import audit_denoising_strength
from trace_ct.audit.logger import AuditLogger
from trace_ct.audit.schemas import StageHashes, StagePassFailRecord
from trace_ct.cli.audit_data import run_g0_audit
from trace_ct.cli.real_data_smoke import build_hash_context, make_real_batch
from trace_ct.cli.real_residual_smoke import build_residual_candidates, cleanup_tensor_artifacts, make_real_g5_validation_batch, summarize_residual_metadata
from trace_ct.config.defaults import load_dataset_config, load_thresholds_config
from trace_ct.audit.residual_audit import ResidualPool
from trace_ct.models.denoising_strength import DenoisingStrengthController
from trace_ct.training.stages import (
    G1MaskedBaseline,
    G2ContextGating,
    G4BaselineProposals,
    G5ProposalQualification,
    G6DynamicTargetGating,
    G7EndToEndSelfSupervised,
    G8CycleStability,
    get_denoise_gate,
)
from trace_ct.training.state_machine import Stage, TraceCTStateMachine
from trace_ct.utils.hashing import compute_dict_hash
from trace_ct.utils.paths import setup_run_directories
from trace_ct.data.dataset import TraceCTDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal TRACE-CT staged training entrypoint with optional torchrun DDP.")
    parser.add_argument("--config", default="configs/train_l40.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--thresholds", default="configs/thresholds.yaml")
    parser.add_argument("--strength-config", default="configs/stage_g45_strength.yaml")
    parser.add_argument("--protocol", default="configs/protocol.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--artifact-mode", choices=["minimal", "full"], default=None)
    return parser.parse_args()


def load_train_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def init_distributed(train_cfg: Dict) -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = train_cfg.get("distributed", {}).get("backend", "nccl")
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_main(rank: int) -> bool:
    return rank == 0


def log_stage(logger: AuditLogger, stage: Stage, status: str, policy_hash: str, dataset_config: str, metrics=None, failure_reasons=None) -> None:
    global_hashes, architecture_hashes = build_hash_context(dataset_config)
    order = [Stage.G0, Stage.G1, Stage.G2, Stage.G3, Stage.G4, Stage.G45, Stage.G5, Stage.G6, Stage.G7, Stage.G8]
    idx = order.index(stage)
    next_allowed = order[idx + 1].value if status == "pass" and idx + 1 < len(order) else None
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


def run_steps(stage_obj, batch, steps: int, stage_name: str, loss_history: dict, world_size: int = 1) -> float:
    last_loss = 0.0
    for _ in range(max(1, int(steps))):
        loss_val = float(stage_obj.step(batch))
        if world_size > 1:
            loss_tensor = torch.tensor(loss_val, device=stage_obj.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_val = float((loss_tensor / world_size).item())
        loss_history[stage_name].append(loss_val)
        last_loss = loss_val
    return last_loss


def run_steps_loader(stage_obj, dataloader, steps: int, stage_name: str, loss_history: dict, world_size: int = 1) -> float:
    last_loss = 0.0
    data_iter = iter(dataloader)
    for step_idx in range(max(1, int(steps))):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        loss_val = float(stage_obj.step(batch))
        if world_size > 1:
            loss_tensor = torch.tensor(loss_val, device=stage_obj.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_val = float((loss_tensor / world_size).item())
        loss_history[stage_name].append(loss_val)
        last_loss = loss_val
        
        # Unbuffered step logging
        if (step_idx + 1) % 100 == 0 or (step_idx + 1) == steps:
            if world_size == 1 or dist.get_rank() == 0:
                print(f"[Stage {stage_name}] Step {step_idx + 1}/{steps} - Loss: {loss_val:.6f}", flush=True)
                
    return last_loss


def save_checkpoint(run_dir: Path, name: str, **objects) -> Path:
    path = run_dir / "checkpoints" / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    for key, obj in objects.items():
        if hasattr(obj, "state_dict"):
            unwrap = obj.module if hasattr(obj, "module") else obj
            state[key] = unwrap.state_dict()
        elif obj is not None:
            state[key] = obj
    torch.save(state, path)
    return path


def handoff_state(previous_stage: object | None, current_stage: object) -> None:
    if previous_stage is None:
        return
    target = current_stage.g7_stage if hasattr(current_stage, "g7_stage") else current_stage
    source = previous_stage.g7_stage if hasattr(previous_stage, "g7_stage") else previous_stage
    names = ["denoiser", "context_encoder", "proposal_generator"]
    for name in names:
        if hasattr(source, name) and hasattr(target, name):
            s_module = getattr(source, name)
            t_module = getattr(target, name)
            if s_module is not None and t_module is not None:
                s_unwrap = s_module.module if hasattr(s_module, "module") else s_module
                t_unwrap = t_module.module if hasattr(t_module, "module") else t_module
                if hasattr(s_unwrap, "state_dict") and hasattr(t_unwrap, "load_state_dict"):
                    t_unwrap.load_state_dict(s_unwrap.state_dict())


def save_training_plots(run_dir: Path, loss_history: dict, batch: dict, denoiser, context_encoder, device, dataset_config_path: str = "configs/dataset.yaml") -> None:
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save loss curves plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for stage_name, losses in loss_history.items():
            if losses:
                plt.plot(losses, label=f"Stage {stage_name}")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("TRACE-CT Staged Training Loss Curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(report_dir / "loss_curves.png", dpi=150)
        plt.close()
    except Exception as exc:
        print(f"Failed to save loss curves plot: {exc}")

    # 2. Save denoising comparison plot using a full middle slice from the validation volume "10"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from trace_ct.cli.real_data_smoke import decode_zarr_v3_chunk, read_zarr_v3_metadata
        from trace_ct.data.normalization import apply_normalization, calculate_volume_statistics
        from trace_ct.training.stages import compute_gradient
        
        # Load dataset config
        dataset_yaml = load_dataset_config(dataset_config_path)
        dataset_root = Path(dataset_yaml.dataset.root)
        dataset_dir = dataset_root / dataset_yaml.dataset.dataset_dir
        
        # We use validation volume "10"
        volume_id = "10"
        root = dataset_dir / f"{volume_id}_ome.zarr"
        reg_path = root / "REG" / dataset_yaml.dataset.reg_level
        hr_path = root / "HR" / dataset_yaml.dataset.hr_level
        
        # Read shapes and configure slices
        reg_meta = read_zarr_v3_metadata(reg_path)
        shape = tuple(reg_meta["shape"])
        z = shape[0] // 2  # middle slice
        
        # Load volume normalization stats from the center chunk
        cz, cy, cx = reg_meta["chunk_grid"]["configuration"]["chunk_shape"]
        y_center = shape[1] // 2
        x_center = shape[2] // 2
        chunk = decode_zarr_v3_chunk(reg_path, (z // cz, y_center // cy, x_center // cx), reg_meta).astype(np.float32)
        mean, std = calculate_volume_statistics(chunk, dataset_yaml.normalization)
        
        # Helper to read a full slice from chunks
        def read_full_slice(array_path: Path, z_index: int, metadata: dict) -> np.ndarray:
            cz_s, cy_s, cx_s = metadata["chunk_grid"]["configuration"]["chunk_shape"]
            z_chunk = z_index // cz_s
            z_offset = z_index % cz_s
            
            full_slice = np.zeros((shape[1], shape[2]), dtype=np.float32)
            num_y_chunks = int(np.ceil(shape[1] / cy_s))
            num_x_chunks = int(np.ceil(shape[2] / cx_s))
            
            for y_chunk in range(num_y_chunks):
                for x_chunk in range(num_x_chunks):
                    chunk_data = decode_zarr_v3_chunk(array_path, (z_chunk, y_chunk, x_chunk), metadata)
                    slice_patch = chunk_data[z_offset]
                    y_start = y_chunk * cy_s
                    x_start = x_chunk * cx_s
                    y_end = min(y_start + cy_s, shape[1])
                    x_end = min(x_start + cx_s, shape[2])
                    
                    valid_h = y_end - y_start
                    valid_w = x_end - x_start
                    full_slice[y_start:y_end, x_start:x_end] = slice_patch[:valid_h, :valid_w]
            return full_slice
            
        reg_slice = read_full_slice(reg_path, z, reg_meta)
        adjacent_slice = read_full_slice(reg_path, max(0, z - 1), reg_meta)
        
        hr_meta = read_zarr_v3_metadata(hr_path)
        hr_slice = read_full_slice(hr_path, z, hr_meta)
        
        # Apply normalization
        norm_reg = apply_normalization(reg_slice, mean, std, dataset_yaml.normalization).astype(np.float32)
        norm_adjacent = apply_normalization(adjacent_slice, mean, std, dataset_yaml.normalization).astype(np.float32)
        norm_hr = apply_normalization(hr_slice, mean, std, dataset_yaml.normalization).astype(np.float32)
        
        # Prepare inputs
        noisy = torch.from_numpy(norm_reg).unsqueeze(0).unsqueeze(0).to(device)
        adjacent = torch.from_numpy(norm_adjacent).unsqueeze(0).unsqueeze(0).to(device)
        clean_gt = torch.from_numpy(norm_hr).unsqueeze(0).unsqueeze(0).to(device)
        
        # Generate edge and lesion masks for the full slice (to get denoise gate)
        grad = compute_gradient(clean_gt)
        edge = (grad > torch.quantile(grad, 0.90)).float()
        lesion = torch.zeros_like(edge)
        
        d_unwrap = denoiser.module if hasattr(denoiser, "module") else denoiser
        c_unwrap = None
        if context_encoder is not None:
            c_unwrap = context_encoder.module if hasattr(context_encoder, "module") else context_encoder
        
        with torch.no_grad():
            if c_unwrap is not None:
                context = c_unwrap(adjacent)
            else:
                context = torch.zeros(noisy.shape[0], 16, noisy.shape[2], noisy.shape[3], device=device)
            denoise_gate = get_denoise_gate(edge, lesion, device, noisy.shape)
            s_hat = d_unwrap(y_h_M=noisy, x_h=noisy, c_h=context, denoise_gate=denoise_gate)
            
        noisy_np = noisy[0, 0].cpu().numpy()
        s_hat_np = s_hat[0, 0].cpu().numpy()
        hr_np = clean_gt[0, 0].cpu().numpy()
        noise_removed = noisy_np - s_hat_np
        gate_np = denoise_gate[0, 0].cpu().numpy()
        
        # 5-panel layout
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        im0 = axes[0].imshow(noisy_np, cmap="gray")
        axes[0].set_title("Noisy Input (REG)")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        im1 = axes[1].imshow(s_hat_np, cmap="gray")
        axes[1].set_title("Denoised Output (s_hat)")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        im2 = axes[2].imshow(hr_np, cmap="gray")
        axes[2].set_title("Ground Truth (HR)")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        im3 = axes[3].imshow(noise_removed, cmap="coolwarm")
        axes[3].set_title("Noise Removed (Input - s_hat)")
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        
        im4 = axes[4].imshow(gate_np, cmap="jet", vmin=0.0, vmax=1.0)
        axes[4].set_title("Denoise Gate Map")
        fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
        
        for ax in axes:
            ax.axis("off")
            
        plt.suptitle("TRACE-CT Denoising Quality Verification (Full Slice)")
        plt.tight_layout()
        plt.savefig(report_dir / "comparison.png", dpi=150)
        plt.close()
        print(f"Saved full-slice 5-panel comparison plot to {report_dir / 'comparison.png'}")
    except Exception as exc:
        print(f"Failed to save comparison plot: {exc}")
        import traceback
        traceback.print_exc()


def audit_g45_from_stage(stage_obj, batch, run_dir: Path, strength_config: str, checkpoint_name: str) -> dict:
    with open(strength_config, "r") as f:
        cfg = yaml.safe_load(f)
    controller = DenoisingStrengthController.from_mapping(cfg.get("thresholds", {}))
    noisy = batch["noisy"]
    adjacent = batch["adjacent_noisy"]
    homo = batch["homogeneous_mask"]
    edge = batch["edge_mask"]
    lesion = batch["lesion_mask"]
    denoiser = stage_obj.denoiser
    # Unwrap DDP denoiser / context_encoder for auditing
    d_unwrap = denoiser.module if hasattr(denoiser, "module") else denoiser
    context_encoder = getattr(stage_obj, "context_encoder", None)
    c_unwrap = context_encoder.module if (context_encoder is not None and hasattr(context_encoder, "module")) else context_encoder
    
    with torch.no_grad():
        context = c_unwrap(adjacent) if c_unwrap is not None else None
        denoise_gate = get_denoise_gate(edge, lesion, noisy.device, noisy.shape)
        output = d_unwrap(y_h_M=noisy, x_h=noisy, c_h=context, denoise_gate=denoise_gate)
    return audit_denoising_strength(
        noisy=noisy,
        output=output,
        homogeneous_mask=homo,
        edge_mask=edge,
        lesion_mask=lesion,
        controller=controller,
        checkpoint=checkpoint_name,
        dataset_split="formal_training_g45",
        run_dir=run_dir,
        write_json=True,
    )


def main_train(args: argparse.Namespace) -> int:
    train_cfg = load_train_config(args.config)
    rank, world_size, local_rank = init_distributed(train_cfg)
    seed = int(train_cfg.get("run", {}).get("seed", 0))
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)

    requested_device = args.device or train_cfg.get("run", {}).get("device", "cpu")
    if requested_device == "cuda":
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    artifact_mode = args.artifact_mode or train_cfg.get("run", {}).get("artifact_mode", "minimal")

    run_dir = setup_run_directories(args.run_id)
    logger = AuditLogger(args.run_id)
    thresholds = load_thresholds_config(args.thresholds)
    machine = TraceCTStateMachine(logger, thresholds=thresholds)
    policy_hash = compute_dict_hash({"train_config": train_cfg, "dataset_config": args.dataset_config, "world_size": world_size})

    if is_main(rank):
        run_g0_audit(SimpleNamespace(config=args.dataset_config, run_id=args.run_id))
    if world_size > 1:
        dist.barrier()

    data_cfg = train_cfg.get("data", {})
    
    # Initialize datasets and dataloaders for large-scale training
    train_dataset = TraceCTDataset(args.dataset_config, split="train", patches_per_volume=256)
    val_dataset = TraceCTDataset(args.dataset_config, split="val", patches_per_volume=256)
    
    train_sampler = DistributedSampler(train_dataset, seed=seed) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=seed) if world_size > 1 else None
    
    batch_size = int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 4))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    if train_sampler is not None:
        train_sampler.set_epoch(0)
    if val_sampler is not None:
        val_sampler.set_epoch(0)
        
    g5_batch = make_real_g5_validation_batch(args.dataset_config, device, "10")
    steps = train_cfg.get("training", {})
    
    loss_history = {
        "G1": [],
        "G2": [],
        "G4": [],
        "G5": [],
        "G6": [],
        "G7": [],
        "G8": []
    }

    try:
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        # G1 Stage
        g1 = G1MaskedBaseline(machine, device=device)
        if world_size > 1:
            g1.denoiser = DDP(g1.denoiser, device_ids=[local_rank])
            g1.optimizer = torch.optim.Adam(g1.denoiser.parameters(), lr=1e-3)
        loss = run_steps_loader(g1, train_loader, steps.get("g1_steps", 20), "G1", loss_history, world_size)
        if is_main(rank):
            log_stage(logger, Stage.G1, "pass", policy_hash, args.dataset_config, {"loss": loss})
            save_training_plots(run_dir, loss_history, g5_batch, g1.denoiser, None, device, dataset_config_path=args.dataset_config)
        if world_size > 1:
            dist.barrier()

        # G2 Stage
        g2 = G2ContextGating(machine, device=device)
        handoff_state(g1, g2)
        if world_size > 1:
            g2.denoiser = DDP(g2.denoiser, device_ids=[local_rank])
            g2.context_encoder = DDP(g2.context_encoder, device_ids=[local_rank])
            g2.optimizer = torch.optim.Adam(
                list(g2.denoiser.parameters()) + list(g2.context_encoder.parameters()), lr=1e-3
            )
        loss = run_steps_loader(g2, train_loader, steps.get("g2_steps", 20), "G2", loss_history, world_size)
        if is_main(rank):
            log_stage(logger, Stage.G2, "pass", policy_hash, args.dataset_config, {"loss": loss})
            save_training_plots(run_dir, loss_history, g5_batch, g2.denoiser, g2.context_encoder, device, dataset_config_path=args.dataset_config)
        if world_size > 1:
            dist.barrier()

        # Bootstrap G3 residual pool
        g3_args = SimpleNamespace(config=args.dataset_config, thresholds=args.thresholds, protocol=args.protocol, run_id=args.run_id, device=device, seed=seed, max_candidates=256, scan_multiplier=8, candidate_family="paired_highpass", scale_min=0.5, scale_max=2.0, artifact_mode=artifact_mode)
        donor_id, receiver_id, residuals, edge_masks, proxies, candidate_metrics, report = build_residual_candidates(g3_args)
        pool = ResidualPool(run_dir=run_dir, thresholds=thresholds.residual_audit, donor_volume_ids=[donor_id], audit_version_hash=policy_hash)
        stats = pool.add_volume_residuals(volume_id=donor_id, residuals=residuals.cpu(), edge_masks=edge_masks.cpu(), validation_hr_proxies=proxies.cpu(), patch_size=tuple(load_dataset_config(args.dataset_config).dataset.patch_size))
        double_track = pool.get_double_track_stats()
        if not double_track["passed_threshold"]:
            raise RuntimeError("G3 residual audit failed before formal training.")
        if is_main(rank):
            log_stage(logger, Stage.G3, "pass", policy_hash, args.dataset_config, {**candidate_metrics, "accepted": float(stats["accepted"]), "accepted_rate": float(stats["accepted_rate"])})
        if world_size > 1:
            dist.barrier()

        # G4 Stage
        g4 = G4BaselineProposals(machine, device=device, donor_volume_ids=[donor_id])
        handoff_state(g2, g4)
        if world_size > 1:
            g4.denoiser = DDP(g4.denoiser, device_ids=[local_rank])
            g4.context_encoder = DDP(g4.context_encoder, device_ids=[local_rank])
            g4.optimizer = torch.optim.Adam(
                list(g4.denoiser.parameters()) + list(g4.context_encoder.parameters()), lr=3e-3
            )
        loss = run_steps_loader(g4, train_loader, steps.get("g4_steps", 20), "G4", loss_history, world_size)
        
        # Save checkpoints and run audit
        ckpt = save_checkpoint(run_dir, "g4_checkpoint", denoiser=g4.denoiser, context_encoder=g4.context_encoder) if is_main(rank) else None
        
        g45_passed = torch.tensor(0.0, device=device)
        if is_main(rank):
            log_stage(logger, Stage.G4, "pass", policy_hash, args.dataset_config, {"loss": loss})
            report_g45 = audit_g45_from_stage(g4, g5_batch, run_dir, args.strength_config, str(ckpt))
            metrics = {key: float(value) for key, value in report_g45["metrics"].items()}
            metrics["release_D"] = float(bool(report_g45["flags"].get("release_D", False)))
            status = "pass" if report_g45["flags"].get("release_D", False) else "fail"
            log_stage(logger, Stage.G45, status, policy_hash, args.dataset_config, metrics, [str(r) for r in report_g45.get("failure_reasons", [])])
            save_training_plots(run_dir, loss_history, g5_batch, g4.denoiser, g4.context_encoder, device, dataset_config_path=args.dataset_config)
            if status == "pass":
                g45_passed.fill_(1.0)
        
        if world_size > 1:
            dist.broadcast(g45_passed, src=0)
            dist.barrier()
            
        if g45_passed.item() == 0.0:
            if is_main(rank):
                print(f"Formal training stopped at G4.5: {run_dir}")
            return 1

        # G5 Stage
        g5 = G5ProposalQualification(machine, device=device)
        handoff_state(g4, g5)
        if world_size > 1:
            g5.proposal_generator = DDP(g5.proposal_generator, device_ids=[local_rank])
            g5.context_encoder = DDP(g5.context_encoder, device_ids=[local_rank])
            g5.optimizer = torch.optim.Adam(
                list(g5.proposal_generator.parameters()) + list(g5.context_encoder.parameters()), lr=1e-3
            )
        loss = run_steps_loader(g5, val_loader, steps.get("g5_steps", 100), "G5", loss_history, world_size)
        if is_main(rank):
            log_stage(logger, Stage.G5, "pass", policy_hash, args.dataset_config, {"loss": loss})
            save_training_plots(run_dir, loss_history, g5_batch, g4.denoiser, g5.context_encoder, device, dataset_config_path=args.dataset_config)
        if world_size > 1:
            dist.barrier()
 
        # G6 Stage
        g6 = G6DynamicTargetGating(machine, device=device)
        handoff_state(g5, g6)
        if world_size > 1:
            g6.denoiser = DDP(g6.denoiser, device_ids=[local_rank])
            g6.context_encoder = DDP(g6.context_encoder, device_ids=[local_rank])
            g6.optimizer = torch.optim.Adam(
                list(g6.denoiser.parameters()) + list(g6.context_encoder.parameters()), lr=3e-3
            )
        loss = run_steps_loader(g6, val_loader, steps.get("g6_steps", 20), "G6", loss_history, world_size)
        if is_main(rank):
            log_stage(logger, Stage.G6, "pass", policy_hash, args.dataset_config, {"loss": loss})
            save_training_plots(run_dir, loss_history, g5_batch, g6.denoiser, g6.context_encoder, device, dataset_config_path=args.dataset_config)
        if world_size > 1:
            dist.barrier()
 
        # G7 Stage
        g7 = G7EndToEndSelfSupervised(machine, device=device)
        handoff_state(g6, g7)
        if world_size > 1:
            g7.denoiser = DDP(g7.denoiser, device_ids=[local_rank])
            g7.context_encoder = DDP(g7.context_encoder, device_ids=[local_rank])
            g7.proposal_generator = DDP(g7.proposal_generator, device_ids=[local_rank])
            g7.opt_d = torch.optim.Adam(
                list(g7.denoiser.parameters()) + list(g7.context_encoder.parameters()), lr=3e-3
            )
            g7.opt_p = torch.optim.Adam(g7.proposal_generator.parameters(), lr=2e-3)
        loss = run_steps_loader(g7, val_loader, steps.get("g7_steps", 5), "G7", loss_history, world_size)
        if is_main(rank):
            log_stage(logger, Stage.G7, "pass", policy_hash, args.dataset_config, {"loss": loss})
            save_training_plots(run_dir, loss_history, g5_batch, g7.denoiser, g7.context_encoder, device, dataset_config_path=args.dataset_config)
        if world_size > 1:
            dist.barrier()
 
        # G8 Stage
        g8 = G8CycleStability(machine, device=device)
        handoff_state(g7, g8)
        if world_size > 1:
            g8.g7_stage.denoiser = DDP(g8.g7_stage.denoiser, device_ids=[local_rank])
            g8.g7_stage.context_encoder = DDP(g8.g7_stage.context_encoder, device_ids=[local_rank])
            g8.g7_stage.proposal_generator = DDP(g8.g7_stage.proposal_generator, device_ids=[local_rank])
            g8.g7_stage.opt_d = torch.optim.Adam(
                list(g8.g7_stage.denoiser.parameters()) + list(g8.g7_stage.context_encoder.parameters()), lr=3e-3
            )
            g8.g7_stage.opt_p = torch.optim.Adam(g8.g7_stage.proposal_generator.parameters(), lr=2e-3)
        loss = run_steps_loader(g8, val_loader, steps.get("g8_steps", 5), "G8", loss_history, world_size)
        if is_main(rank):
            log_stage(logger, Stage.G8, "pass", policy_hash, args.dataset_config, {"loss": loss})
            save_checkpoint(run_dir, "final_checkpoint", denoiser=g8.g7_stage.denoiser, proposal_generator=g8.g7_stage.proposal_generator, context_encoder=g8.g7_stage.context_encoder)
            
            # Save loss history and generate plots
            save_training_plots(run_dir, loss_history, g5_batch, g8.g7_stage.denoiser, g8.g7_stage.context_encoder, device, dataset_config_path=args.dataset_config)
            
            if artifact_mode == "minimal":
                cleanup_tensor_artifacts(run_dir)
            print(f"Formal staged training complete: {run_dir}")
        return 0
    except Exception as exc:
        if is_main(rank):
            print(f"Formal staged training failed: {exc}")
            if artifact_mode == "minimal":
                cleanup_tensor_artifacts(run_dir)
        return 1
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    raise SystemExit(main_train(parse_args()))


if __name__ == "__main__":
    main()
