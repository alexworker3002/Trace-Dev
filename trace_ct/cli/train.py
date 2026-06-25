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
    parser.add_argument("--strict-rollback", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--final-strength-audit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--g6-denoiser-lr", type=float, default=None)
    parser.add_argument("--g7-denoiser-lr", type=float, default=None)
    parser.add_argument("--g7-proposal-lr", type=float, default=None)
    parser.add_argument("--g8-denoiser-lr", type=float, default=None)
    parser.add_argument("--g8-proposal-lr", type=float, default=None)
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


def _get_stage_machine(stage_obj) -> TraceCTStateMachine | None:
    if hasattr(stage_obj, "state_machine"):
        return stage_obj.state_machine
    if hasattr(stage_obj, "g7_stage") and hasattr(stage_obj.g7_stage, "state_machine"):
        return stage_obj.g7_stage.state_machine
    return None


def _raise_on_rollback(stage_obj, stage_name: str) -> None:
    machine = _get_stage_machine(stage_obj)
    if machine is None:
        return
    failed_stage, reasons = machine.get_active_rollback()
    if failed_stage is not None:
        reason_text = "; ".join(reasons) if reasons else "unknown rollback reason"
        raise RuntimeError(f"{stage_name} stopped after {failed_stage.value} rollback: {reason_text}")


def _resolve_bool(cli_value: bool | None, cfg: Dict, key: str, default: bool) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    return bool(cfg.get(key, default))


def _resolve_float(cli_value: float | None, cfg: Dict, key: str, default: float) -> float:
    if cli_value is not None:
        return float(cli_value)
    return float(cfg.get(key, default))


def _configure_g6_optimizer(stage_obj, lr: float) -> None:
    stage_obj.optimizer = torch.optim.Adam(stage_obj.denoiser.parameters(), lr=lr)


def _configure_g7_optimizers(stage_obj, denoiser_lr: float, proposal_lr: float) -> None:
    target = stage_obj.g7_stage if hasattr(stage_obj, "g7_stage") else stage_obj
    target.opt_d = torch.optim.Adam(target.denoiser.parameters(), lr=denoiser_lr)
    target.opt_p = torch.optim.Adam(target.proposal_generator.parameters(), lr=proposal_lr)


def _configure_input_noise(stage_obj, cfg: Dict, stage_name: str) -> None:
    if not bool(cfg.get("enabled", False)):
        return
    stages = [str(item).upper() for item in cfg.get("stages", [])]
    if stages and stage_name.upper() not in stages:
        return
    stage_obj.input_noise = {
        "enabled": True,
        "type": str(cfg.get("type", "poisson")),
        "peak": float(cfg.get("peak", 80.0)),
        "strength": float(cfg.get("strength", 1.0)),
    }


def _has_trainable_params(module: torch.nn.Module) -> bool:
    return any(param.requires_grad for param in module.parameters())


def _ddp_if_trainable(module: torch.nn.Module, local_rank: int):
    from torch.nn.parallel import DistributedDataParallel as DDP

    if _has_trainable_params(module):
        return DDP(module, device_ids=[local_rank])
    return module


def move_batch_to_device(batch: Dict, device: str) -> Dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def run_steps(stage_obj, batch, steps: int, stage_name: str, loss_history: dict, world_size: int = 1, strict_rollback: bool = True) -> float:
    last_loss = 0.0
    for _ in range(max(1, int(steps))):
        loss_val = float(stage_obj.step(batch))
        if strict_rollback:
            _raise_on_rollback(stage_obj, stage_name)
        if world_size > 1:
            loss_tensor = torch.tensor(loss_val, device=stage_obj.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_val = float((loss_tensor / world_size).item())
        loss_history[stage_name].append(loss_val)
        last_loss = loss_val
    return last_loss


def run_steps_loader(stage_obj, dataloader, steps: int, stage_name: str, loss_history: dict, world_size: int = 1, strict_rollback: bool = True) -> float:
    last_loss = 0.0
    data_iter = iter(dataloader)
    for step_idx in range(max(1, int(steps))):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        loss_val = float(stage_obj.step(batch))
        if strict_rollback:
            _raise_on_rollback(stage_obj, stage_name)
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


def copy_module_state(source_stage: object, target_stage: object, module_name: str) -> None:
    source = source_stage.g7_stage if hasattr(source_stage, "g7_stage") else source_stage
    target = target_stage.g7_stage if hasattr(target_stage, "g7_stage") else target_stage
    if not hasattr(source, module_name) or not hasattr(target, module_name):
        return
    s_module = getattr(source, module_name)
    t_module = getattr(target, module_name)
    if s_module is None or t_module is None:
        return
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


def audit_g45_from_stage(
    stage_obj,
    batch,
    run_dir: Path,
    strength_config: str,
    checkpoint_name: str,
    dataset_split: str = "formal_training_g45",
) -> dict:
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
        dataset_split=dataset_split,
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
    training_options = train_cfg.get("training", {})
    optimizer_options = train_cfg.get("optimizer", {})
    input_noise_options = train_cfg.get("input_noise", {})
    strict_rollback = _resolve_bool(args.strict_rollback, training_options, "strict_rollback", True)
    final_strength_audit = _resolve_bool(args.final_strength_audit, training_options, "final_strength_audit", True)
    allow_unreleased_denoiser = bool(training_options.get("allow_unreleased_denoiser", False))
    post_g6_strength_audit = bool(training_options.get("post_g6_strength_audit", True))
    g6_denoiser_lr = _resolve_float(args.g6_denoiser_lr, optimizer_options, "g6_denoiser_lr", 1.0e-3)
    g7_denoiser_lr = _resolve_float(args.g7_denoiser_lr, optimizer_options, "g7_denoiser_lr", 1.0e-3)
    g7_proposal_lr = _resolve_float(args.g7_proposal_lr, optimizer_options, "g7_proposal_lr", 5.0e-4)
    g8_denoiser_lr = _resolve_float(args.g8_denoiser_lr, optimizer_options, "g8_denoiser_lr", 7.0e-4)
    g8_proposal_lr = _resolve_float(args.g8_proposal_lr, optimizer_options, "g8_proposal_lr", 3.0e-4)

    run_dir = setup_run_directories(args.run_id)
    logger = AuditLogger(args.run_id)
    thresholds = load_thresholds_config(args.thresholds)
    machine = TraceCTStateMachine(logger, thresholds=thresholds)
    machine.allow_unreleased_denoiser = allow_unreleased_denoiser
    policy_hash = compute_dict_hash({
        "train_config": train_cfg,
        "dataset_config": args.dataset_config,
        "world_size": world_size,
        "strict_rollback": strict_rollback,
        "final_strength_audit": final_strength_audit,
        "allow_unreleased_denoiser": allow_unreleased_denoiser,
        "post_g6_strength_audit": post_g6_strength_audit,
        "input_noise": input_noise_options,
        "closed_loop_lrs": {
            "g6_denoiser_lr": g6_denoiser_lr,
            "g7_denoiser_lr": g7_denoiser_lr,
            "g7_proposal_lr": g7_proposal_lr,
            "g8_denoiser_lr": g8_denoiser_lr,
            "g8_proposal_lr": g8_proposal_lr,
        },
    })

    if is_main(rank):
        run_g0_audit(SimpleNamespace(config=args.dataset_config, run_id=args.run_id))
    if world_size > 1:
        dist.barrier()

    data_cfg = train_cfg.get("data", {})
    input_mode = str(data_cfg.get("input_mode", "patch"))
    residual_input_mode = str(data_cfg.get("residual_input_mode", "patch"))
    residual_patches_per_slice = int(data_cfg.get("residual_patches_per_slice", 1))
    patches_per_volume = int(data_cfg.get("patches_per_volume", 256))
    slices_per_volume = int(data_cfg.get("slices_per_volume", 16))
    if input_mode == "full_slice" and residual_input_mode == "full_slice":
        thresholds.residual_audit.min_accepted_count = min(
            thresholds.residual_audit.min_accepted_count,
            max(1, slices_per_volume // 2),
        )
        thresholds.residual_audit.min_accepted_rate = max(
            thresholds.residual_audit.min_accepted_rate,
            0.5,
        )
    
    # Initialize datasets and dataloaders for large-scale training
    train_dataset = TraceCTDataset(
        args.dataset_config,
        split="train",
        patches_per_volume=patches_per_volume,
        input_mode=input_mode,
        slices_per_volume=slices_per_volume,
    )
    val_dataset = TraceCTDataset(
        args.dataset_config,
        split="val",
        patches_per_volume=patches_per_volume,
        input_mode=input_mode,
        slices_per_volume=slices_per_volume,
    )
    
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
        
    if input_mode == "full_slice":
        audit_slices_per_volume = min(4, slices_per_volume)
        audit_dataset = TraceCTDataset(
            args.dataset_config,
            split="val",
            patches_per_volume=patches_per_volume,
            input_mode="full_slice",
            slices_per_volume=audit_slices_per_volume,
        )
        g5_batch = move_batch_to_device(next(iter(DataLoader(audit_dataset, batch_size=audit_slices_per_volume, shuffle=False, num_workers=0))), device)
    else:
        g5_batch = make_real_g5_validation_batch(args.dataset_config, device, "10")
    steps = training_options
    
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
        # G1 Stage
        g1 = G1MaskedBaseline(machine, device=device)
        _configure_input_noise(g1, input_noise_options, "G1")
        if world_size > 1:
            g1.denoiser = _ddp_if_trainable(g1.denoiser, local_rank)
            g1.optimizer = torch.optim.Adam(g1.denoiser.parameters(), lr=1e-3)
        loss = run_steps_loader(g1, train_loader, steps.get("g1_steps", 20), "G1", loss_history, world_size, strict_rollback)
        if is_main(rank):
            log_stage(logger, Stage.G1, "pass", policy_hash, args.dataset_config, {"loss": loss})
            save_training_plots(run_dir, loss_history, g5_batch, g1.denoiser, None, device, dataset_config_path=args.dataset_config)
        if world_size > 1:
            dist.barrier()

        # G2 Stage
        g2 = G2ContextGating(machine, device=device)
        _configure_input_noise(g2, input_noise_options, "G2")
        handoff_state(g1, g2)
        if world_size > 1:
            g2.denoiser = _ddp_if_trainable(g2.denoiser, local_rank)
            g2.context_encoder = _ddp_if_trainable(g2.context_encoder, local_rank)
            g2.optimizer = torch.optim.Adam(
                list(g2.denoiser.parameters()) + list(g2.context_encoder.parameters()), lr=1e-3
            )
        loss = run_steps_loader(g2, train_loader, steps.get("g2_steps", 20), "G2", loss_history, world_size, strict_rollback)
        if is_main(rank):
            log_stage(logger, Stage.G2, "pass", policy_hash, args.dataset_config, {"loss": loss})
            save_training_plots(run_dir, loss_history, g5_batch, g2.denoiser, g2.context_encoder, device, dataset_config_path=args.dataset_config)
        if world_size > 1:
            dist.barrier()

        # Bootstrap G3 residual pool
        donor_id = None
        g3_passed = torch.tensor(0.0, device=device)
        g3_error = ""
        if is_main(rank):
            try:
                g3_max_candidates = slices_per_volume if residual_input_mode == "full_slice" else 256
                g3_scale_min = 0.0 if residual_input_mode == "full_slice" else 0.5
                g3_scale_max = 0.5 if residual_input_mode == "full_slice" else 2.0
                g3_args = SimpleNamespace(config=args.dataset_config, thresholds=args.thresholds, protocol=args.protocol, run_id=args.run_id, device=device, seed=seed, max_candidates=g3_max_candidates, scan_multiplier=8, input_mode=residual_input_mode, candidate_family="paired_highpass", scale_min=g3_scale_min, scale_max=g3_scale_max, artifact_mode=artifact_mode)
                donor_id, receiver_id, residuals, edge_masks, proxies, candidate_metrics, report = build_residual_candidates(g3_args)
                pool = ResidualPool(run_dir=run_dir, thresholds=thresholds.residual_audit, donor_volume_ids=[donor_id], audit_version_hash=policy_hash)
                residual_sample_size = tuple(residuals.shape[-2:]) if residual_input_mode == "full_slice" else tuple(load_dataset_config(args.dataset_config).dataset.patch_size)
                stats = pool.add_volume_residuals(
                    volume_id=donor_id,
                    residuals=residuals.cpu(),
                    edge_masks=edge_masks.cpu(),
                    validation_hr_proxies=proxies.cpu(),
                    patch_size=residual_sample_size,
                    sample_mode=residual_input_mode,
                )
                double_track = pool.get_double_track_stats()
                if not double_track["passed_threshold"]:
                    g3_error = (
                        f"G3 residual audit failed: accepted_count={double_track['accepted_count']}, "
                        f"accepted_rate={double_track['accepted_rate']:.4f}"
                    )
                else:
                    log_stage(logger, Stage.G3, "pass", policy_hash, args.dataset_config, {**candidate_metrics, "accepted": float(stats["accepted"]), "accepted_rate": float(stats["accepted_rate"])})
                    g3_passed.fill_(1.0)
            except Exception as exc:
                g3_error = str(exc)
        if world_size > 1:
            dist.broadcast(g3_passed, src=0)
            donor_payload = [donor_id, g3_error]
            dist.broadcast_object_list(donor_payload, src=0)
            donor_id, g3_error = donor_payload
            dist.barrier()
        if g3_passed.item() == 0.0 or donor_id is None:
            raise RuntimeError(g3_error or "G3 residual audit failed before formal training.")

        # G4 Stage
        g4 = G4BaselineProposals(machine, device=device, donor_volume_ids=[donor_id])
        _configure_input_noise(g4, input_noise_options, "G4")
        g4.residual_patches_per_slice = residual_patches_per_slice
        handoff_state(g2, g4)
        if world_size > 1:
            g4.denoiser = _ddp_if_trainable(g4.denoiser, local_rank)
            g4.context_encoder = _ddp_if_trainable(g4.context_encoder, local_rank)
            g4.optimizer = torch.optim.Adam(
                list(g4.denoiser.parameters()) + list(g4.context_encoder.parameters()), lr=3e-3
            )
        loss = run_steps_loader(g4, train_loader, steps.get("g4_steps", 20), "G4", loss_history, world_size, strict_rollback)
        
        # Save checkpoints and run audit
        ckpt = save_checkpoint(run_dir, "g4_checkpoint", denoiser=g4.denoiser, context_encoder=g4.context_encoder) if is_main(rank) else None
        
        g45_passed = torch.tensor(0.0, device=device)
        if is_main(rank):
            g4_metrics = {"loss": loss}
            g4_metrics.update(getattr(g4, "last_metrics", {}))
            log_stage(logger, Stage.G4, "pass", policy_hash, args.dataset_config, g4_metrics)
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
            if allow_unreleased_denoiser:
                if is_main(rank):
                    print(f"Exploratory training continues after G4.5 fail: {run_dir}")
            else:
                if is_main(rank):
                    print(f"Formal training stopped at G4.5: {run_dir}")
                return 1

        # G5 Stage
        g5 = G5ProposalQualification(machine, device=device)
        handoff_state(g4, g5)
        if world_size > 1:
            g5.proposal_generator = _ddp_if_trainable(g5.proposal_generator, local_rank)
            g5.optimizer = torch.optim.Adam(g5.proposal_generator.parameters(), lr=1e-3)
        loss = run_steps_loader(g5, val_loader, steps.get("g5_steps", 100), "G5", loss_history, world_size, strict_rollback)
        if is_main(rank):
            g5_metrics = {"loss": loss}
            if allow_unreleased_denoiser:
                g5_metrics["exploratory_unreleased_denoiser"] = 1.0
            log_stage(logger, Stage.G5, "pass", policy_hash, args.dataset_config, g5_metrics)
            save_training_plots(run_dir, loss_history, g5_batch, g4.denoiser, g5.context_encoder, device, dataset_config_path=args.dataset_config)
        if world_size > 1:
            dist.barrier()

        # G6 Stage
        g6 = G6DynamicTargetGating(machine, device=device)
        copy_module_state(g4, g6, "denoiser")
        copy_module_state(g4, g6, "context_encoder")
        copy_module_state(g5, g6, "proposal_generator")
        if world_size > 1:
            g6.denoiser = _ddp_if_trainable(g6.denoiser, local_rank)
            g6.context_encoder = _ddp_if_trainable(g6.context_encoder, local_rank)
        _configure_g6_optimizer(g6, g6_denoiser_lr)
        loss = run_steps_loader(g6, val_loader, steps.get("g6_steps", 20), "G6", loss_history, world_size, strict_rollback)
        g6_strength_passed = torch.tensor(1.0, device=device)
        if is_main(rank):
            g6_metrics = {"loss": loss}
            log_g6_pass = True
            if post_g6_strength_audit:
                g6_report = audit_g45_from_stage(
                    g6,
                    g5_batch,
                    run_dir,
                    args.strength_config,
                    "g6_uncheckpointed",
                    dataset_split="formal_training_g6_g45",
                )
                for key, value in g6_report["metrics"].items():
                    g6_metrics[f"post_g6_{key}"] = float(value)
                g6_release = bool(g6_report["flags"].get("release_D", False))
                g6_metrics["post_g6_release_D"] = float(g6_release)
                if not g6_release:
                    reasons = [str(r) for r in g6_report.get("failure_reasons", [])]
                    log_stage(logger, Stage.G45, "fail", policy_hash, args.dataset_config, g6_metrics, reasons)
                    if allow_unreleased_denoiser:
                        g6_metrics["exploratory_post_g6_strength_fail_continue"] = 1.0
                    else:
                        log_stage(logger, Stage.G6, "fail", policy_hash, args.dataset_config, g6_metrics, ["Post-G6 G4.5 audit failed"] + reasons)
                        g6_strength_passed.fill_(0.0)
                        log_g6_pass = False
            if log_g6_pass:
                log_stage(logger, Stage.G6, "pass", policy_hash, args.dataset_config, g6_metrics)
            save_training_plots(run_dir, loss_history, g5_batch, g6.denoiser, g6.context_encoder, device, dataset_config_path=args.dataset_config)
        if world_size > 1:
            dist.broadcast(g6_strength_passed, src=0)
            dist.barrier()
        if g6_strength_passed.item() == 0.0:
            if is_main(rank):
                print(f"Formal training stopped at post-G6 G4.5 audit: {run_dir}")
            return 1
 
        # G7 Stage
        g7 = G7EndToEndSelfSupervised(machine, device=device)
        copy_module_state(g6, g7, "denoiser")
        copy_module_state(g6, g7, "context_encoder")
        copy_module_state(g5, g7, "proposal_generator")
        if world_size > 1:
            g7.denoiser = _ddp_if_trainable(g7.denoiser, local_rank)
            g7.context_encoder = _ddp_if_trainable(g7.context_encoder, local_rank)
            g7.proposal_generator = _ddp_if_trainable(g7.proposal_generator, local_rank)
        _configure_g7_optimizers(g7, g7_denoiser_lr, g7_proposal_lr)
        loss = run_steps_loader(g7, val_loader, steps.get("g7_steps", 5), "G7", loss_history, world_size, strict_rollback)
        if is_main(rank):
            log_stage(logger, Stage.G7, "pass", policy_hash, args.dataset_config, {"loss": loss})
            save_training_plots(run_dir, loss_history, g5_batch, g7.denoiser, g7.context_encoder, device, dataset_config_path=args.dataset_config)
        if world_size > 1:
            dist.barrier()
 
        # G8 Stage
        g8 = G8CycleStability(machine, device=device)
        handoff_state(g7, g8)
        if world_size > 1:
            g8.g7_stage.denoiser = _ddp_if_trainable(g8.g7_stage.denoiser, local_rank)
            g8.g7_stage.context_encoder = _ddp_if_trainable(g8.g7_stage.context_encoder, local_rank)
            g8.g7_stage.proposal_generator = _ddp_if_trainable(g8.g7_stage.proposal_generator, local_rank)
        _configure_g7_optimizers(g8, g8_denoiser_lr, g8_proposal_lr)
        loss = run_steps_loader(g8, val_loader, steps.get("g8_steps", 5), "G8", loss_history, world_size, strict_rollback)
        final_passed = torch.tensor(1.0, device=device)
        if is_main(rank):
            final_ckpt = save_checkpoint(run_dir, "final_checkpoint", denoiser=g8.g7_stage.denoiser, proposal_generator=g8.g7_stage.proposal_generator, context_encoder=g8.g7_stage.context_encoder)
            final_metrics = {"loss": loss}
            final_reasons = []
            if final_strength_audit:
                final_report = audit_g45_from_stage(
                    g8.g7_stage,
                    g5_batch,
                    run_dir,
                    args.strength_config,
                    str(final_ckpt),
                    dataset_split="formal_training_final_g45",
                )
                for key, value in final_report["metrics"].items():
                    final_metrics[f"final_{key}"] = float(value)
                final_release = bool(final_report["flags"].get("release_D", False))
                final_metrics["final_release_D"] = float(final_release)
                if not final_release:
                    final_reasons = [str(r) for r in final_report.get("failure_reasons", [])]
                    log_stage(logger, Stage.G45, "fail", policy_hash, args.dataset_config, final_metrics, final_reasons)
                    log_stage(logger, Stage.G8, "fail", policy_hash, args.dataset_config, final_metrics, ["Final G4.5 audit failed"] + final_reasons)
                    final_passed.fill_(0.0)
            if final_passed.item() == 1.0:
                log_stage(logger, Stage.G8, "pass", policy_hash, args.dataset_config, final_metrics)
            
            # Save loss history and generate plots
            save_training_plots(run_dir, loss_history, g5_batch, g8.g7_stage.denoiser, g8.g7_stage.context_encoder, device, dataset_config_path=args.dataset_config)
            
            if artifact_mode == "minimal":
                cleanup_tensor_artifacts(run_dir)
            if final_passed.item() == 1.0:
                print(f"Formal staged training complete: {run_dir}")
            else:
                print(f"Formal staged training stopped at final G4.5 audit: {run_dir}")
        if world_size > 1:
            dist.broadcast(final_passed, src=0)
            dist.barrier()
        if final_passed.item() == 0.0:
            return 1
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
