"""
infer_50_slices.py
------------------
Standalone inference script for TRACE-CT.

Loads the final_checkpoint.pt from a run directory, randomly samples 50 full
slices from the TRAIN split (volumes 01, 03-09), runs the three-network
pipeline (ContextEncoder → ProposalGenerator → Denoiser), and writes:

  <run_dir>/reports/infer_50/
      ├── infer_50_grid.png          # all 50 slices in a 5-col grid
      ├── infer_50_best5.png         # top-5 by ΔSSIM (denoised vs noisy)
      ├── infer_50_worst5.png        # bottom-5 by ΔSSIM
      ├── infer_50_metrics.json      # per-slice PSNR / SSIM table
      └── infer_50_summary.png       # SSIM scatter / ΔPSNR histogram

Usage (from repo root):
    python scripts/infer_50_slices.py \\
        --run-id formal_l40_train_large \\
        --device cuda            # or cpu
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Ensure repo root is on sys.path so trace_ct is importable
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from numcodecs import Blosc

from trace_ct.config.defaults import load_dataset_config
from trace_ct.data.normalization import apply_normalization, calculate_volume_statistics
from trace_ct.data.splits import load_splits
from trace_ct.models.context import ContextEncoder
from trace_ct.models.denoiser import Denoiser
from trace_ct.models.proposal import ProposalGenerator
from trace_ct.training.stages import compute_gradient


# ─────────────────────────────────────────────────────────────────────────────
# Zarr v3 I/O helpers (previously in trace_ct/cli/real_data_smoke.py)
# ─────────────────────────────────────────────────────────────────────────────

def read_zarr_v3_metadata(array_path: Path) -> Dict:
    with open(array_path / "zarr.json", "r") as f:
        return json.load(f)


def _dtype_from_zarr(data_type: str):
    return {
        "int16":   np.dtype("<i2"),
        "uint16":  np.dtype("<u2"),
        "int32":   np.dtype("<i4"),
        "uint32":  np.dtype("<u4"),
        "float32": np.dtype("<f4"),
    }[data_type]


def decode_zarr_v3_chunk(array_path: Path, chunk_coords: Tuple[int, int, int], metadata: Dict) -> np.ndarray:
    chunk_shape = tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
    dtype = _dtype_from_zarr(str(metadata["data_type"]))
    chunk_path = array_path / "c" / str(chunk_coords[0]) / str(chunk_coords[1]) / str(chunk_coords[2])
    if not chunk_path.exists():
        return np.full(chunk_shape, metadata.get("fill_value", 0), dtype=dtype)
    raw = chunk_path.read_bytes()
    blosc_cfg = next(
        (c.get("configuration", {}) for c in metadata.get("codecs", []) if c.get("name") == "blosc"),
        None,
    )
    if blosc_cfg is None:
        raise ValueError(f"No blosc codec found in {array_path}")
    shuffle = {"noshuffle": Blosc.NOSHUFFLE, "shuffle": Blosc.SHUFFLE, "bitshuffle": Blosc.BITSHUFFLE}[
        blosc_cfg.get("shuffle", "noshuffle")
    ]
    dec = Blosc(cname=blosc_cfg.get("cname", "lz4"), clevel=int(blosc_cfg.get("clevel", 3)),
                shuffle=shuffle, blocksize=int(blosc_cfg.get("blocksize", 0)))
    return np.frombuffer(dec.decode(raw), dtype=dtype).reshape(chunk_shape)


def read_full_slice(array_path: Path, z_index: int, metadata: Dict) -> np.ndarray:
    shape = tuple(metadata["shape"])
    chunk_shape = tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
    cz, cy, cx = chunk_shape
    _, height, width = shape
    z_chunk = z_index // cz
    z_in = z_index % cz
    out = np.zeros((height, width), dtype=np.float32)
    for y0 in range(0, height, cy):
        for x0 in range(0, width, cx):
            chunk = decode_zarr_v3_chunk(array_path, (z_chunk, y0 // cy, x0 // cx), metadata)
            y1 = min(y0 + cy, height)
            x1 = min(x0 + cx, width)
            out[y0:y1, x0:x1] = chunk[z_in, : y1 - y0, : x1 - x0]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse < 1e-12:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Simple full-image SSIM on [0,1] normalised arrays."""
    mu_a = a.mean()
    mu_b = b.mean()
    sig_a = a.std()
    sig_b = b.std()
    sig_ab = np.mean((a - mu_a) * (b - mu_b))
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a ** 2 + sig_b ** 2 + C2)
    return float(num / den)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def collect_candidate_slices(
    dataset_yaml,
    volume_ids: List[str],
    slices_per_volume: int,
) -> List[Tuple[str, int]]:
    """Return (volume_id, z) pairs sampled evenly from each volume."""
    dataset_root = Path(dataset_yaml.dataset.root)
    dataset_dir = dataset_root / dataset_yaml.dataset.dataset_dir
    candidates = []
    for vid in volume_ids:
        reg_path = dataset_dir / f"{vid}_ome.zarr" / "REG" / dataset_yaml.dataset.reg_level
        meta = read_zarr_v3_metadata(reg_path)
        n_z = meta["shape"][0]
        z_min, z_max = 16, n_z - 17
        zs = np.linspace(z_min, z_max, slices_per_volume, dtype=int).tolist()
        candidates.extend([(vid, int(z)) for z in zs])
    return candidates


def load_slice_triple(dataset_yaml, volume_id: str, z: int, device: str) -> Dict:
    """Load (REG noisy, adjacent REG, HR GT) for one z slice, normalised."""
    dataset_root = Path(dataset_yaml.dataset.root)
    dataset_dir = dataset_root / dataset_yaml.dataset.dataset_dir

    root_path = dataset_dir / f"{volume_id}_ome.zarr"
    reg_path = root_path / "REG" / dataset_yaml.dataset.reg_level
    hr_path  = root_path / "HR"  / dataset_yaml.dataset.hr_level

    reg_meta = read_zarr_v3_metadata(reg_path)
    n_z = reg_meta["shape"][0]

    offsets = dataset_yaml.dataset.context_offsets or [-1, 1]
    adj_z = min(max(z + offsets[0], 0), n_z - 1)

    # Use centre-chunk statistics for normalisation (consistent with training)
    chunk_shape = tuple(reg_meta["chunk_grid"]["configuration"]["chunk_shape"])
    cz, cy, cx = chunk_shape
    shape = tuple(reg_meta["shape"])
    stat_z = shape[0] // 2
    stat_y = (shape[1] // 2) // cy * cy
    stat_x = (shape[2] // 2) // cx * cx
    stat_block = decode_zarr_v3_chunk(
        reg_path,
        (stat_z // cz, stat_y // cy, stat_x // cx),
        reg_meta,
    ).astype(np.float32)
    mean, std = calculate_volume_statistics(stat_block, dataset_yaml.normalization)

    center_raw = read_full_slice(reg_path, z,     reg_meta).astype(np.float32)
    adj_raw    = read_full_slice(reg_path, adj_z, reg_meta).astype(np.float32)

    norm_center = apply_normalization(center_raw, mean, std, dataset_yaml.normalization).astype(np.float32)
    norm_adj    = apply_normalization(adj_raw,    mean, std, dataset_yaml.normalization).astype(np.float32)

    hr_meta = read_zarr_v3_metadata(hr_path)
    hr_raw  = read_full_slice(hr_path, z, hr_meta).astype(np.float32)
    norm_hr = apply_normalization(hr_raw, mean, std, dataset_yaml.normalization).astype(np.float32)

    return {
        "volume_id":    volume_id,
        "z":            z,
        "mean":         mean,
        "std":          std,
        "norm_center":  norm_center,
        "norm_adj":     norm_adj,
        "norm_hr":      norm_hr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_models(checkpoint_path: Path, device: str):
    ctx  = ContextEncoder().to(device)
    prop = ProposalGenerator().to(device)
    deno = Denoiser().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Checkpoint format: {"denoiser": state_dict, "proposal_generator": state_dict, "context_encoder": state_dict}
    # Fall back to flat prefixed format or bare model_state_dict if needed.
    if "denoiser" in ckpt and isinstance(ckpt["denoiser"], dict):
        ctx_state  = ckpt.get("context_encoder", {})
        prop_state = ckpt.get("proposal_generator", {})
        deno_state = ckpt.get("denoiser", {})
    else:
        # Fallback: flat prefixed state_dict
        state = ckpt.get("model_state_dict", ckpt)
        def extract(prefix):
            return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        ctx_state  = extract("context_encoder.")
        prop_state = extract("proposal_generator.")
        deno_state = extract("denoiser.")

    if ctx_state:
        ctx.load_state_dict(ctx_state)
    if prop_state:
        prop.load_state_dict(prop_state)
    if deno_state:
        deno.load_state_dict(deno_state)

    ctx.eval(); prop.eval(); deno.eval()
    print(f"[INFO] Checkpoint loaded: {checkpoint_path}")
    print(f"  context keys  : {len(ctx_state)}")
    print(f"  proposal keys : {len(prop_state)}")
    print(f"  denoiser keys : {len(deno_state)}")
    return ctx, prop, deno


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(ctx_enc, prop_gen, denoiser, sample: Dict, device: str) -> np.ndarray:
    """Run full 3-network pipeline on one full-slice sample."""
    noisy_t = torch.from_numpy(sample["norm_center"]).unsqueeze(0).unsqueeze(0).to(device)
    adj_t   = torch.from_numpy(sample["norm_adj"]).unsqueeze(0).unsqueeze(0).to(device)

    context = ctx_enc(adj_t)
    p_h, w_adj, w_safety, w_hom, k_str, sigma_h, A_h, g_ctx = prop_gen(
        noisy_t, adj_t, adj_t, context
    )
    grad = compute_gradient(noisy_t)
    denoise_gate = (grad < torch.quantile(grad, 0.90)).float()

    s_hat = denoiser(
        y_h_M=noisy_t,
        x_h=noisy_t,
        p_h=p_h,
        c_h=context,
        k_str=k_str,
        denoise_gate=denoise_gate,
    )
    return s_hat.squeeze().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

WINDOW = (-160, 240)   # soft-tissue HU display window


def hu_display(arr_norm: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Denormalise to HU and clip to soft-tissue window → [0,1]."""
    hu = arr_norm * std + mean
    lo, hi = WINDOW
    return np.clip((hu - lo) / (hi - lo), 0, 1)


def save_grid(results: List[Dict], out_path: Path, cols: int = 5) -> None:
    """5-panel grid: Noisy | Denoised | HR GT | Diff(D-N)×5 | Diff(D-HR)×5"""
    n = len(results)
    rows = (n + cols - 1) // cols
    PANELS = 5
    fig_w = PANELS * cols * 1.55
    fig_h = rows * 1.55 + 0.9

    fig, axes = plt.subplots(rows, cols * PANELS, figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor("#0d1117")

    col_titles = ["Noisy REG", "Denoised", "HR GT", "Diff(D-N)×5", "Diff(D-HR)×5"]
    col_cmaps  = ["gray",      "gray",     "gray",  "RdBu_r",      "RdBu_r"]

    for idx, res in enumerate(results):
        row_i = idx // cols
        col_i = idx % cols
        mean, std = res["mean"], res["std"]

        noisy_d  = hu_display(res["noisy"],    mean, std)
        denoise_d= hu_display(res["denoised"], mean, std)
        hr_d     = hu_display(res["hr"],       mean, std)
        # difference maps: amplified, centred at 0.5
        diff_dn  = np.clip((denoise_d - noisy_d)  * 5 + 0.5, 0, 1)
        diff_dhr = np.clip((denoise_d - hr_d)     * 5 + 0.5, 0, 1)
        imgs = [noisy_d, denoise_d, hr_d, diff_dn, diff_dhr]

        for panel, (img, cmap) in enumerate(zip(imgs, col_cmaps)):
            ax = axes[row_i, col_i * PANELS + panel]
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if row_i == 0 and col_i == 0:
                ax.set_title(col_titles[panel], color="white", fontsize=7, pad=2)

        delta = res["ssim_denoised_hr"] - res["ssim_noisy_hr"]
        axes[row_i, col_i * PANELS].set_ylabel(
            f"vol{res['volume_id']} z{res['z']}\nΔSSIM={delta:+.4f}",
            color="#aaaaaa", fontsize=5.5, rotation=0,
            labelpad=52, va="center"
        )

    for idx in range(n, rows * cols):
        for panel in range(PANELS):
            axes[idx // cols, (idx % cols) * PANELS + panel].set_visible(False)

    fig.suptitle("TRACE-CT Inference — 50 Random Train Slices (Soft-tissue window: −160~240 HU)",
                 color="white", fontsize=10, y=1.002)
    plt.tight_layout(pad=0.25)
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Grid saved → {out_path}")


def save_panel_comparison(results: List[Dict], out_path: Path, title: str) -> None:
    """3-row × N-col panel: Noisy / Denoised / HR GT."""
    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(n * 3.2, 9.5), dpi=130)
    fig.patch.set_facecolor("#0d1117")
    row_labels = ["Noisy REG", "Denoised", "HR GT"]
    for col, res in enumerate(results):
        mean, std = res["mean"], res["std"]
        imgs = [
            hu_display(res["noisy"],    mean, std),
            hu_display(res["denoised"], mean, std),
            hu_display(res["hr"],       mean, std),
        ]
        delta = res["ssim_denoised_hr"] - res["ssim_noisy_hr"]
        for row, (img, rl) in enumerate(zip(imgs, row_labels)):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if row == 0:
                ax.set_title(
                    f"vol{res['volume_id']} z{res['z']}\nΔSSIM={delta:+.4f}",
                    color="white", fontsize=8,
                )
            if col == 0:
                ax.set_ylabel(rl, color="#cccccc", fontsize=9)
    fig.suptitle(title, color="white", fontsize=11, y=1.01)
    plt.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Panel saved → {out_path}")


def save_summary_plot(results: List[Dict], out_path: Path) -> None:
    """SSIM scatter + per-volume ΔSSIM box-plot + ΔPSNR histogram."""
    delta_ssim = [r["ssim_denoised_hr"] - r["ssim_noisy_hr"] for r in results]
    delta_psnr = [r["psnr_denoised_hr"] - r["psnr_noisy_hr"] for r in results]
    ssim_n = [r["ssim_noisy_hr"]    for r in results]
    ssim_d = [r["ssim_denoised_hr"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=130)
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # 1. SSIM scatter
    axes[0].scatter(ssim_n, ssim_d, c="#58a6ff", alpha=0.65, s=35, edgecolors="none")
    lo = min(min(ssim_n), min(ssim_d)) - 0.003
    hi = max(max(ssim_n), max(ssim_d)) + 0.003
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1, alpha=0.5, label="no change (y=x)")
    axes[0].set_xlabel("SSIM(Noisy vs HR GT)")
    axes[0].set_ylabel("SSIM(Denoised vs HR GT)")
    axes[0].set_title("SSIM Scatter: Noisy → Denoised")
    axes[0].legend(fontsize=7, labelcolor="white", facecolor="#21262d", edgecolor="#30363d")

    # 2. ΔSSIM per volume
    vols = sorted(set(r["volume_id"] for r in results))
    data_by_vol = [[r["ssim_denoised_hr"] - r["ssim_noisy_hr"]
                    for r in results if r["volume_id"] == v] for v in vols]
    bp = axes[1].boxplot(data_by_vol, patch_artist=True,
                         medianprops={"color": "#f78166", "lw": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f6feb"); patch.set_alpha(0.6)
    for w in bp["whiskers"] + bp["caps"]: w.set_color("#8b949e")
    for fl in bp["fliers"]: fl.set(marker="o", color="#8b949e", alpha=0.4, markersize=4)
    axes[1].axhline(0, color="#f78166", lw=1, ls="--", alpha=0.7)
    axes[1].set_xticklabels([f"vol{v}" for v in vols], fontsize=8)
    axes[1].set_xlabel("Volume")
    axes[1].set_ylabel("ΔSSIM (Denoised − Noisy)")
    axes[1].set_title("ΔSSIM per Volume")

    # 3. ΔPSNR histogram
    axes[2].hist(delta_psnr, bins=15, color="#3fb950", alpha=0.85, edgecolor="#0d1117")
    axes[2].axvline(0,                    color="#f78166", lw=1.5, ls="--", label="No change")
    axes[2].axvline(np.mean(delta_psnr),  color="#ffa657", lw=1.5,
                    label=f"Mean = {np.mean(delta_psnr):+.2f} dB")
    axes[2].set_xlabel("ΔPSNR (Denoised − Noisy) dB")
    axes[2].set_ylabel("Slice count")
    axes[2].set_title("ΔPSNR Distribution (vs HR GT)")
    axes[2].legend(fontsize=7, labelcolor="white", facecolor="#21262d", edgecolor="#30363d")

    fig.suptitle(
        f"TRACE-CT 50-Slice Inference Summary  |  "
        f"ΔSSIM mean={np.mean(delta_ssim):+.4f}  "
        f"ΔPSNR mean={np.mean(delta_psnr):+.2f} dB",
        color="white", fontsize=11,
    )
    plt.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Summary saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id",   default="formal_l40_train_large")
    p.add_argument("--config",   default=None,
                   help="dataset.yaml path (default: runs/<run_id>/configs/dataset.yaml)")
    p.add_argument("--checkpoint", default=None,
                   help=".pt file (default: runs/<run_id>/checkpoints/final_checkpoint.pt)")
    p.add_argument("--split",    default="train", choices=["train", "val"])
    p.add_argument("--n-slices", type=int, default=50)
    p.add_argument("--slices-per-volume", type=int, default=10,
                   help="Candidate z positions per volume before random down-sampling")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir   = REPO_ROOT / "runs" / args.run_id
    cfg_path  = args.config     or str(run_dir / "configs" / "dataset.yaml")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
                run_dir / "checkpoints" / "final_checkpoint.pt"
    out_dir   = run_dir / "reports" / "infer_50"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Run          : {args.run_id}")
    print(f"[INFO] Config       : {cfg_path}")
    print(f"[INFO] Checkpoint   : {ckpt_path}")
    print(f"[INFO] Device       : {args.device}")
    print(f"[INFO] Split        : {args.split}")
    print(f"[INFO] Output dir   : {out_dir}")

    dataset_yaml = load_dataset_config(cfg_path)
    dataset_root = Path(dataset_yaml.dataset.root)
    splits   = load_splits(dataset_yaml.splits.split_file, dataset_root, dataset_yaml.splits)
    vol_ids  = splits.get(args.split, [])
    if not vol_ids:
        raise ValueError(f"No volumes for split '{args.split}'")
    print(f"[INFO] Volumes ({args.split}): {vol_ids}")

    candidates = collect_candidate_slices(dataset_yaml, vol_ids, args.slices_per_volume)
    print(f"[INFO] Candidate slices: {len(candidates)}")
    selected = random.sample(candidates, min(args.n_slices, len(candidates)))
    selected.sort(key=lambda x: (x[0], x[1]))
    print(f"[INFO] Sampled slices : {len(selected)}")

    ctx_enc, prop_gen, denoiser = load_models(ckpt_path, args.device)

    results = []
    for i, (vol_id, z) in enumerate(selected):
        print(f"  [{i+1:02d}/{len(selected)}] vol={vol_id} z={z} ... ", end="", flush=True)
        try:
            sample = load_slice_triple(dataset_yaml, vol_id, z, args.device)
        except Exception as e:
            print(f"SKIP (load: {e})")
            continue

        if sample["norm_center"].std() < 0.05:
            print("SKIP (air slice)")
            continue

        denoised = run_inference(ctx_enc, prop_gen, denoiser, sample, args.device)

        # Compute metrics in normalised space (rescaled to [0,1])
        noisy_n  = sample["norm_center"]
        hr_n     = sample["norm_hr"]
        lo = min(noisy_n.min(), denoised.min(), hr_n.min())
        hi = max(noisy_n.max(), denoised.max(), hr_n.max())
        rng = max(hi - lo, 1e-8)
        noisy_01  = (np.clip(noisy_n,  lo, hi) - lo) / rng
        deno_01   = (np.clip(denoised, lo, hi) - lo) / rng
        hr_01     = (np.clip(hr_n,     lo, hi) - lo) / rng

        ssim_n  = ssim(noisy_01, hr_01)
        ssim_d  = ssim(deno_01,  hr_01)
        psnr_n  = psnr(noisy_01, hr_01)
        psnr_d  = psnr(deno_01,  hr_01)

        results.append({
            "volume_id":          vol_id,
            "z":                  z,
            "mean":               float(sample["mean"]),
            "std":                float(sample["std"]),
            "noisy":              noisy_n,
            "denoised":           denoised,
            "hr":                 hr_n,
            "ssim_noisy_hr":      ssim_n,
            "ssim_denoised_hr":   ssim_d,
            "psnr_noisy_hr":      psnr_n,
            "psnr_denoised_hr":   psnr_d,
        })
        print(f"SSIM {ssim_n:.4f}→{ssim_d:.4f}  PSNR {psnr_n:.2f}→{psnr_d:.2f} dB")

    if not results:
        print("[ERROR] No slices processed.")
        sys.exit(1)

    print(f"\n[INFO] Processed: {len(results)} slices")

    # Aggregate stats
    delta_ssim = [r["ssim_denoised_hr"] - r["ssim_noisy_hr"] for r in results]
    delta_psnr = [r["psnr_denoised_hr"] - r["psnr_noisy_hr"] for r in results]
    print(f"\n{'─'*55}")
    print(f"  ΔSSIM  mean={np.mean(delta_ssim):+.4f}  std={np.std(delta_ssim):.4f}")
    print(f"  ΔPSNR  mean={np.mean(delta_psnr):+.2f} dB  std={np.std(delta_psnr):.2f}")
    print(f"  Improved (ΔSSIM>0): {sum(d > 0 for d in delta_ssim)}/{len(results)}")
    print(f"{'─'*55}\n")

    # Save metrics JSON (no ndarray)
    metrics_out = out_dir / "infer_50_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump([
            {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
            for r in results
        ], f, indent=2)
    print(f"[INFO] Metrics JSON → {metrics_out}")

    # Visualisations
    save_grid(results, out_dir / "infer_50_grid.png", cols=5)

    sorted_res = sorted(results, key=lambda r: r["ssim_denoised_hr"] - r["ssim_noisy_hr"], reverse=True)
    save_panel_comparison(
        sorted_res[:5],
        out_dir / "infer_50_best5.png",
        "Top-5 Slices by ΔSSIM (Denoised − Noisy, vs HR GT)",
    )
    save_panel_comparison(
        sorted_res[-5:],
        out_dir / "infer_50_worst5.png",
        "Bottom-5 Slices by ΔSSIM (Denoised − Noisy, vs HR GT)",
    )
    save_summary_plot(results, out_dir / "infer_50_summary.png")

    print(f"\n[DONE] All outputs → {out_dir}")


if __name__ == "__main__":
    main()
