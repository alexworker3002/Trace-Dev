# Trace-Dev

TRACE-CT research code for staged self-supervised CT denoising, protocol audits, and formal training orchestration.

## Repository Layout

- `trace_ct/models/`: denoiser, context, proposal, and denoising-strength controller modules.
- `trace_ct/training/`: staged G0-G8 training logic, residual control, state machine, and rollback gates.
- `trace_ct/audit/`: data, residual, proposal, and G4.5 denoising-strength audit implementations.
- `trace_ct/cli/`: command-line entry points for protocol validation, real-data smoke tests, cleanup, and formal training.
- `trace_ct/config/`: typed configuration loading and schema validation.
- `trace_ct/data/`: dataset access utilities.
- `configs/`: default dataset, protocol, threshold, G4.5, and L40 training configs.
- `datasets/manifests/` and `datasets/splits/`: lightweight dataset metadata only.
- `references/`: theory and architecture guidance documents.
- `tests/`: unit and protocol-gating tests.

Large local artifacts are intentionally ignored: `runs/`, checkpoints, DDP logs, caches, and raw zarr data.

## Common Commands

Run tests:

```bash
conda run -n trace python -m pytest -q
```

Validate the protocol chain:

```bash
conda run -n trace python -m trace_ct.cli.main validate-protocol \
  --dataset-config configs/dataset.yaml \
  --thresholds configs/thresholds.yaml \
  --protocol configs/protocol.yaml \
  --run-id protocol_check \
  --stages G0-G8 \
  --artifact-mode minimal \
  --device cpu \
  --phantom-size 64
```

Run formal 4-GPU training after G4.5 release is confirmed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m trace_ct.cli.main train \
  --config configs/train_l40.yaml \
  --dataset-config configs/dataset.yaml \
  --thresholds configs/thresholds.yaml \
  --strength-config configs/stage_g45_strength.yaml \
  --run-id formal_l40_run \
  --device cuda \
  --artifact-mode minimal
```
