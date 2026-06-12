import argparse
from trace_ct.cli import audit_data, audit_g45_strength, clean_runs, real_data_smoke, real_late_smoke, real_residual_smoke, train, validate_protocol

def main():
    parser = argparse.ArgumentParser(description="TRACE-CT Unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Audit Data
    audit_parser = subparsers.add_parser("audit-data")
    audit_parser.add_argument("--config", type=str, required=True, help="Path to dataset.yaml")
    audit_parser.add_argument("--run-id", type=str, required=True, help="Unique Run ID")

    validate_parser = subparsers.add_parser("validate-protocol")
    validate_parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    validate_parser.add_argument("--thresholds", default="configs/thresholds.yaml")
    validate_parser.add_argument("--protocol", default="configs/protocol.yaml")
    validate_parser.add_argument("--run-id", required=True)
    validate_parser.add_argument("--stages", default="G0-G8")
    validate_parser.add_argument("--artifact-mode", choices=["minimal", "full"], default="minimal")
    validate_parser.add_argument("--max-steps", type=int, default=None)
    validate_parser.add_argument("--device", default="cpu")
    validate_parser.add_argument("--phantom-size", type=int, default=64)


    g45_parser = subparsers.add_parser("audit-g45-strength")
    g45_parser.add_argument("--config", default="configs/dataset.yaml")
    g45_parser.add_argument("--strength-config", default="configs/stage_g45_strength.yaml")
    g45_parser.add_argument("--run-id", required=True)
    g45_parser.add_argument("--device", default="cpu")
    g45_parser.add_argument("--volume-id", default=None)
    g45_parser.add_argument("--checkpoint", default=None)
    g45_parser.add_argument("--synthetic-mode", choices=["none", "identity", "oversmooth", "edge_deletion", "controlled"], default="none")
    g45_parser.add_argument("--allow-fail", action="store_true")

    clean_parser = subparsers.add_parser("clean-runs")
    clean_parser.add_argument("--runs-dir", default="runs")
    clean_parser.add_argument("--run-id", default=None)
    clean_parser.add_argument("--older-than-days", type=float, default=None)
    clean_parser.add_argument("--keep-stage-records", action="store_true")
    clean_parser.add_argument("--dry-run", action="store_true")

    smoke_parser = subparsers.add_parser("real-data-smoke")
    smoke_parser.add_argument("--config", default="configs/dataset.yaml")
    smoke_parser.add_argument("--thresholds", default="configs/thresholds.yaml")
    smoke_parser.add_argument("--run-id", required=True)
    smoke_parser.add_argument("--stages", default="G1-G2", choices=["G1", "G1-G2"])
    smoke_parser.add_argument("--device", default="cpu")
    smoke_parser.add_argument("--volume-id", default=None)
    smoke_parser.add_argument("--split", default="train")
    smoke_parser.add_argument("--seed", type=int, default=0)

    residual_parser = subparsers.add_parser("real-residual-smoke")
    residual_parser.add_argument("--config", default="configs/dataset.yaml")
    residual_parser.add_argument("--thresholds", default="configs/thresholds.yaml")
    residual_parser.add_argument("--protocol", default="configs/protocol.yaml")
    residual_parser.add_argument("--run-id", required=True)
    residual_parser.add_argument("--device", default="cpu")
    residual_parser.add_argument("--seed", type=int, default=0)
    residual_parser.add_argument("--max-candidates", type=int, default=256)
    residual_parser.add_argument("--scan-multiplier", type=int, default=8)
    residual_parser.add_argument("--candidate-family", choices=["paired_raw", "paired_highpass"], default="paired_highpass")
    residual_parser.add_argument("--scale-min", type=float, default=0.5)
    residual_parser.add_argument("--scale-max", type=float, default=2.0)
    residual_parser.add_argument("--artifact-mode", choices=["minimal", "full"], default="minimal")
    residual_parser.add_argument("--g4-steps", type=int, default=2)
    residual_parser.add_argument("--include-g5", action="store_true")
    residual_parser.add_argument("--g5-steps", type=int, default=1)
    

    late_parser = subparsers.add_parser("real-late-smoke")
    late_parser.add_argument("--config", default="configs/dataset.yaml")
    late_parser.add_argument("--thresholds", default="configs/thresholds.yaml")
    late_parser.add_argument("--protocol", default="configs/protocol.yaml")
    late_parser.add_argument("--run-id", required=True)
    late_parser.add_argument("--device", default="cpu")
    late_parser.add_argument("--seed", type=int, default=0)
    late_parser.add_argument("--max-candidates", type=int, default=256)
    late_parser.add_argument("--scan-multiplier", type=int, default=8)
    late_parser.add_argument("--candidate-family", choices=["paired_raw", "paired_highpass"], default="paired_highpass")
    late_parser.add_argument("--scale-min", type=float, default=0.5)
    late_parser.add_argument("--scale-max", type=float, default=2.0)
    late_parser.add_argument("--artifact-mode", choices=["minimal", "full"], default="minimal")
    late_parser.add_argument("--g4-steps", type=int, default=120)
    late_parser.add_argument("--g5-steps", type=int, default=100)
    late_parser.add_argument("--g6-steps", type=int, default=40)
    late_parser.add_argument("--g7-steps", type=int, default=15)
    late_parser.add_argument("--g8-steps", type=int, default=5)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", default="configs/train_l40.yaml")
    train_parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    train_parser.add_argument("--thresholds", default="configs/thresholds.yaml")
    train_parser.add_argument("--strength-config", default="configs/stage_g45_strength.yaml")
    train_parser.add_argument("--protocol", default="configs/protocol.yaml")
    train_parser.add_argument("--run-id", required=True)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--resume", default=None)
    train_parser.add_argument("--artifact-mode", choices=["minimal", "full"], default=None)
    args = parser.parse_args()
    
    if args.command == "audit-data":
        audit_data.run_g0_audit(args)
    elif args.command == "validate-protocol":
        raise SystemExit(validate_protocol.run_validation(args))
    elif args.command == "clean-runs":
        clean_runs.run_clean(args)
    elif args.command == "audit-g45-strength":
        raise SystemExit(audit_g45_strength.run_audit(args))
    elif args.command == "real-data-smoke":
        raise SystemExit(real_data_smoke.run_smoke(args))
    elif args.command == "real-residual-smoke":
        raise SystemExit(real_residual_smoke.run_residual_smoke(args))
    elif args.command == "real-late-smoke":
        raise SystemExit(real_late_smoke.run_late_smoke(args))
    elif args.command == "train":
        raise SystemExit(train.main_train(args))

if __name__ == "__main__":
    main()
