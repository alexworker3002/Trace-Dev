import argparse
import shutil
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean TRACE-CT validation run artifacts.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--run-id", default=None, help="Clean a single run id.")
    parser.add_argument("--older-than-days", type=float, default=None)
    parser.add_argument("--keep-stage-records", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def should_clean(path: Path, run_id: str | None, older_than_days: float | None) -> bool:
    if run_id is not None:
        return path.name == run_id
    if older_than_days is None:
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds > older_than_days * 86400.0


def clean_run(path: Path, keep_stage_records: bool, dry_run: bool) -> None:
    if dry_run:
        print(f"Would clean {path}")
        return
    if keep_stage_records:
        for child in path.iterdir():
            if child.name == "audit":
                for audit_child in child.iterdir():
                    if audit_child.name != "stage_records":
                        if audit_child.is_dir():
                            shutil.rmtree(audit_child)
                        else:
                            audit_child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        shutil.rmtree(path)
    print(f"Cleaned {path}")


def run_clean(args: argparse.Namespace) -> None:
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"No runs directory found: {runs_dir}")
        return
    for run_path in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        if should_clean(run_path, args.run_id, args.older_than_days):
            clean_run(run_path, args.keep_stage_records, args.dry_run)


def main() -> None:
    run_clean(parse_args())


if __name__ == "__main__":
    main()
